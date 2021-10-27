import os
import time
from abc import *
from functools import partial

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from mlflow.tracking import MlflowClient
from model import MnistNet
from query import INSERT_BEST_MODEL, SELECT_EXIST_MODEL, UPDATE_BEST_MODEL
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sqlalchemy import create_engine
from torch.utils.data import DataLoader, Dataset, random_split

from prefect import task

engine = create_engine("postgresql://ehddnr:0000@localhost:5431/postgres")


class MnistDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]

        image = item[1:].values.astype(np.uint8).reshape((28, 28))
        label = item[0]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_df = pd.read_csv(
        "/Users/TFG5076XG/Documents/MLOps/prefect/mnist/mnist_train.csv"
    )
    valid_df = pd.read_csv(
        "/Users/TFG5076XG/Documents/MLOps/prefect/mnist/mnist_valid.csv"
    )
    trainset = MnistDataset(train_df, transform)
    validset = MnistDataset(valid_df, transform)

    return trainset, validset, train_df


def preprocess_train(train_df, valid_df, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = MnistDataset(train_df, transform)
    validset = MnistDataset(valid_df, transform)
    train_loader = DataLoader(trainset, batch_size=batch_size)
    valid_loader = DataLoader(validset, batch_size=batch_size)
    total_batch = len(train_loader)

    return (train_loader, valid_loader, total_batch)


def cnn_training(config, checkpoint_dir=None):
    Net = MnistNet(config["l1"])
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            Net = nn.DataParallel(Net)
    Net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(Net.parameters(), lr=config["lr"], momentum=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )
        Net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, validset, train_df = load_data()

    # test_abs = int(len(trainset) * 0.8)
    # train_subset, val_subset = random_split(
    #     trainset, [test_abs, len(trainset) - test_abs]
    # )

    train_loader = DataLoader(trainset, batch_size=int(config["batch_size"]))
    valid_loader = DataLoader(validset, batch_size=int(config["batch_size"]))

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = Net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valid_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = Net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((Net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)


@task
def tune_cnn(num_samples, max_num_epochs):

    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(7, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([64, 128, 256]),
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    result = tune.run(
        partial(cnn_training),
        resources_per_trial={"cpu": 2},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    return result


def save_best_model(
    model_name, model_type, metric, metric_score, exp_name, is_knn=False
):
    client = MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    exp_id = exp.experiment_id
    runs = mlflow.search_runs([exp_id])
    best_score = runs[f"metrics.{metric}"].min()
    best_run = runs[runs[f"metrics.{metric}"] == best_score]
    run_id = best_run.run_id.item()
    if is_knn:
        recent_knn = (
            runs[~runs["params.time"].isna()]["params.time"]
            .astype(float)
            .max()
        )
        run_id = runs[runs["params.time"] == str(recent_knn)]["run_id"].item()
        print()

    exist_model = engine.execute(
        SELECT_EXIST_MODEL.format(model_name)
    ).fetchone()

    if exist_model and exist_model.metric_score >= metric_score:
        engine.execute(
            UPDATE_BEST_MODEL.format(
                run_id, model_type, metric, metric_score, model_name
            )
        )
    else:  # 생성
        engine.execute(
            INSERT_BEST_MODEL.format(
                model_name, run_id, model_type, metric, metric_score
            )
        )


@task
def log_experiment(results, host_url, exp_name, metric):
    mlflow.set_tracking_uri(host_url)
    mlflow.set_experiment(exp_name)
    client = MlflowClient()
    exp = client.get_experiment_by_name(exp_name)

    best_trial = results.get_best_trial("loss", "min", "last")
    metrics = {
        "loss": best_trial.last_result["loss"],
        "accuracy": best_trial.last_result["accuracy"],
    }
    configs = {
        "l1": best_trial.config["l1"],
        "lr": best_trial.config["lr"],
        "batch_size": best_trial.config["batch_size"],
    }

    best_trained_model = MnistNet(configs["l1"])
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(
        os.path.join(best_checkpoint_dir, "checkpoint")
    )
    best_trained_model.load_state_dict(model_state)
    best_trained_model = torch.jit.script(best_trained_model)
    exp_id = exp.experiment_id
    runs = mlflow.search_runs([exp_id])
    if runs.empty:
        with mlflow.start_run(experiment_id=exp_id):
            mlflow.log_metrics(metrics)
            mlflow.log_params(configs)
            mlflow.pytorch.log_model(best_trained_model, artifact_path="model")

            save_best_model(
                exp_name, "pytorch", metric, metrics[metric], exp_name
            )
        return True
    else:
        best_score = runs[f"metrics.{metric}"].min()

        if best_score > metrics[metric]:
            with mlflow.start_run(experiment_id=exp_id):
                mlflow.log_metrics(metrics)
                mlflow.log_params(configs)
                mlflow.pytorch.log_model(
                    best_trained_model, artifact_path="model"
                )
                save_best_model(
                    exp_name, "pytorch", metric, metrics[metric], exp_name
                )
            return True
        else:
            return False


@task
def make_feature_weight(results, device):
    best_trial = results.get_best_trial("loss", "min", "last")
    train_df = pd.read_csv(
        "/Users/TFG5076XG/Documents/MLOps/prefect/mnist/mnist_train.csv"
    )
    configs = {
        "l1": best_trial.config["l1"],
        "lr": best_trial.config["lr"],
        "batch_size": best_trial.config["batch_size"],
    }
    best_trained_model = MnistNet(configs["l1"])
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(
        os.path.join(best_checkpoint_dir, "checkpoint")
    )
    best_trained_model.load_state_dict(model_state)
    best_trained_model = torch.nn.Sequential(
        *list(best_trained_model.children())[:-1]
    )
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = MnistDataset(train_df, transform)
    train_loader = DataLoader(trainset, batch_size=int(configs["batch_size"]))

    temp = pd.DataFrame(
        columns=[f"{i}_feature" for i in range(32)], index=train_df.index
    )
    batch_index = 0
    batch_size = train_loader.batch_size
    optimizer = torch.optim.Adam(
        best_trained_model.parameters(), lr=configs["lr"]
    )

    for i, (mini_batch, _) in enumerate(train_loader):  # 미니 배치 단위로 꺼내온다.
        mini_batch = mini_batch.to(device)
        optimizer.zero_grad()
        outputs = best_trained_model(mini_batch)
        batch_index = i * batch_size
        temp.iloc[
            batch_index : batch_index + batch_size, :
        ] = outputs.detach().numpy()

    temp.reset_index(inplace=True)
    feature_weight_df = temp

    return feature_weight_df


@task
def train_knn(feature_weight_df, metric, exp_name, results):
    KNN = KNeighborsClassifier(n_neighbors=3)
    KNN.fit(
        feature_weight_df.iloc[:, 1:].values,
        feature_weight_df.iloc[:, 0].values,
    )
    best_trial = results.get_best_trial("loss", "min", "last")
    metrics = {
        "loss": best_trial.last_result["loss"],
        "accuracy": best_trial.last_result["accuracy"],
    }
    mlflow.sklearn.log_model(KNN, artifact_path="model")
    mlflow.log_param("time", time.time())
    save_best_model("mnist_knn", "sklearn", metric, 9999, exp_name, True)


@task
def case2():
    print("end")


# def test(results):
#     aa = results.get_best_trial("loss", "min", "last")
#     print(dir(aa))
#     print(aa.trial_id)

# if __name__ == "__main__":
#     # data_path = "C:\Users\TFG5076XG\Documents\MLOps\prefect\mnist\mnist.csv"
#     host_url = "http://localhost:5001"
#     exp_name = "mnist"
#     batch_size = 64
#     learning_rate = 1e-3
#     device = "cpu"
#     l1 = 128
#     num_samples = 6
#     max_num_epochs = 2
#     metric = 'loss'

#     mlflow.set_tracking_uri(host_url)
#     mlflow.set_experiment(exp_name)

#     results = tune_cnn(num_samples, max_num_epochs)
#     is_end = log_experiment(results, host_url, exp_name, metric)

#     if is_end:
#         print('True')
#         feature_weight_df = make_feature_weight(results, device)
#         train_knn(feature_weight_df, metric, exp_name, results)
#     else:
#         print('False')
#         # feature_weight_df = make_feature_weight(results, device)
#         # train_knn(feature_weight_df, metric, exp_name, results)
