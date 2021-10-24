import os
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
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, Dataset, random_split

from prefect import task
from sqlalchemy import create_engine

from query import SELECT_EXIST_MODEL, INSERT_BEST_MODEL, UPDATE_BEST_MODEL

engine = create_engine(
    "postgresql://ehddnr:0000@localhost:5431/postgres"
)

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


class MnistNet(torch.nn.Module):
    def __init__(self, l1):
        super(MnistNet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(7 * 7 * 64, l1, bias=True)
        self.last_layer = torch.nn.Linear(l1, 10, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.flatten(out)
        out = self.fc(out)
        out = self.last_layer(out)
        return out


def load_data(data_path):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    df = pd.read_csv(data_path)
    train_df, valid_df = train_test_split(
        df, test_size=0.1, stratify=df["label"]
    )
    trainset = MnistDataset(train_df, transform)
    validset = MnistDataset(valid_df, transform)

    return trainset, validset


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


def cnn_training(config, checkpoint_dir=None, data_path=None):
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

    trainset, validset = load_data(data_path)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

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
def tune_cnn(num_samples, max_num_epochs, data_path):
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
        partial(cnn_training, data_path=data_path),
        resources_per_trial={"cpu": 2},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    return result

def save_best_model(model_name, run_id, model_type, metric, metric_score):
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
    client = MlflowClient()
    exp_id = client.get_experiment_by_name(exp_name).experiment_id

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

    runs = mlflow.search_runs([exp_id])
    best_score = runs[f"metrics.{metric}"].min()
    best_run = runs[runs[f'metrics.{metric}'] == best_score]

    if not best_score or best_score > metrics[metric]:
        with mlflow.start_run(experiment_id=exp_id):
            mlflow.log_metrics(metrics)
            mlflow.log_params(configs)
            mlflow.pytorch.log_model(best_trained_model, artifact_path="model")
            save_best_model(exp_name, best_run.run_id.item(), 'pytorch', metric, metrics[metric])


# if __name__ == "__main__":
#     data_path = "/Users/don/Documents/MLOps/prefect/mnist/mnist.csv"
#     host_url = "http://localhost:5001"
#     exp_name = "mnist"
#     batch_size = 64
#     learning_rate = 1e-3
#     device = "cpu"
#     l1 = 128
#     num_samples = 2
#     max_num_epochs = 1
#     metric = 'loss'

#     mlflow.set_tracking_uri(host_url)
#     mlflow.set_experiment(exp_name)

#     results = tune_cnn(num_samples, max_num_epochs, data_path)
#     log_experiment(results, host_url, exp_name, metric)
