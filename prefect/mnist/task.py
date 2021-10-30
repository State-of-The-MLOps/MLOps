import os
import time
from functools import partial

import mlflow
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from model import MnistNet
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from utils import (
    MnistDataset,
    MnistNet,
    cnn_training,
    load_data,
    save_best_model,
)

from prefect import task

load_dotenv()

@task
def tune_cnn(num_samples, max_num_epochs, is_cloud):

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
        partial(cnn_training, is_cloud=is_cloud),
        resources_per_trial={"cpu": 2},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    return result


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
def make_feature_weight(results, device, is_cloud):
    best_trial = results.get_best_trial("loss", "min", "last")

    train_df, _ = load_data(is_cloud)

    configs = {
        "l1": best_trial.config["l1"],
        "lr": best_trial.config["lr"],
        "batch_size": best_trial.config["batch_size"],
    }
    best_trained_model = MnistNet(configs["l1"])
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, _ = torch.load(
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

    for i, (mini_batch, _) in enumerate(train_loader):
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
def train_knn(feature_weight_df, metric, exp_name):
    KNN = KNeighborsClassifier(n_neighbors=3)
    KNN.fit(
        feature_weight_df.iloc[:, 1:].values,
        feature_weight_df.iloc[:, 0].values,
    )

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
#     is_cloud=True

#     mlflow.set_tracking_uri(host_url)
#     mlflow.set_experiment(exp_name)

#     results = tune_cnn(num_samples, max_num_epochs, is_cloud)
#     is_end = log_experiment(results, host_url, exp_name, metric)

#     if is_end:
#         print('True')
#         feature_weight_df = make_feature_weight(results, device)
#         train_knn(feature_weight_df, metric, exp_name)
#     else:
#         print('False')
# feature_weight_df = make_feature_weight(results, device)
# train_knn(feature_weight_df, metric, exp_name)
