import os
from io import StringIO

import mlflow
import numpy as np
import pandas as pd
import sqlalchemy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from dotenv import load_dotenv
from google.cloud import storage
from mlflow.tracking import MlflowClient
from query import INSERT_BEST_MODEL, SELECT_EXIST_MODEL, UPDATE_BEST_MODEL
from ray import tune
from torch.utils.data import DataLoader, Dataset

load_dotenv()


def connect(db):
    """Returns a connection and a metadata object"""

    POSTGRES_USER = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_SERVER = os.getenv("POSTGRES_SERVER")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT")
    POSTGRES_DB = db

    url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}"

    connection = sqlalchemy.create_engine(url)

    return connection


POSTGRES_DB = os.getenv("POSTGRES_DB")
engine = connect(POSTGRES_DB)


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
        self.fc2 = torch.nn.Linear(l1, 64, bias=True)
        self.last_layer = torch.nn.Linear(64, 10, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.flatten(out)
        out = self.fc(out)
        out = self.fc2(out)
        out = self.last_layer(out)
        return out


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

    exist_model = engine.execute(
        SELECT_EXIST_MODEL.format(model_name)
    ).fetchone()

    if exist_model and exist_model.metric_score >= metric_score:
        engine.execute(
            UPDATE_BEST_MODEL.format(
                run_id, model_type, metric, metric_score, model_name
            )
        )
    else:
        engine.execute(
            INSERT_BEST_MODEL.format(
                model_name, run_id, model_type, metric, metric_score
            )
        )


def load_data_cloud(bucket_name, data_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(data_path)

    bytes_data = blob.download_as_bytes()

    s = str(bytes_data, "utf-8")

    data = StringIO(s)
    df = pd.read_csv(data)

    return df


def get_data_path_from_db(data_version, exp_name):
    select_query = """
        SELECT *
        FROM data_info
        where version = {} and exp_name = '{}'
    """
    (train_path, _, _, _), (valid_path, _, _, _) = engine.execute(
        select_query.format(data_version, exp_name)
    ).fetchall()

    return train_path, valid_path


def load_data(is_cloud, data_version, exp_name):

    if is_cloud:
        CLOUD_STORAGE_NAME = os.getenv("CLOUD_STORAGE_NAME")
        train_path, valid_path = get_data_path_from_db(data_version, exp_name)
        train_df = load_data_cloud(CLOUD_STORAGE_NAME, train_path)
        valid_df = load_data_cloud(CLOUD_STORAGE_NAME, valid_path)
    else:
        TRAIN_MNIST = os.getenv("TRAIN_MNIST")
        VALID_MNIST = os.getenv("VALID_MNIST")
        train_df = pd.read_csv(TRAIN_MNIST)
        valid_df = pd.read_csv(VALID_MNIST)

    return train_df, valid_df


def cnn_training(
    config, data_version, exp_name, checkpoint_dir=None, is_cloud=True
):
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

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_df, valid_df = load_data(is_cloud, data_version, exp_name)

    trainset = MnistDataset(train_df, transform)
    validset = MnistDataset(valid_df, transform)
    train_loader = DataLoader(trainset, batch_size=int(config["batch_size"]))
    valid_loader = DataLoader(validset, batch_size=int(config["batch_size"]))

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = Net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0

        classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        result_pred = {
            f"{classname}_acc_percentage": 0 for classname in classes
        }

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

                for label, prediction in zip(labels, predicted):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((Net.state_dict(), optimizer.state_dict()), path)

        for classname, correct_count in correct_pred.items():
            accuracy = round(
                100 * float(correct_count) / total_pred[classname], 2
            )
            result_pred[f"{classname}_acc_percentage"] = accuracy

        tune.report(
            loss=(val_loss / val_steps),
            accuracy=correct / total,
            result_pred=result_pred,
        )


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

def get_mnist_avg(df):
    return np.round(df.groupby('label').mean().mean(axis=1).values,2).tolist()