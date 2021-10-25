from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os
import sys
import time
import sqlalchemy
import pandas as pd

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from ray import tune
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from query import INSERT_BEST_MODEL, SELECT_EXIST_MODEL, UPDATE_BEST_MODEL


def connect(db):
    """Returns a connection and a metadata object"""

    load_dotenv(verbose=True)

    POSTGRES_USER = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_SERVER = os.getenv("POSTGRES_SERVER")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT")
    POSTGRES_DB = db

    url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}"

    connection = sqlalchemy.create_engine(url)

    return connection

conn = connect('postgres')


def load_to_db(data):
    data.to_sql("atmos_stn108", conn, index=False, if_exists='append')


def get_st_date():
    start_date = conn.execute(
            "SELECT time FROM atmos_stn108 ORDER BY time DESC;"
            ).fetchone()[0]
    return start_date


def get_org_data(start_date):
    org_data_query = f"""
    SELECT *
    FROM atmos_stn108
    WHERE time < '{start_date}';
    """
    org_data = pd.read_sql(org_data_query, conn)
    return org_data


def preprocess(data):
    # missing data
    data = data.fillna(method="ffill")

    # etc.

    return data


class Tuner(ABC):
    def __init__(self):
        self.model = None
        self.data_X = None
        self.data_y = None
        self.config = None


    def _split(self, test_size):
        """
        self.data_X, self.data_y 를 split
        data_X와 data_y는 상속받은 class에서 값을 받게 되어있음.
        """
        train_X, valid_X, train_y, valid_y = train_test_split(
            self.data_X,
            self.data_y,
            test_size=test_size,
        )

        return train_X, valid_X, train_y, valid_y


    def _split_ts(self, data, label, window_size=365, predsize=None):
        feature_list = []
        label_list = []

        if isinstance(predsize, int):
            for i in range(len(data) - (window_size + predsize)):
                feature_list.append(np.array(data.iloc[i : i + window_size]))
                label_list.append(
                    np.array(label.iloc[i + window_size : i + window_size + predsize])
                )
        else:
            for i in range(len(data) - window_size):
                feature_list.append(np.array(data.iloc[i : i + window_size]))
                label_list.append(np.array(label.iloc[i + window_size]))

        return np.array(feature_list), np.array(label_list)


    def _get_divided_index(self, data_length, ratio):
        """
        return index based on ratio
        --------------------------------------------------
        example

        >>> split_data(data_length = 20, ratio = [1,2,3])
        [3, 10]
        --------------------------------------------------
        """
        ratio = np.cumsum(np.array(ratio) / np.sum(ratio))

        idx = []
        for i in ratio[:-1]:
            idx.append(round(data_length * i))

        return idx


    @abstractmethod
    def exec(self):
        pass


class AtmosTuner(Tuner):
    def __init__(self, host_url, exp_name, metric):
        self.host_url = host_url
        self.exp_name = exp_name
        self.metric = metric
        self.TUNE_METRIC_DICT = {"mae": "min", 
                                 "mse": "min", 
                                 "rmse": "min"}


    def _log_experiments(self, config, metrics, tf_model):
        best_score = None
        mlflow.set_tracking_uri(self.host_url)

        client = MlflowClient()
        exp_id = client.get_experiment_by_name(self.exp_name).experiment_id
        runs = mlflow.search_runs([exp_id])

        if len(runs) > 0:
            try:
                best_score = runs[f"metrics.{self.metric}"].min()
            except Exception as e:
                print(e)

        with mlflow.start_run(experiment_id=exp_id):
            mlflow.log_metrics(metrics)
            mlflow.log_params(config)

            if not best_score or best_score > metrics[self.metric]:
                print("log model")
                input_schema = Schema([TensorSpec(np.dtype(np.float), (-1, 72, 1))])
                output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 24))])
                signature = ModelSignature(inputs=input_schema, outputs=output_schema)

                mlflow.keras.log_model(tf_model,
                                        signature = signature,
                                        artifact_path = "model")


    def _trainable(self, config):
        data = pd.read_sql("select tmp \
                            from atmos_stn108 \
                            where time > '2020-12-31 23:00';", conn)
        data = preprocess(data)
        train_feature, train_label = self._split_ts(data, data, 72, 24)

        idx = self._get_divided_index(train_feature.shape[0], [6, 3, 1])
        X_train, X_valid, X_test = (
            train_feature[: idx[0]],
            train_feature[idx[0] : idx[1]],
            train_feature[idx[1] :],
        )
        y_train, y_valid, y_test = (
            train_label[: idx[0]],
            train_label[idx[0] : idx[1]],
            train_label[idx[1] :],
        )

        model = Sequential()
        for layer in range(config["layer_n"]):
            if layer == config["layer_n"] - 1:
                model.add(GRU(config["cell"]))
            else:
                model.add(
                    GRU(
                        config["cell"],
                        return_sequences=True,
                        input_shape=[None, train_feature.shape[2]],
                    )
                )
        model.add(Dense(24))

        model.compile(loss=self.metric, optimizer=keras.optimizers.Adam(lr=0.001))
        early_stop = EarlyStopping(monitor="val_loss", patience=5)
        
        model.fit(
            X_train,
            y_train,
            epochs=2,
            batch_size=128,
            validation_data=(X_valid, y_valid),
            callbacks=[early_stop],
        )

        y_true = y_test.reshape(y_test.shape[0], y_test.shape[1])
        y_hat = model.predict(X_test)

        mae = mean_absolute_error(y_true, y_hat)
        mse = mean_squared_error(y_true, y_hat)
        
        return {"mae":mae, "mse":mse}, model


    def _run(self, config):
        metrics, tf_model = self._trainable(config)

        self._log_experiments(config, metrics, tf_model)
        tune.report(**metrics)


    def exec(self, tune_config=None, num_trials=1):
        DEFAULT_CONFIG = {
            "layer_n": tune.randint(2, 3),
            "cell": tune.randint(24, 30)
        }

        config = tune_config if tune_config else DEFAULT_CONFIG
        tune.run(
            self._run,
            config=config,
            metric=self.metric,
            mode=self.TUNE_METRIC_DICT[self.metric],
            num_samples=num_trials,
        )

def save_best_model(
    run_id, model_type, metric, metric_score, model_name
):

    exist_model = conn.execute(
        SELECT_EXIST_MODEL.format(model_name)
    ).fetchone()

    # 업데이트
    if exist_model and exist_model.metric_score >= metric_score:
        conn.execute(
            UPDATE_BEST_MODEL.format(
                run_id, model_type, metric, metric_score, model_name
            )
        )
    else:  # 생성
        conn.execute(
            INSERT_BEST_MODEL.format(
                model_name, run_id, model_type, metric, metric_score
            )
        )