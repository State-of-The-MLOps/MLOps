import os
import random
from abc import ABC, abstractmethod

import mlflow
import pandas as pd
import xgboost as xgb
from db import engine
from mlflow.tracking import MlflowClient
from query import INSERT_BEST_MODEL, SELECT_EXIST_MODEL, UPDATE_BEST_MODEL
from ray import tune
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from prefect import task


class ETL:
    def __init__(self, data_extract_query):
        self.df = None
        self.data_extract_query = data_extract_query

    def _extract(self):
        self.df = pd.read_sql(self.data_extract_query, engine)

    def _scaling(self, scale_list, scaler):
        self.df.loc[:, scale_list] = scaler().fit_transform(
            self.df.loc[:, scale_list]
        )

    def _encoder(self, enc_list, encoder):
        for col in enc_list:
            self.df.loc[:, col] = encoder().fit_transform(self.df.loc[:, col])

    def _load(self):
        return self.df.iloc[:, :-1].values, self.df.iloc[:, -1].values

    def exec(self, *args):
        self._extract()
        if args is not None:
            for trans_list, transformer in args:
                if "encoder" in transformer.__name__.lower():
                    self._encoder(trans_list, transformer)
                elif "scaler" in transformer.__name__.lower():
                    self._scaling(trans_list, transformer)
                else:
                    break
        return self._load()


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

    @abstractmethod
    def exec(self):
        pass


class InsuranceTuner(Tuner):
    def __init__(self, data_X, data_y, host_url, exp_name, metric):
        self.host_url = host_url
        self.exp_name = exp_name
        self.metric = metric
        self.data_X = data_X
        self.data_y = data_y
        self.TUNE_METRIC_DICT = {"mae": "min", "mse": "min", "rmse": "min"}

    def _log_experiments(self, config, metrics, xgb_model):
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
                mlflow.xgboost.log_model(
                    xgb_model,
                    artifact_path="model",
                )

    def _trainable(self, config):
        train_x, test_x, train_y, test_y = super()._split(0.2)
        train_set = xgb.DMatrix(train_x, label=train_y)
        test_set = xgb.DMatrix(test_x, label=test_y)

        results = {}
        xgb_model = xgb.train(
            config,
            train_set,
            evals=[(test_set, "eval")],
            evals_result=results,
            verbose_eval=False,
        )
        return results["eval"], xgb_model

    def _run(self, config):
        results, xgb_model = self._trainable(config)

        metrics = {
            "mae": min(results["mae"]),
            "rmse": min(results["rmse"]),
        }

        self._log_experiments(config, metrics, xgb_model)
        tune.report(**metrics)

    def exec(self, tune_config=None, num_trials=1):
        DEFAULT_CONFIG = {
            "objective": "reg:squarederror",
            "eval_metric": ["mae", "rmse"],
            "max_depth": tune.randint(1, 9),
            "min_child_weight": tune.choice([1, 2, 3]),
            "subsample": tune.uniform(0.5, 1.0),
            "eta": tune.loguniform(1e-4, 1e-1),
        }

        config = tune_config if tune_config else DEFAULT_CONFIG
        tune.run(
            self._run,
            config=config,
            metric=self.metric,
            mode=self.TUNE_METRIC_DICT[self.metric],
            num_samples=num_trials,
        )


def save_best_model(run_id, model_type, metric, metric_score, model_name):

    exist_model = engine.execute(
        SELECT_EXIST_MODEL.format(model_name)
    ).fetchone()

    # 업데이트
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


# @task(nout=2)
def etl(query):
    etl = ETL(query)

    label_encode = [["sex", "smoker", "region"], LabelEncoder]
    standard_scale = [["age", "bmi", "children"], StandardScaler]

    X, y = etl.exec(label_encode, standard_scale)

    return X, y


# @task
def train_mlflow_ray(X, y, host_url, exp_name, metric, num_trials):
    mlflow.set_tracking_uri(host_url)
    mlflow.set_experiment(exp_name)

    it = InsuranceTuner(
        data_X=X, data_y=y, host_url=host_url, exp_name=exp_name, metric=metric
    )
    it.exec(num_trials=num_trials)

    return True


# @task
def log_best_model(is_end, host_url, exp_name, metric, model_type):
    mlflow.set_tracking_uri(host_url)

    client = MlflowClient()
    exp_id = client.get_experiment_by_name(exp_name).experiment_id
    runs = mlflow.search_runs([exp_id])

    best_score = runs["metrics.mae"].min()
    best_run = runs[runs["metrics.mae"] == best_score]
    run_id = best_run.run_id.item()

    save_best_model(
        run_id,
        model_type,
        metric,
        metric_score=best_score,
        model_name=exp_name,
    )


if __name__ == "__main__":
    extract_query = "SELECT * FROM insurance"
    host_url = "http://localhost:5001"
    exp_name = "insurance"
    metric = "mae"
    model_type = "xgboost"
    num_trials = 1

    X, y = etl(extract_query)
    is_end = train_mlflow_ray(X, y, host_url, exp_name, metric, num_trials)

    if is_end:
        log_best_model(is_end, host_url, exp_name, metric, model_type)
