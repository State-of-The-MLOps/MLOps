import asyncio
import os
import threading
import time
from io import StringIO

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from dotenv import load_dotenv
from google.cloud import storage

from app.database import engine
from app.query import *
from logger import L

load_dotenv()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def softmax(x):

    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

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


def load_data_cloud(bucket_name):
    data_path, _ = get_data_path_from_db(2, 'mnist')
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(data_path)

    bytes_data = blob.download_as_bytes()

    s = str(bytes_data, "utf-8")

    data = StringIO(s)
    df = pd.read_csv(data)

    return df


class VarTimer:
    def __init__(self, caching_time=5):
        self._var = None
        self._caching_time = caching_time
        self._reset_flag = False

    def cache_var(self, var, caching_time=False):
        if caching_time:
            self._change_timedelta(caching_time)
        self._var = var
        self._reset_flag = True
        cleaner = threading.Thread(target=self._value_cleaner)
        cleaner.start()

    def _value_cleaner(self):
        while self._reset_flag:
            self._reset_flag = False
            time.sleep(self._caching_time)
        self._var = None

    def get_var(self):
        self._reset_flag = True
        return self._var

    def reset_timer(self, caching_time=False):
        if caching_time:
            self._change_timedelta(caching_time)
        self._reset_flag = True

    def _change_timedelta(self, caching_time):
        if not (
            isinstance(caching_time, int) | isinstance(caching_time, float)
        ):

            print(
                (
                    f"timedelta must be int or float! "
                    f'"{caching_time}"(type {type(caching_time)}) isn\'t applied'
                )
            )
        else:
            self._caching_time = caching_time

    @property
    def is_var(self):
        return True if self._var else False


class CachingModel(VarTimer):
    def __init__(self, model_type, caching_time=5):
        super().__init__(caching_time)
        self._run_id = None
        self._model_type = model_type
        self._lock = asyncio.Lock()
        self._model_name = None
        self._model_type_dict = {
            "keras": mlflow.keras.load_model,
            "pytorch": mlflow.pytorch.load_model,
            "sklearn": mlflow.sklearn.load_model,
            "xgboost": mlflow.xgboost.load_model,
        }
        self._model_load_way = {
            "production": "models:/{}/Production",
            "score": "runs:/{}/model",
        }

    def _load_run_id(self, model_name):
        self._run_id = engine.execute(
            SELECT_BEST_MODEL.format(model_name)
        ).fetchone()[0]

    def _load_model_mlflow(self, load_type):
        _model_load_arg = {
            "production": self._model_name,
            "score": self._run_id,
        }
        model_uri = self._model_load_way[load_type].format(
            _model_load_arg[load_type]
        )
        model = self._model_type_dict[self._model_type](model_uri)

        return model

    async def get_model(self, model_name, load_type="production"):
        if not super().is_var:
            async with self._lock:
                if not super().is_var:
                    if load_type == "production":
                        self._model_name = model_name
                    else:
                        self._load_run_id(model_name)
                    super().cache_var(self._load_model_mlflow(load_type))
        else:
            super().reset_timer()

    def predict(self, data, cut=False):
        if self._model_type == "pytorch":
            if cut:
                return torch.nn.Sequential(
                    *list(self._var.children())[:-1]
                ).forward(data)
            else:
                return self._var.forward(data)
        else:
            return self._var.predict(data)
