# -*- coding: utf-8 -*-
import ast
import asyncio
import os
from typing import List

import mlflow
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import xgboost as xgb
from dotenv import load_dotenv
from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from app import schema
from app.api.data_class import MnistData, ModelCorePrediction
from app.database import engine
from app.query import SELECT_BEST_MODEL
from app.utils import (
    CachingModel,
    VarTimer,
    load_data_cloud,
    softmax,
)
from logger import L

load_dotenv()

schema.Base.metadata.create_all(bind=engine)

host_url = os.getenv("MLFLOW_HOST")
mlflow.set_tracking_uri(host_url)
reset_sec = 5
CLOUD_STORAGE_NAME = os.getenv("CLOUD_STORAGE_NAME")
CLOUD_TRAIN_MNIST = os.getenv("CLOUD_TRAIN_MNIST")
CLOUD_VALID_MNIST = os.getenv("CLOUD_VALID_MNIST")

router = APIRouter(
    prefix="/predict",
    tags=["predict"],
    responses={404: {"description": "Not Found"}},
)


mnist_model = CachingModel("pytorch", 600)
knn_model = CachingModel("sklearn", 600)
data_lock = asyncio.Lock()
train_df = VarTimer(600)


@router.put("/mnist")
async def predict_mnist(item: MnistData):
    item = np.array(ast.literal_eval(item.mnist_num)).astype(np.uint8)
    global train_df
    global mnist_model, knn_model
    model_name = "mnist"
    model_name2 = "mnist_knn"

    if not isinstance(train_df._var, pd.DataFrame):
        async with data_lock:
            if not isinstance(train_df._var, pd.DataFrame):
                train_df.cache_var(
                    load_data_cloud(CLOUD_STORAGE_NAME, CLOUD_TRAIN_MNIST)
                )

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    reshaped_input = item.reshape(28, 28)
    transformed_input = transform(reshaped_input)
    transformed_input = transformed_input.view(1, 1, 28, 28)

    await mnist_model.get_model(model_name)
    await knn_model.get_model(model_name2)

    def sync_call(mnist_model, knn_model, train_df):
        # Net1
        result = mnist_model.predict(transformed_input)
        p_res = softmax(result.detach().numpy()) * 100
        percentage = np.around(p_res[0], 2).tolist()

        # Net2
        result = mnist_model.predict(transformed_input, True)
        result = result.detach().numpy()

        # KNN
        knn_result = knn_model.predict(result)

        xai_result = train_df.get_var().iloc[knn_result, 1:].values[0].tolist()
        return {
            "result": {
                "percentage": percentage,
                "answer": percentage.index(max(percentage)),
                "xai_result": xai_result,
            },
            "error": None,
        }

    try:
        result = await run_in_threadpool(
            sync_call, mnist_model, knn_model, train_df
        )
        return result
    except Exception as e:
        return {"result": "Can't predict", "error": str(e)}


insurance_model = CachingModel("xgboost", 30)


@router.put("/insurance")
async def predict_insurance(info: ModelCorePrediction):
    info = info.dict()
    test_set = xgb.DMatrix(np.array([*info.values()]).reshape(1, -1))

    model_name = "insurance"
    await insurance_model.get_model(model_name)
    result = insurance_model.predict(test_set)

    result = float(result[0])
    return result

lock = asyncio.Lock()
atmos_model_cache = VarTimer()


@router.put("/atmos_temperature")
async def predict_temperature_(time_series: List[float]):
    """
    온도 1시간 간격 시계열을 입력받아 이후 24시간 동안의 온도를 1시간 간격의 시계열로 예측합니다.
    Args:
        time_series(List): 72시간 동안의 1시간 간격 온도 시계열 입니다. 72개의 원소를 가져야 합니다.
    Returns:
        List[float]: 입력받은 시간 이후 24시간 동안의 1시간 간격 온도 예측 시계열 입니다.
    """

    global lock

    if len(time_series) != 72:
        L.error(f"input time_series: {time_series} is not valid")
        return {"result": "time series must have 72 values", "error": None}

    model_name = "atmos_tmp"

    if not atmos_model_cache.is_var:
        async with lock:
            if not atmos_model_cache.is_var:
                run_id = engine.execute(
                    SELECT_BEST_MODEL.format(model_name)
                ).fetchone()[0]
                print("start load model from mlflow")
                atmos_model_cache.cache_var(
                    mlflow.keras.load_model(f"runs:/{run_id}/model")
                )
                print("end load model from mlflow")

    def sync_pred_ts(time_series):
        """
        none sync 함수를  sync로 만들어 주기 위한 함수이며 입출력은 부모 함수와 같습니다.
        """

        time_series = np.array(time_series).reshape(1, 72, 1)
        result = atmos_model_cache.get_var().predict(time_series)
        atmos_model_cache.reset_timer()
        L.info(
            f"Predict Args info: {time_series.flatten().tolist()}\n\tmodel_name: {model_name}\n\tPrediction Result: {result.tolist()[0]}"
        )

        return {"result": result.tolist(), "error": None}

    try:
        result = await run_in_threadpool(sync_pred_ts, time_series)
        return result

    except Exception as e:
        L.error(e)
        return {"result": "Can't predict", "error": str(e)}
