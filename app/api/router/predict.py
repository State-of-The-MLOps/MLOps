# -*- coding: utf-8 -*-
import datetime
import pickle
from typing import List

import mlflow
import numpy as np
import redis
import xgboost as xgb
from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from app import models
from app.api.schemas import ModelCorePrediction
from app.database import engine
from app.query import SELECT_BEST_MODEL
from app.utils import ScikitLearnModel, my_model
from logger import L

models.Base.metadata.create_all(bind=engine)

client = redis.Redis(host="localhost", port=6379, password=0000, db=0)

router = APIRouter(
    prefix="/predict",
    tags=["predict"],
    responses={404: {"description": "Not Found"}},
)


@router.put("/")
def predict_insurance(info: ModelCorePrediction, model_name: str):
    info = info.dict()
    test_set = xgb.DMatrix(np.array([*info.values()]).reshape(1, -1))

    model = client.get("redis_model")

    if model:
        model = pickle.loads(model)
        print(model)
    else:
        print("else")
        artifact_uri = engine.execute(
            SELECT_BEST_MODEL.format(model_name)
        ).fetchone()
        model = mlflow.xgboost.load_model(artifact_uri)
        client.set(
            "redis_model", pickle.dumps(model), datetime.timedelta(seconds=5)
        )

    result = float(model.predict(test_set)[0])
    return result


@router.put("/insurance")
async def predict_insurance(info: ModelCorePrediction, model_name: str):
    """
    정보를 입력받아 보험료를 예측하여 반환합니다.

    Args:
        info(dict): 다음의 값들을 입력받습니다. age(int), sex(int), bmi(float), children(int), smoker(int), region(int)

    Returns:
        insurance_fee(float): 보험료 예측값입니다.
    """

    def sync_call(info, model_name):
        """
        none sync 함수를  sync로 만들어 주기 위한 함수이며 입출력은 부모 함수와 같습니다.
        """
        model = ScikitLearnModel(model_name)
        model.load_model()

        info = info.dict()
        test_set = np.array([*info.values()]).reshape(1, -1)

        pred = model.predict_target(test_set)
        return {"result": pred.tolist()[0]}

    try:
        result = await run_in_threadpool(sync_call, info, model_name)
        L.info(
            f"Predict Args info: {info}\n\tmodel_name: {model_name}\n\tPrediction Result: {result}"
        )
        return result

    except Exception as e:
        L.error(e)
        return {"result": "Can't predict", "error": str(e)}


@router.put("/atmos")
async def predict_temperature(time_series: List[float]):
    """
    온도 1시간 간격 시계열을 입력받아 이후 24시간 동안의 온도를 1시간 간격의 시계열로 예측합니다.

    Args:
        time_series(List): 72시간 동안의 1시간 간격 온도 시계열 입니다. 72개의 원소를 가져야 합니다.

    Returns:
        List[float]: 입력받은 시간 이후 24시간 동안의 1시간 간격 온도 예측 시계열 입니다.
    """
    if len(time_series) != 72:
        L.error(f"input time_series: {time_series} is not valid")
        return {"result": "time series must have 72 values", "error": None}

    def sync_pred_ts(time_series):
        """
        none sync 함수를  sync로 만들어 주기 위한 함수이며 입출력은 부모 함수와 같습니다.
        """
        time_series = np.array(time_series).reshape(1, -1, 1)
        result = my_model.predict_target(time_series)
        L.info(
            f"Predict Args info: {time_series.flatten().tolist()}\n\tmodel_name: {my_model.model_name}\n\tPrediction Result: {result.tolist()[0]}"
        )

        return {"result": result.tolist(), "error": None}

    try:
        result = await run_in_threadpool(sync_pred_ts, time_series)
        return result

    except Exception as e:
        L.error(e)
        return {"result": "Can't predict", "error": str(e)}
