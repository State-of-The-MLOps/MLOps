# -*- coding: utf-8 -*-
from typing import List

import numpy as np
from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from app import models
from app.api.schemas import ModelCorePrediction
from app.database import engine
from app.utils import ScikitLearnModel, my_model
from logger import L


models.Base.metadata.create_all(bind=engine)


router = APIRouter(
    prefix="/predict",
    tags=["predict"],
    responses={404: {"description": "Not Found"}}
)


@router.put("/insurance")
async def predict_insurance(info: ModelCorePrediction, model_name: str):
    """
    Get information and predict insurance fee
    param:
        info:
            # 임시로 int형태를 받도록 제작
            # preprocess 단계를 거치도록 만들 예정
            age: int
            sex: int
            bmi: float
            children: int
            smoker: int
            region: int
    return:
        insurance_fee: float
    """
    def sync_call(info, model_name):
        model = ScikitLearnModel(model_name)
        model.load_model()

        info = info.dict()
        test_set = np.array([*info.values()]).reshape(1, -1)

        pred = model.predict_target(test_set)
        return {"result": pred.tolist()[0]}
    try:
        result = await run_in_threadpool(sync_call, info, model_name)
        L.info(
            f"Predict Args info: {info}\n\tmodel_name: {model_name}\n\tPrediction Result: {result}")
        return result

    except Exception as e:
        L.error(e)
        return {'error': str(e)}


@router.put("/atmos")
async def predict_temperature(time_series: List[float]):
    if len(time_series) != 72:
        L.error(f'input time_series: {time_series} is not valid')
        return "time series must have 72 values"

    try:
        tf_model = my_model.model
        time_series = np.array(time_series).reshape(1, -1, 1)
        result = tf_model.predict(time_series)

        L.info(
            f"Predict Args info: {time_series.flatten().tolist()}\n\tmodel_name: {tf_model}\n\tPrediction Result: {result.tolist()[0]}")
        return result.tolist()

    except Exception as e:
        L.error(e)
        return {"error": str(e)}
