# -*- coding: utf-8 -*-
import numpy as np
from typing import List

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from app import models
from app.api.schemas import ModelCorePrediction
from app.database import engine
from app.utils import ScikitLearnModel, my_model


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

    result = await run_in_threadpool(sync_call, info, model_name)

    return result


@router.put("/atmos")
async def predict_temperature(time_series: List[float]):
    if len(time_series) != 72:
        return "time series must have 72 values"

    try:
        tf_model = my_model.model
        time_series = np.array(time_series).reshape(1, -1, 1)
        result = tf_model.predict(time_series)
        return result.tolist()

    except Exception as e:
        print(e)
        return {"error": str(e)}
