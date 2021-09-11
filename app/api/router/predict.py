# -*- coding: utf-8 -*-
import codecs
import numpy as np
import pickle
from typing import List

from fastapi import APIRouter, HTTPException

from app import models
from app.api.schemas import ModelCorePrediction
from app.database import engine
from app.utils import my_model


models.Base.metadata.create_all(bind=engine)


router = APIRouter(
    prefix="/predict",
    tags=["predict"],
    responses={404: {"description": "Not Found"}}
)


@router.put("/insurance")
def predict_insurance(info: ModelCorePrediction, model_name: str):
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
    query = """
        SELECT model_file
        FROM model_core
        WHERE model_name='{}';
    """.format(model_name)

    reg_model = engine.execute(query).fetchone()

    if reg_model is None:
        raise HTTPException(
            status_code=404,
            detail="Model Not Found",
            headers={"X-Error": "Model Not Found"},
        )

    loaded_model = pickle.loads(
        codecs.decode(reg_model[0], 'base64'))

    info = info.dict()
    test_set = np.array([*info.values()]).reshape(1, -1)

    pred = loaded_model.predict(test_set)

    return {"result": pred.tolist()[0]}


@router.put("/atmos")
async def predict_temperature(time_series: List[float]):
    if len(time_series) != 72:
        return "time series must have 72 values"

    try:
        tf_model = my_model.my_model
        time_series = np.array(time_series).reshape(1, -1, 1)
        result = tf_model.predict(time_series)
        return result.tolist()

    except Exception as e:
        print(e)
        return {"error": str(e)}
