# -*- coding: utf-8 -*-
import codecs
import pickle
import numpy as np
from fastapi.param_functions import Depends, Query
from app.api.schemas import ModelCorePrediction
from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.utils import my_model

from app import crud, models
from app.database import engine
from app.database import SessionLocal


models.Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


router = APIRouter(
    prefix="/predict",
    tags=["predict"],
    responses={404: {"description": "Not Found"}}
)


@router.put("/insurance")
def predict_insurance(info: ModelCorePrediction, model_name: str, db: Session = Depends(get_db)):
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
    reg_model = crud.get_reg_model(db, model_name=model_name)

    if reg_model:
        loaded_model = pickle.loads(
            codecs.decode(reg_model.model_file, 'base64'))

        test_set = np.array([
            info.age,
            info.sex,
            info.bmi,
            info.children,
            info.smoker,
            info.region
        ]).reshape(1, -1)

        pred = loaded_model.predict(test_set)

        return {"result": pred.tolist()[0]}
    else:
        raise HTTPException(
            status_code=404,
            detail="Model Not Found",
            headers={"X-Error": "Model Not Found"},
        )


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
