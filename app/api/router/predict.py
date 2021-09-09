# -*- coding: utf-8 -*-
import os
import codecs
import pickle
import numpy as np
from fastapi.param_functions import Depends
from app.api.schemas import ModelCorePrediction
from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.utils import *
import tensorflow as tf
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


# import tensorflow as tf
# import zipfile
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# def load_tf_model(model_name):
#     """
#     * DB에 있는 텐서플로우 모델을 불러옵니다.
#     * 모델은 zip형식으로 압축되어 binary로 저장되어 있습니다.
#     * 모델의 이름을 받아 압축 해제 및 tf_model폴더 아래에 저장한 후 로드하여
#       텐서플로우 모델 객체를 반환합니다.
#     """
#     query = f"""SELECT model_file
#                 FROM model_core
#                 WHERE model_name='{model_name}';"""
#     bin_data = engine.execute(query).fetchone()[0]
#     model_buffer = pickle.loads(codecs.decode(bin_data, "base64"))
#     with zipfile.ZipFile(model_buffer, "r") as bf:
#         bf.extractall(f"../../tf_model/{model_name}")
#     tf_model = tf.keras.models.load_model(f"./tf_model/{model_name}")
#     print("tf_model loaded")

#     return tf_model

# ----------------------------------------------------------------

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# model_path = load_tf_model("test_model")
# tf_model = tf.keras.models.load_model(model_path)

# ----------------------------------------------------------------

@router.put("/atmos")
async def predict_temperature(time_series: List[float]):
    print('heloo')
    try:
        # tf_model = load_tf_model("test_model")
        tf_model = my_model.my_model
        # time_series = np.array(time_series).reshape(1, -1, 1)
        # result = tf_model.predict(time_series)
        # return result.tolist()
        return 1
    except Exception as e:
        print(e)  # 나중에 로그남기기용
        return {"result": str(e)}
