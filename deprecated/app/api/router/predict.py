# -*- coding: utf-8 -*-
import numpy as np
import torchvision.transforms as transforms
from dotenv import load_dotenv
from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from app import schema
from app.api.data_class import ModelCorePrediction
from app.database import engine
from app.utils import (
    ScikitLearnModel,
)
from logger import L

load_dotenv()

schema.Base.metadata.create_all(bind=engine)

router = APIRouter(
    prefix="/predict",
    tags=["predict"],
    responses={404: {"description": "Not Found"}},
)

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
