# -*- coding: utf-8 -*-
import datetime
import io
import os
import pickle
from typing import List

import mlflow
import numpy as np
import pandas as pd
import redis
import torch
import torchvision.transforms as transforms
import xgboost as xgb
from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool
from tensorflow.keras.layers import deserialize, serialize

from app import schema
from app.api.data_class import ModelCorePrediction
from app.database import engine
from app.query import SELECT_BEST_MODEL
from app.utils import ScikitLearnModel, softmax
from logger import L

schema.Base.metadata.create_all(bind=engine)

pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
client = redis.Redis(connection_pool=pool)

host_url = os.getenv("MLFLOW_HOST")


mlflow.set_tracking_uri(host_url)
reset_sec = 5

router = APIRouter(
    prefix="/predict",
    tags=["predict"],
    responses={404: {"description": "Not Found"}},
)


@router.put("/insurance")
def predict_insurance(
    info: ModelCorePrediction, model_name: str = "insurance"
):
    info = info.dict()
    test_set = xgb.DMatrix(np.array([*info.values()]).reshape(1, -1))

    model = client.get("redis_model")

    if model:
        model = pickle.loads(model)
        client.expire("redis_model", reset_sec)
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


@router.put("/mnist")
def predict_mnist(num=1):
    model_name = "mnist"
    model_name2 = "mnist_knn"

    Net1 = client.get(f"{model_name}_cached")
    knn_model = client.get(f"{model_name2}_cached")
    if Net1:
        stream = io.BytesIO(Net1)  # implements seek()
        Net1 = torch.load(stream)
    else:
        run_id = engine.execute(
            SELECT_BEST_MODEL.format(model_name)
        ).fetchone()[0]
        Net1 = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
        client.set(
            f"{model_name}_cached",
            Net1.save_to_buffer(),
            datetime.timedelta(seconds=40),
        )

    if knn_model:
        knn_model = pickle.loads(knn_model)
    else:
        run_id2 = engine.execute(
            SELECT_BEST_MODEL.format(model_name2)
        ).fetchone()[0]
        knn_model = mlflow.sklearn.load_model(f"runs:/{run_id2}/model")
        client.set(
            f"{model_name2}_cached",
            pickle.dumps(knn_model),
            datetime.timedelta(seconds=40),
        )

    # Net1
    sample_df = pd.read_csv(
        "/Users/TFG5076XG/Documents/MLOps/prefect/mnist/mnist_valid.csv"
    )
    sample_row = sample_df.iloc[int(num), 1:].values.astype(np.uint8)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    aa = sample_row.reshape(28, 28)
    sample = transform(aa)

    sample = sample.view(1, 1, 28, 28)
    result = Net1.forward(sample)

    p_res = softmax(result.detach().numpy()) * 100
    percentage = np.around(p_res[0], 2).tolist()

    # Net2
    Net2 = torch.nn.Sequential(*list(Net1.children())[:-1])
    result = Net2.forward(sample)
    result = result.detach().numpy()

    # KNN
    knn_result = knn_model.predict(result)
    df2 = pd.read_csv(
        "/Users/TFG5076XG/Documents/MLOps/prefect/mnist/mnist_train.csv"
    )
    xai_result = df2.iloc[knn_result, 1:].values[0].tolist()

    return percentage, percentage.index(max(percentage)), xai_result


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

    model_name = 'atmos_tmp'
    model = client.get(f"{model_name}_cached")
    if model:
        print("predict cached model")
        model = deserialize(pickle.loads(model))
        client.expire(f"{model_name}_cached", reset_sec)
        
    else:
        run_id = engine.execute(
            SELECT_BEST_MODEL.format(model_name)
        ).fetchone()[0]
        print("start load model from mlflow")
        model = mlflow.keras.load_model(f"runs:/{run_id}/model")
        print("end load model from mlflow")
        client.set(
        f"{model_name}_cached", pickle.dumps(serialize(model)), datetime.timedelta(seconds=reset_sec)
        )

    def sync_pred_ts(time_series):
        """
        none sync 함수를  sync로 만들어 주기 위한 함수이며 입출력은 부모 함수와 같습니다.
        """

        time_series = np.array(time_series).reshape(1, 72, 1)
        result = model.predict(time_series)
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
