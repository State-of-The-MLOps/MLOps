# -*- coding: utf-8 -*-
import subprocess
import multiprocessing
import re
import os

from fastapi import APIRouter

from app.utils import write_yml, get_free_port, base_dir, check_expr_over
from logger import L


router = APIRouter(
    prefix="/train",
    tags=["train"],
    responses={404: {"description": "Not Found"}}
)


@router.put("/insurance")
def train_insurance(
    PORT: int = 8080,
    experiment_sec: int = 20,
    experiment_name: str = 'exp1',
    experimenter: str = 'DongUk',
    model_name: str = 'insurance_fee_model',
    version: float = 0.1
):
    """
    Args:
        PORT (int): PORT to run NNi. Defaults to 8080
        experiment_sec (int): Express the experiment time in seconds Defaults to 20
        experiment_name (str): experiment name Defaults to exp1
        experimeter (str): experimenter (author) Defaults to DongUk
        model_name (str): model name Defaults to insurance_fee_model
        version (float): version of experiment Defaults to 0.1
    Returns:
        msg: Regardless of success or not, return address values including PORT.
    """
    L.info(
        f"Train Args info\n\texperiment_sec: {experiment_sec}\n\texperiment_name: {experiment_name}\n\texperimenter: {experimenter}\n\tmodel_name: {model_name}\n\tversion: {version}")
    path = 'experiments/insurance/'
    try:
        L.info("Start NNi")
        write_yml(
            path,
            experiment_name,
            experimenter,
            model_name,
            version
        )
        subprocess.Popen(
            "nnictl create --port {} --config {}/{}.yml && timeout {} && nnictl stop --port {}".format(
                PORT, path, model_name, experiment_sec, PORT),
            shell=True,
        )

    except Exception as e:
        L.error(e)
        return {'error': str(e)}

    return {"msg": f'Check out http://127.0.0.1:{PORT}'}


@router.put("/atmos")
def train_atmos(expr_name: str):
    nni_port = get_free_port()
    expr_path = os.path.join(base_dir, 'experiments', expr_name)

    # subprocess로 nni실행
    try:
        nni_create_result = subprocess.getoutput(
            "nnictl create --port {} --config {}/config.yml".format(
                nni_port, expr_path))
        sucs_msg = "Successfully started experiment!"

        if sucs_msg in nni_create_result:
            p = re.compile(r"The experiment id is ([a-zA-Z0-9]+)\n")
            expr_id = p.findall(nni_create_result)[0]
            m_process = multiprocessing.Process(
                target=check_expr_over,
                args=(expr_id, expr_name, expr_path)
            )
            m_process.start()  # 자식 프로세스 분리(nni 실험 진행상황 감시 및 모델 저장)

            L.info(nni_create_result)
            return nni_create_result

        else:
            L.error(nni_create_result)
            return {"error": nni_create_result}

    except Exception as e:
        L.error(e)
        return {'error': str(e)}

    # 코드는 바이너리로 저장하는건 별로인가?(버전관리 차원에서 score랑 같은 행에...)
