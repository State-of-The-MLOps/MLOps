# -*- coding: utf-8 -*-
import subprocess
import multiprocessing
import re
import os

from fastapi import APIRouter

from app.utils import NniWatcher, write_yml, get_free_port, base_dir, check_expr_over
from logger import L


router = APIRouter(
    prefix="/train",
    tags=["train"],
    responses={404: {"description": "Not Found"}}
)


@router.put("/insurance")
def train_insurance(
    experiment_name: str = 'exp1',
    experimenter: str = 'DongUk',
    model_name: str = 'insurance_fee_model',
    version: float = 0.1
):
    """
    insurance와 관련된 학습을 실행하기 위한 API입니다.

    Args:
        experiment_name (str): 실험이름. 기본 값: exp1
        experimeter (str): 실험자의 이름. 기본 값: DongUk
        model_name (str): 모델의 이름. 기본 값: insurance_fee_model
        version (float): 실험의 버전. 기본 값: 0.1

    Returns:
        msg: 실험 실행의 성공과 상관없이 포트번호를 포함한 NNI Dashboard의 주소를 반환합니다.

    Note:
        실험의 최종 결과를 반환하지 않습니다.
    """
    PORT = get_free_port()
    L.info(
        f"Train Args info\n\texperiment_name: {experiment_name}\n\texperimenter: {experimenter}\n\tmodel_name: {model_name}\n\tversion: {version}")
    path = 'experiments/insurance/'
    try:
        write_yml(
            path,
            experiment_name,
            experimenter,
            model_name,
            version
        )
        nni_create_result = subprocess.getoutput(
            "nnictl create --port {} --config {}/{}.yml".format(
                PORT, path, model_name)
        )
        sucs_msg = "Successfully started experiment!"

        if sucs_msg in nni_create_result:
            p = re.compile(r"The experiment id is ([a-zA-Z0-9]+)\n")
            expr_id = p.findall(nni_create_result)[0]
            # expr id 랑 expr name 주고 instance 만들어서
            # a.excute를 target으로 넘겨주자.
            nni_watcher = NniWatcher(expr_id, experiment_name)
            m_process = multiprocessing.Process(
                target=nni_watcher.excute
            )
            m_process.start()

            L.info(nni_create_result)
            return nni_create_result

    except Exception as e:
        L.error(e)
        return {'error': str(e)}


@router.put("/atmos")
def train_atmos(expr_name: str):
    """
    온도 시계열과 관련된 학습을 실행하기 위한 API입니다.

    Args:
        expr_name(str): NNI가 실행할 실험의 이름 입니다. 이 파라미터를 기반으로 project_dir/experiments/[expr_name] 경로로 찾아가 config.yml을 이용하여 NNI를 실행합니다.

    Returns:
        str: NNI실험이 실행된 결과값을 반환하거나 실행과정에서 발생한 에러 메세지를 반환합니다.

    Note:
        실험의 최종 결과를 반환하지 않습니다.
    """

    nni_port = get_free_port()
    expr_path = os.path.join(base_dir, 'experiments', expr_name)

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
            m_process.start()

            L.info(nni_create_result)
            return nni_create_result

        else:
            L.error(nni_create_result)
            return {"error": nni_create_result}

    except Exception as e:
        L.error(e)
        return {'error': str(e)}
