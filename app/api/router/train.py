# -*- coding: utf-8 -*-
import subprocess

from fastapi import APIRouter

from app.utils import write_yml


router = APIRouter(
    prefix="/train",
    tags=["train"],
    responses={404: {"description": "Not Found"}}
)


@router.put("/")
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
    path = 'experiments/insurance/'
    try:
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
        print('error')
        print(e)

    return {"msg": f'Check out http://127.0.0.1:{PORT}'}
