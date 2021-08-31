# -*- coding: utf-8 -*-
import pickle
from app import crud

from fastapi import Depends, APIRouter
import numpy as np
from sqlalchemy.orm import Session

from .. import models
from ..database import SessionLocal, engine


models.Base.metadata.create_all(bind=engine)

# Dependency


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


router = APIRouter(
    prefix="/test",
    tags=["test"],
    responses={404: {"description": "Not Found"}}
)


@router.get('/file')
def read_file(column: int = 0, row: int = 5, db: Session = Depends(get_db)):
    """
    Read file Temporary

    param
        (validation 진행 필요)
        column: int (default=0)
        row: int (deafult=5)
    return
        (임시적으로 구성)
        file_name: str (path of filename)
        pickle:
            path: str
            version: int
    """
    pkl = crud.get_dataset(db, version=1)
    try:
        data = np.load(pkl.path, allow_pickle=True)
        return {'data': data[:row, 0:column+1].tolist()}
    except Exception as e:
        print(e)
        return 0


@router.get('/model')
def read_model(version=1, name='random_forest', db: Session = Depends(get_db)):
    """
    Read Model Temporary

    param
        version: int
        name: str
    return
        path: str
        version: int
        name: str
        classes: int
    """
    clf_model = crud.get_clf_model(db, version=version, name=name)
    try:
        loaded_model = pickle.load(open(clf_model.path, 'rb'))
        test = pickle.load(open('test_mnist.pkl', 'rb')).reshape(1, -1)

        pred = loaded_model.predict(test)

        return pred.tolist()

    except Exception as e:
        print(e)
        return 11
