# -*- coding: utf-8 -*-
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
def read_file(column=0, row=5, db: Session = Depends(get_db)):
    """
    Upload file Temporary

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
    pkl = crud.get_pickle(db, version=1)
    try:
        data = np.load(pkl.path, allow_pickle=True)
        return {'data': data[:int(row), 0:int(column)+1].tolist()}
    except Exception as e:
        print(e)
        return 0
