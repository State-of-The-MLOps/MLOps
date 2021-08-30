# -*- coding: utf-8 -*-
import shutil

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app import crud
from .. import models
from ..database import SessionLocal, engine


models.Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


router = APIRouter(
    prefix="/upload",
    tags=["upload"],
    responses={404: {"description": "Not Found"}}
)


@router.post('/file')
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload file Temporary

    param
        file: File
    return (임시적으로 구성)
        file_name: str (path of filename)
        pickle:
            path: str
            version: int
    """
    with open(f'{file.filename}', 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    pickle = crud.create_pickle(db, pickle={
        'path': file.filename,
        'version': 1
    })
    return {'file_name': file.filename, 'pickle': pickle}
