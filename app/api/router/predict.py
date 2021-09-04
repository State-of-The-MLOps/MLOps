# -*- coding: utf-8 -*-
from fastapi import APIRouter

from app import models
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


@router.get("/")
def hello_world():
    return {"message": "Hello predict"}
