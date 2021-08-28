from fastapi import FastAPI

from app.router import predict
from app.database import SessionLocal

app = FastAPI()

app.include_router(predict.router)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def hello_world():
    return {"message": "Hello World"}
