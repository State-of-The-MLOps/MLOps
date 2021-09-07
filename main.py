from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import predict

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)


@app.get("/")
def hello_world():
    return {"message": "Hello World"}
