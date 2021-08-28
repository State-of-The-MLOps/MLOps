from fastapi import APIRouter

router = APIRouter(
    prefix="/predict",
    tags=["predict"],
    responses={404: {"description": "Not Found"}}
)


@router.get("/")
def hello_world():
    return {"message": "Hello predict"}
