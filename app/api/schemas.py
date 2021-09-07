from pydantic import BaseModel


class RegModelBase(BaseModel):
    model_name: str


class RegModelPrediction(RegModelBase):
    age: int
    sex: int
    bmi: float
    children: int
    smoker: int
    region: int


class RegModel(RegModelBase):
    class Config:
        orm_mode = True
