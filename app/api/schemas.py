from pydantic import BaseModel


class ModelCoreBase(BaseModel):
    model_name: str


class ModelCorePrediction(BaseModel):
    age: int
    sex: int
    bmi: float
    children: int
    smoker: int
    region: int


class ModelCore(ModelCoreBase):
    class Config:
        orm_mode = True
