from pydantic import BaseModel


class ModelCoreBase(BaseModel):
    model_name: str


class ModelCorePrediction(BaseModel):
    """
    predict_insurance API의 입력 값 검증을 위한 pydantic 클래스입니다.
    
    Attributes:
        age(int)
        sex(int)
        bmi(float)
        children(int)
        smoker(int)
        region(int)
    """
    age: int
    sex: int
    bmi: float
    children: int
    smoker: int
    region: int


class ModelCore(ModelCoreBase):
    class Config:
        orm_mode = True
