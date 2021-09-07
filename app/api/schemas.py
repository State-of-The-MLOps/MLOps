from pydantic import BaseModel


class ItemBase(BaseModel):
    title: str


class ItemCreate(ItemBase):
    pass


class Item(ItemBase):
    class Config:
        orm_mode = True


class DatasetBase(BaseModel):
    path: str
    version: int


class DatasetCreate(DatasetBase):
    pass


class Dataset(DatasetBase):
    class config:
        orm_mode = True


class ClfModelBase(BaseModel):
    path: str
    version: int
    name: str
    classes: int
    score: float


class ClfModelCreate(ClfModelBase):
    pass


class ClfModel(ClfModelBase):
    class Config:
        orm_mode = True


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
