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
