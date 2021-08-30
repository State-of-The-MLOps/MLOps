from pydantic import BaseModel


class ItemBase(BaseModel):
    title: str


class ItemCreate(ItemBase):
    pass


class Item(ItemBase):
    class Config:
        orm_mode = True


class PickleBase(BaseModel):
    path: str
    version: int


class PickleCreate(PickleBase):
    pass


class Pickle(PickleBase):
    class config:
        orm_mode = True
