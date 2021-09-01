# -*- coding: utf-8 -*-
from sqlalchemy import Column, Integer, String

from app.database import Base


class Item(Base):
    __tablename__ = 'items'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String, index=True, default='test')


class Dataset(Base):
    __tablename__ = 'dataset'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    path = Column(String, index=True)
    version = Column(Integer, index=True, autoincrement=True)


class ClfModel(Base):
    __tablename__ = 'clf_model'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    path = Column(String, index=True)
    version = Column(Integer, index=True, autoincrement=True)
    name = Column(String, index=True)
    classes = Column(Integer)
