# -*- coding: utf-8 -*-
from sqlalchemy import Column, Integer, String

from .database import Base


class Item(Base):
    __tablename__ = 'items'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String, index=True, default='test')


class Pickle(Base):
    __tablename__ = 'pickle'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    path = Column(String, index=True)
    version = Column(Integer, index=True, autoincrement=True)
