# -*- coding: utf-8 -*-
import datetime
from sqlalchemy import Column, Integer, String, FLOAT, DateTime
from sqlalchemy.sql.functions import now

from app.database import Base

KST = datetime.timezone(datetime.timedelta(hours=9))


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
    score = Column(FLOAT)


class RegModel(Base):
    __tablename__ = 'reg_model'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    path = Column(String, nullable=False)
    experimenter = Column(String, nullable=False)
    experiment_name = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    version = Column(FLOAT, autoincrement=True)
    metric = Column(String, nullable=False)
    score = Column(FLOAT, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=now())
