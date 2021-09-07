# -*- coding: utf-8 -*-
import datetime
from sqlalchemy import Column, String, FLOAT, DateTime, ForeignKey
from sqlalchemy.sql.functions import now
from sqlalchemy.orm import relationship

from app.database import Base

KST = datetime.timezone(datetime.timedelta(hours=9))


class RegModel(Base):
    __tablename__ = 'reg_model'

    model_name = Column(String, primary_key=True)
    path = Column(String, nullable=False)

    model_metadata = relationship(
        "RegModelMetadata", backref="reg_model.model_name")


class RegModelMetadata(Base):
    __tablename__ = 'reg_model_metadata'

    experiment_name = Column(String, primary_key=True)
    reg_model_name = Column(String, ForeignKey(
        'reg_model.model_name'), nullable=False)
    experimenter = Column(String, nullable=False)
    version = Column(FLOAT)
    train_mae = Column(FLOAT, nullable=False)
    val_mae = Column(FLOAT, nullable=False)
    train_mse = Column(FLOAT, nullable=False)
    val_mse = Column(FLOAT, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=now())
