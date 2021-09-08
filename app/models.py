# -*- coding: utf-8 -*-
import datetime
from sqlalchemy import Column, Integer, String, FLOAT, DateTime, ForeignKey, LargeBinary
from sqlalchemy.sql.functions import now
from sqlalchemy.orm import relationship

from app.database import Base

KST = datetime.timezone(datetime.timedelta(hours=9))


class ModelCore(Base):
    __tablename__ = 'model_core'

    model_name = Column(String, primary_key=True)
    model_file = Column(LargeBinary, nullable=False)

    model_metadata_relation = relationship(
        "ModelMetadata", backref="model_core.model_name")


class ModelMetadata(Base):
    __tablename__ = 'model_metadata'

    experiment_name = Column(String, primary_key=True)
    model_core_name = Column(String, ForeignKey(
        'model_core.model_name'), nullable=False)
    experimenter = Column(String, nullable=False)
    version = Column(FLOAT)
    train_mae = Column(FLOAT, nullable=False)
    val_mae = Column(FLOAT, nullable=False)
    train_mse = Column(FLOAT, nullable=False)
    val_mse = Column(FLOAT, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=now())
