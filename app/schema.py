# -*- coding: utf-8 -*-
import datetime

from sqlalchemy import (
    FLOAT,
    Column,
    String,
)

from app.database import Base

KST = datetime.timezone(datetime.timedelta(hours=9))

class BestModelData(Base):
    __tablename__ = "best_model_data"

    model_name = Column(String, primary_key=True)
    run_id = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    metric = Column(String, nullable=False)
    metric_score = Column(FLOAT, nullable=False)

