from sqlalchemy.orm import Session

from app import models
from app.api import schemas


def get_reg_model(db: Session, model_name: schemas.ModelCoreBase):
    return db.query(models.ModelCore).filter_by(model_name=model_name).first()
