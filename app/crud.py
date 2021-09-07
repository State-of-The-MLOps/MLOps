from sqlalchemy.orm import Session

from app import models
from app.api import schemas


def get_reg_model(db: Session, model_name: schemas.RegModelBase):
    return db.query(models.RegModel).filter(
        models.RegModel.model_name == model_name
    ).first()
