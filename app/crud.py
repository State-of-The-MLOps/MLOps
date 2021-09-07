from sqlalchemy.orm import Session

from app import models
from app.api import schemas


def get_dataset(db: Session, version=1):
    return db.query(models.Dataset).filter(
        models.Dataset.version == version
    ).first()


def create_dataset(db: Session, Dataset: schemas.DatasetCreate):
    db_dataset = models.Dataset(**Dataset)
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    return db_dataset


def get_clf_model(db: Session, version=1, name='random_forest'):
    return db.query(models.ClfModel).filter(
        models.ClfModel.version == version and
        models.ClfModel.name == name
    ).first()


def create_clf_model(db: Session, clf_model: schemas.ClfModelCreate):
    db_cf_model = models.ClfModel(**clf_model)
    db.add(db_cf_model)
    db.commit()
    db.refresh(db_cf_model)
    return db_cf_model


def get_reg_model(db: Session, model_name: schemas.RegModelBase):
    return db.query(models.RegModel).filter(
        models.RegModel.model_name == model_name
    ).first()
