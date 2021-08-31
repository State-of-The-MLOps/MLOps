from sqlalchemy.orm import Session

from . import models, schemas


def get_pickle(db: Session, version=1):
    return db.query(models.Pickle).filter(
        models.Pickle.version == version
    ).first()


def create_pickle(db: Session, pickle: schemas.PickleCreate):
    db_pickle = models.Pickle(**pickle)
    db.add(db_pickle)
    db.commit()
    db.refresh(db_pickle)
    return db_pickle


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
