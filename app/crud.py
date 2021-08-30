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
