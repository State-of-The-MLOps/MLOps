# -*- coding: utf-8 -*-
from fastapi import APIRouter
from fastapi import Depends
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy.orm import Session

from app import crud
from app.database import engine
from app.database import SessionLocal
import app.models as models
from app.util import mnist_preprocessing


models.Base.metadata.create_all(bind=engine)

# Dependency


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


router = APIRouter(
    prefix="/train",
    tags=["train"],
    responses={404: {"description": "Not Found"}}
)


@router.post('/mnist')
def train_mnist_rf(
    model_name: str = 'model.pkl',
    version: int = 1,
    db: Session = Depends(get_db)
):
    """
    param
        version: int
    return
        path: str
        version: int
        name: str
        classes: int
    """

    dataset = crud.get_dataset(db, version=version)
    data = np.load(dataset.path, allow_pickle=True)

    X_train, X_valid, y_train, y_test = mnist_preprocessing(data)

    clf_model = RandomForestClassifier(n_estimators=500,
                                       max_depth=3,
                                       random_state=0)

    clf_model.fit(X_train, y_train)

    pickle_md = crud.create_clf_model(db, clf_model={
        'path': 'model.pkl',
        'version': version,
        'name': 'random_forest',
        'classes': len(np.unique(y_train))
    })

    pickle.dump(clf_model, open('model.pkl', 'wb'))

    return pickle_md
