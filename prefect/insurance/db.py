import pandas as pd
import sqlalchemy
import os


def connect(db):
    """Returns a connection and a metadata object"""

    POSTGRES_USER = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_SERVER = os.getenv("POSTGRES_SERVER")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT")
    POSTGRES_DB = db

    url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}"

    connection = sqlalchemy.create_engine(url)

    return connection


POSTGRES_DB = os.getenv("POSTGRES_DB")
engine = connect(POSTGRES_DB)