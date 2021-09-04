import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


def connect(db):

    load_dotenv(verbose=True)

    POSTGRES_USER = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT")
    POSTGRES_SERVER = os.getenv("POSTGRES_SERVER")

    SQLALCHEMY_DATABASE_URL = \
        f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@' +\
        f'{POSTGRES_SERVER}:{POSTGRES_PORT}/{db}'

    connection = create_engine(SQLALCHEMY_DATABASE_URL)

    return connection


POSTGRES_DB = os.getenv("POSTGRES_DB")

engine = connect(POSTGRES_DB)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
