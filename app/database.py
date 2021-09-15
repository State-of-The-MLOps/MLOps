import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

load_dotenv(verbose=True)

def connect(db):
    """
    database와의 연결을 위한 함수 입니다.
    
    Args:
        db(str): 사용할 데이터베이스의 이름을 전달받습니다.
        
    Returns:
        created database engine: 데이터베이스에 연결된 객체를 반환합니다.
    """
    print(db)

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
