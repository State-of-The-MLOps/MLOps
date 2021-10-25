import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://ehddnr:0000@localhost:5431/ehddnr"
)  # set yours

table_check = engine.execute(
    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
).fetchall()
table_list = [list(dict(r).values())[0] for r in table_check]

# 임시적으로 추가한 부분으로 추후 삭제
if "insurance" not in table_list:
    print("make insurance table")
    ins = pd.read_csv("./insurance.csv")
    ins.to_sql("insurance", engine, index=False)
