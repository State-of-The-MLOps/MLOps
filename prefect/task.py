import requests
import json
import os
from dotenv import load_dotenv
import sqlalchemy
import pandas as pd
from datetime import timedelta
import tensorflow_data_validation as tfdv
from prefect import task


def connect(db):
    """Returns a connection and a metadata object"""

    load_dotenv(verbose=True)

    POSTGRES_USER = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_SERVER = os.getenv("POSTGRES_SERVER")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT")
    POSTGRES_DB = db

    url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}"

    connection = sqlalchemy.create_engine(url)

    return connection


@task
def data_extract(conn, start_date, end_date):
    endpoint = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"

    stt_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')

    if stt_date_str == end_date_str\
    and f'{start_date.hour:>02}' == '00':
        return False, 'data is already up to date'

    date_range = [start_date, 
                    *pd.date_range(stt_date_str, 
                                    end_date_str)[1:]]

    atmos_data = []
    for dr in date_range:
        req_url = (f"{endpoint}?serviceKey={os.getenv('ATMOS_API_KEY')}"
                f"&numOfRows=24&dataType=JSON&dataCd=ASOS&dateCd=HR"
                f"&startDt={dr.strftime('%Y%m%d')}&startHh={dr.hour:>02}"
                f"&endDt={dr.strftime('%Y%m%d')}&endHh=23&stnIds=108")
        
        resp = requests.get(req_url)

        json_file = json.loads(resp.text)
        json_data = json_file['response']['body']['items']['item']

        raw_data = list(map(lambda x: x.values(), js_data))
        atmos_data.extend(raw_data)
    
    atmos_df = pd.DataFrame(atmos_data,
                            columns = js_data[0].keys())
    atmos_df = atmos_df[['tm', 'ta', 'rn', 'wd', 'ws', 'pa', 'ps', 'hm']]
    atmos_df.columns = ['time', 'tmp', 'precip', 'wd', 'ws', 'p', 'mslp', 'rh']

    atmos_df['precip'] = atmos_df['precip'].apply(lambda x: 0 if not x else x)

    return True, atmos_df


@task
def data_validation(conn, start_date, df):
    # if not df_or_flag:
    #     return False

    org_data_query = f"""
    SELECT *
    FROM atmos_stn108
    WHERE time < '{start_date}';
    """

    org_data = pd.read_sql(org_data_query, conn)
    new_data = df

    org_stats = tfdv.generate_statistics_from_dataframe(org_data)
    new_stats = tfdv.generate_statistics_from_dataframe(new_data)

    org_schema = tfdv.infer_schema(org_stats)

    for i in org_data.keys()[1:]:
        temp=tfdv.get_feature(org_schema, i)
        temp.drift_comparator.infinity_norm.threshold = 0.01

    drift_anomaly = tfdv.validate_statistics(statistics=new_stats, 
                                             schema=org_schema,
                                             previous_statistics=org_stats)

    drift_stats = []
    for anm in drift_anomaly.drift_skew_info:
        if anm.drift_measurements[0].value > anm.drift_measurements[0].threshold:
            drift_stats.append(anm)

    if not drift_stats:
        return True, new_data
    else:
        return False, drift_stats


@task
def data_load_to_db(conn, new_data):
    logger = prefect.context.get("logger")
    try:
        pd.to_sql("atmos_stn108", conn, index=False, if_exists='append')
        logger.info("data has been saved successfully!")

    except Exception as e:
        logger.info(e)