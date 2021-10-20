import requests
import json
import os
from dotenv import load_dotenv
import sqlalchemy
import pandas as pd
from datetime import timedelta
import tensorflow_data_validation as tfdv
import prefect
from prefect import task
from prefect.tasks.prefect.flow_run_cancel import CancelFlowRun

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

conn = connect('postgres')

def load_to_db(data):
    data.to_sql("atmos_stn108", conn, index=False, if_exists='append')

def get_st_date():
    start_date = conn.execute(
            "SELECT time FROM atmos_stn108 ORDER BY time DESC;"
            ).fetchone()[0]
    return start_date

def get_org_data(start_date):
    org_data_query = f"""
    SELECT *
    FROM atmos_stn108
    WHERE time < '{start_date}';
    """
    org_data = pd.read_sql(org_data_query, conn)
    return org_data


@task
def data_extract(api_key):
    logger = prefect.context.get("logger")

    start_date = get_st_date()
    start_date = pd.to_datetime(start_date) + timedelta(hours=1)
    end_date = pd.Timestamp.utcnow() - timedelta(hours=15)
    endpoint = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"

    stt_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    if start_date > pd.to_datetime(end_date_str + " 23:00"):
        up_to_date = 'data is already up to date'
        logger.info(up_to_date)
        CFR = CancelFlowRun()
        CFR.run()
        return False, up_to_date

    date_range = [start_date,
                 *pd.date_range(stt_date_str, 
                                end_date_str)[1:]]
    logger.warning(date_range)

    flag = 1
    atmos_data = []
    for dr in date_range:

        req_url = (f"{endpoint}?serviceKey={api_key}"
                   f"&numOfRows=24&dataType=JSON&dataCd=ASOS&dateCd=HR"
                   f"&startDt={dr.strftime('%Y%m%d')}&startHh={dr.hour:>02}"
                   f"&endDt={dr.strftime('%Y%m%d')}&endHh=23&stnIds=108")
        
        resp = requests.get(req_url)
        logger.info(resp)

        if not resp.ok:
            json_file = json.loads(resp.text)
            logger.error(f"status code: {resp.status_code}")
            logger.error(json_file)

        json_file = json.loads(resp.text)
        if flag == 1:
            logger.info(json_file)
            flag -= 1

        json_data = json_file['response']['body']['items']['item']

        raw_data = list(map(lambda x: x.values(), json_data))
        atmos_data.extend(raw_data)
    
    atmos_df = pd.DataFrame(atmos_data,
                            columns = json_data[0].keys())
    atmos_df = atmos_df[['tm', 'ta', 'rn', 'wd', 'ws', 'pa', 'ps', 'hm']]
    atmos_df.columns = ['time', 'tmp', 'precip', 'wd', 'ws', 'p', 'mslp', 'rh']

    atmos_df['precip'] = atmos_df['precip'].apply(lambda x: 0 if not x else x)
    atmos_df['time'] = pd.to_datetime(atmos_df['time'])
    atmos_df.iloc[:, 1:] = atmos_df.iloc[:, 1:].astype(float)

    return True, atmos_df


@task
def data_validation(new_data):

    logger = prefect.context.get("logger")
    logger.info((f"data type: {type(new_data)}\n\r"
                 f"data shape: {new_data.shape}"))
    
    start_date = get_st_date()
    org_data = get_org_data(start_date)
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
    logger.info(f"data drift vars: {drift_stats}")

    if not drift_stats:
        logger.info(True)
        return True, new_data
    else:
        logger.info(False)
        logger.info(drift_stats)
        CFR = CancelFlowRun()
        CFR.run()


@task
def data_load_to_db(new_data, ps_user, ps_host, ps_pw):
    logger = prefect.context.get("logger")
    logger.info((f"data type: {type(new_data)}\n\r"
                 f"data shape: {new_data.shape}"))
    load_to_db(new_data)
    logger.info("data has been saved successfully!")
