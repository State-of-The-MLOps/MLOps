import requests
import os
from utils import *
import sqlalchemy
import pandas as pd
from datetime import timedelta
import tensorflow_data_validation as tfdv
import prefect
from prefect import task
from prefect.tasks.prefect.flow_run_cancel import CancelFlowRun
import mlflow

@task
def data_extract(api_key):
    logger = prefect.context.get("logger")

    timenow = pd.Timestamp.utcnow()\
              .tz_convert("Asia/Seoul")\
              .strftime("%Y-%m-%d %H:%M")
    logger.info(f"{timenow}(KST): start data ETL process")

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
        return False

    date_range = [start_date,
                 *pd.date_range(stt_date_str, 
                                end_date_str)[1:]]
    logger.info(date_range)

    atmos_data = []
    for dr in date_range:
        req_url = (f"{endpoint}?serviceKey={api_key}"
                   f"&numOfRows=24&dataType=JSON&dataCd=ASOS&dateCd=HR"
                   f"&startDt={dr.strftime('%Y%m%d')}&startHh={dr.hour:>02}"
                   f"&endDt={dr.strftime('%Y%m%d')}&endHh=23&stnIds=108")
        
        resp = requests.get(req_url)
        logger.info(f"{resp}: {dr}")

        if not resp.ok:
            logger.error(f"status code: {resp.status_code}")
            logger.error(f"request error: {resp.text}")
            break

        try:
            json_file = resp.json()
            json_data = json_file['response']['body']['items']['item']
        except Exception as e:
            logger.info(f"response text: {resp.text}")
            logger.error(e)
            break

        raw_data = list(map(lambda x: x.values(), json_data))
        atmos_data.extend(raw_data)
    
    if not atmos_data:
        logger.error("failed to request data")
        CFR = CancelFlowRun()
        CFR.run()
        return False
    
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
        return False


@task
def data_load_to_db(new_data, ps_user, ps_host, ps_pw):
    logger = prefect.context.get("logger")
    logger.info((f"data type: {type(new_data)}\n\r"
                 f"data shape: {new_data.shape}"))
    try:
        load_to_db(new_data)
        logger.info("data has been saved successfully!")
        return True
    except Exception as e:
        logger.error(e)
        CFR = CancelFlowRun()
        CFR.run()
        return False


@task
def train_mlflow_ray(load_data_suc, host_url, exp_name, metric, num_trials):
    mlflow.set_tracking_uri(host_url)
    mlflow.set_experiment(exp_name)

    it = AtmosTuner(
        host_url=host_url, exp_name=exp_name, metric=metric
    )
    it.exec(num_trials=num_trials)

    return True


@task
def log_best_model(is_end, host_url, exp_name, metric, model_type):
    mlflow.set_tracking_uri(host_url)

    client = MlflowClient()
    exp_id = client.get_experiment_by_name(exp_name).experiment_id
    runs = mlflow.search_runs([exp_id])

    best_score = runs["metrics.mae"].min()
    best_run = runs[runs["metrics.mae"] == best_score]
    run_data = mlflow.get_run(best_run.run_id.item()).data
    history = eval(run_data.tags["mlflow.log-model.history"])

    artifact_uri = best_run["artifact_uri"].item()
    artifact_path = history[0]["artifact_path"]

    artifact_uri = artifact_uri + f"/{artifact_path}"

    save_best_model(
        artifact_uri,
        model_type,
        metric,
        metric_score=best_score,
        model_name=exp_name,
    )
