import os
import codecs
import pickle
import sys
import getopt
sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))

from dotenv import load_dotenv
import pandas as pd
import numpy as np
import nni
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from xgboost.sklearn import XGBRegressor

from expr_db import connect
from query import *

load_dotenv(verbose=True)

def preprocess(x_train, x_valid, col_list):
    """
    param:
        x_train : train dataset dataframe
        x_valid : validation dataset dataframe
        col_list : columns that required for LabelEncoding
    return:
        tmp_x_train.values : numpy.ndarray
        tmp_x_valid.values : numpy.ndarray
    """
    tmp_x_train = x_train.copy()
    tmp_x_valid = x_valid.copy()

    tmp_x_train.reset_index(drop=True, inplace=True)
    tmp_x_valid.reset_index(drop=True, inplace=True)

    encoder = LabelEncoder()

    for col in col_list:
        tmp_x_train.loc[:, col] = encoder.fit_transform(
            tmp_x_train.loc[:, col])
        tmp_x_valid.loc[:, col] = encoder.transform(tmp_x_valid.loc[:, col])

    return tmp_x_train.values, tmp_x_valid.values


def main(params, engine, experiment_info, connection):
    """
    param:
        params: Parameters determined by NNi
        engine: sqlalchemy engine
        experiment_info: information of experiment [dict]
        connection: connection used to communicate with DB
    """

    df = pd.read_sql(SELECT_ALL_INSURANCE, engine)
    experimenter = experiment_info['experimenter']
    experiment_name = experiment_info['experiment_name']
    model_name = experiment_info['model_name']
    version = experiment_info['version']

    label_col = ['sex', 'smoker', 'region']

    y = df.charges.to_frame()
    x = df.iloc[:, :-1]

    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.2, random_state=42)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_mse, cv_mae, tr_mse, tr_mae = [], [], [], []
    fold_mae, fold_model = 1e10, None

    for trn_idx, val_idx in kf.split(x, y):
        x_train, y_train = x.iloc[trn_idx], y.iloc[trn_idx]
        x_valid, y_valid = x.iloc[val_idx], y.iloc[val_idx]

        # 전처리
        x_tra, x_val = preprocess(x_train, x_valid, label_col)

        # 모델 정의 및 파라미터 전달
        model = XGBRegressor(**params)

        # 모델 학습 및 Early Stopping 적용
        model.fit(x_tra, y_train, eval_set=[
                  (x_val, y_valid)], early_stopping_rounds=10)

        y_train_pred = model.predict(x_tra)
        y_valid_pred = model.predict(x_val)
        # Loss 계산
        train_mse = mean_squared_error(y_train, y_train_pred)
        valid_mse = mean_squared_error(y_valid, y_valid_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        valid_mae = mean_absolute_error(y_valid, y_valid_pred)

        cv_mse.append(valid_mse)
        cv_mae.append(valid_mae)
        tr_mse.append(train_mse)
        tr_mae.append(train_mae)

        new_mae = min(fold_mae, valid_mae)
        if new_mae != fold_mae:
            fold_model = model

    cv_mse_mean = np.mean(cv_mse)
    cv_mae_mean = np.mean(cv_mae)
    tr_mse_mean = np.mean(tr_mse)
    tr_mae_mean = np.mean(tr_mae)

    best_model = pd.read_sql(SELECT_MODEL_CORE % (model_name), engine)

    if len(best_model) == 0:

        pickled_model = codecs.encode(pickle.dumps(model), "base64").decode()
        connection.execute(INSERT_MODEL_CORE % (model_name, pickled_model))
        connection.execute(INSERT_MODEL_METADATA % (
            experiment_name,
            model_name,
            experimenter,
            version,
            tr_mae_mean,
            tr_mse_mean,
            cv_mae_mean,
            cv_mse_mean)
        )

    else:
        best_model_metadata = pd.read_sql(
            SELECT_VAL_MAE % (model_name), engine)
        saved_score = best_model_metadata.values[0]

        if saved_score > valid_mae:
            pickled_model = codecs.encode(
                pickle.dumps(fold_model), "base64").decode()

            connection.execute(UPDATE_MODEL_CORE % (pickled_model, model_name))
            connection.execute(UPDATE_MODEL_METADATA % (
                tr_mae_mean,
                cv_mae_mean,
                tr_mse_mean,
                cv_mse_mean,
                experiment_name)
            )

    nni.report_final_result(cv_mae_mean)
    print('Final result is %g', cv_mae_mean)
    print('Send final result done.')


if __name__ == '__main__':
    params = nni.get_next_parameter()
    engine = connect()
    argv = sys.argv
    experiment_info = {}

    try:
        opts, etc_args = getopt.getopt(
            argv[1:],
            "e:n:m:v:",
            [
                "experimenter=",
                "experiment_name=",
                "model_name=",
                "version="
            ])
        for opt, arg in opts:
            if opt in ('-e', "--experimenter"):
                experiment_info['experimenter'] = f"'{arg}'"
            elif opt in ("-n", "--experiment_name"):
                experiment_info['experiment_name'] = f"'{arg}'"
            elif opt in ("-m", "--model_name"):
                experiment_info['model_name'] = f"'{arg}'"
            elif opt in ("-v", "--version"):
                experiment_info['version'] = arg

    except getopt.GetoptError:
        sys.exit(2)

    with engine.connect() as connection:
        with connection.begin():
            main(params, engine, experiment_info, connection)
