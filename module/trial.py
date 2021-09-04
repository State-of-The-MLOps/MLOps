import os
import pickle

from dotenv import load_dotenv
import pandas as pd
import numpy as np
import nni
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sqlalchemy.engine import create_engine
from xgboost.sklearn import XGBRegressor

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


def main(params, df, engine, experiment_info, connection):
    """
    param:
        params: Parameters determined by NNi
        df: Dataframe read from DB
        engine: sqlalchemy engine
        experiment_info: information of experiment [dict]
        connection: connection used to communicate with DB
    """

    path = experiment_info['path']
    experimenter = experiment_info['experimenter']
    experiment_name = experiment_info['experiment_name']
    model_name = experiment_info['model_name']
    version = experiment_info['version']

    global best_model, best_mae

    label_col = ['sex', 'smoker', 'region']

    y = df.charges.to_frame()
    x = df.iloc[:, :-1]

    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.2, random_state=42)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_mse = []
    cv_mae = []
    tr_mse = []
    tr_mae = []

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

    cv_mse_mean = np.mean(cv_mse)
    cv_mae_mean = np.mean(cv_mae)
    tr_mse_mean = np.mean(tr_mse)
    tr_mae_mean = np.mean(tr_mae)

    best_model = pd.read_sql(f"""
                    SELECT *
                    FROM reg_model
                    WHERE experiment_name = {experiment_name}
                """, engine)

    if len(best_model) == 0:

        with open(f"{os.path.join(path, model_name)}.pkl".replace("'", ""), "wb") as f:
            pickle.dump(model, f)
        connection.execute(f"""
            INSERT INTO reg_model (
                path,
                experimenter,
                experiment_name,
                model_name,
                version,
                train_mae,
                val_mae,
                train_mse,
                val_mse
                ) VALUES (
                    {path},
                    {experimenter},
                    {experiment_name},
                    {model_name},
                    {version},
                    {tr_mae_mean},
                    {tr_mse_mean},
                    {cv_mae_mean},
                    {cv_mse_mean}
                )
        """)
    else:
        with open(f"{os.path.join(path, model_name)}.pkl".replace("'", ""), "wb") as f:
            pickle.dump(model, f)
        saved_score = best_model['val_mae'].values[0]
        if saved_score > valid_mae:
            connection.execute(f"""
                UPDATE reg_model
                SET 
                    train_mae = {tr_mae_mean},
                    val_mae = {cv_mae_mean},
                    train_mse = {tr_mse_mean},
                    val_mse = {cv_mse_mean}
                WHERE experiment_name = {experiment_name}
            """)

    nni.report_final_result(cv_mae_mean)
    print('Final result is %g', cv_mae_mean)
    print('Send final result done.')


if __name__ == '__main__':
    params = nni.get_next_parameter()
    POSTGRES_USER = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT")
    POSTGRES_SERVER = os.getenv("POSTGRES_SERVER")
    POSTGRES_DB = os.getenv("POSTGRES_DB")
    SQLALCHEMY_DATABASE_URL = \
        f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@' +\
        f'{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}'
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

    experiment_info = {
        'path': "'C:\\Users\\TFG5076XG\\Documents\\MLOps'",
        'experimenter': "'DongUk'",
        'experiment_name': "'insurance0903'",
        'model_name': "'keep_update_model'",
        'version': 0.1
    }

    df = pd.read_sql("""
        SELECT *
        FROM insurance
    """, engine)

    with engine.connect() as connection:
        with connection.begin():
            main(params, df, engine, experiment_info, connection)
