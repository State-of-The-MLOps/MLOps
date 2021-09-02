from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import nni
from os.path import join
import multiprocessing
from sqlalchemy.engine import create_engine
from xgboost.sklearn import XGBRegressor

n_cpus = multiprocessing.cpu_count()


def preprocess(x_train, x_valid, col_list):
    tmp_x_train = x_train.copy()
    tmp_x_valid = x_valid.copy()

    tmp_x_train.reset_index(drop=True, inplace=True)
    tmp_x_valid.reset_index(drop=True, inplace=True)

    # 전처리 함수 작성
    encoder = LabelEncoder()

    for col in col_list:
        tmp_x_train.loc[:, col] = encoder.fit_transform(
            tmp_x_train.loc[:, col])
        tmp_x_valid.loc[:, col] = encoder.transform(tmp_x_valid.loc[:, col])

    return tmp_x_train.values, tmp_x_valid.values


def main(params, df):
    global best_model, best_mae

    # df = pd.read_csv(join('.', 'insurance.csv'))

    label_col = ['sex', 'smoker', 'region']

    y = df.charges.to_frame()
    x = df.iloc[:, :-1]

    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.2, random_state=42)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_mse = []
    cv_mae = []

    best_model = None
    best_mae = 99999999

    for trn_idx, val_idx in kf.split(x, y):
        x_train, y_train = x.iloc[trn_idx], y.iloc[trn_idx]
        x_valid, y_valid = x.iloc[val_idx], y.iloc[val_idx]

        # 전처리
        x_tra, x_val = preprocess(x_train, x_valid, label_col)

        # 모델 정의 및 파라미터 전달
        model = XGBRegressor(**params)

        # 모델 학습 및 Early Stopping 적용
        model.fit(x_tra, y_train)

        y_train_pred = model.predict(x_tra)
        y_valid_pred = model.predict(x_val)
        # Loss 계산
        train_mse = mean_squared_error(y_train, y_train_pred)
        valid_mse = mean_squared_error(y_valid, y_valid_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        valid_mae = mean_absolute_error(y_valid, y_valid_pred)

        cv_mse.append(valid_mse)
        cv_mae.append(valid_mae)

        if best_model:
            temp = best_mae
            best_mae = min(best_mae, valid_mae)
            print('best_mae::', best_mae)
            if temp != best_mae:
                best_model = model
                print('best_model::', best_model)
        else:
            best_mae = valid_mae
            best_model = model
            print(best_mae)

    cv_mse_mean = np.mean(cv_mse)
    cv_mae_mean = np.mean(cv_mae)

    print(cv_mse_mean, cv_mae_mean)
    # 학습 결과 리포팅
    print(best_model)
    nni.report_final_result(cv_mae_mean)
    print('Final result is %g', cv_mae_mean)
    print('Send final result done.')


if __name__ == '__main__':
    params = nni.get_next_parameter()
    engine = create_engine(
        "postgresql://postgres:0000@localhost:5432/postgres")

    df = pd.read_sql("""
        SELECT *
        FROM insurance
    """, engine)

    main(params, df)
