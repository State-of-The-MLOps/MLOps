import os
import sys
import time
from preprocessing import preprocess
sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))

import numpy as np
import nni
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import GRU

from expr_db import connect

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def make_dataset(data, label, window_size=365, predsize=None):
    feature_list = []
    label_list = []

    if isinstance(predsize, int):
        for i in range(len(data) - (window_size + predsize)):
            feature_list.append(np.array(data.iloc[i : i + window_size]))
            label_list.append(
                np.array(label.iloc[i + window_size : i + window_size + predsize])
            )
    else:
        for i in range(len(data) - window_size):
            feature_list.append(np.array(data.iloc[i : i + window_size]))
            label_list.append(np.array(label.iloc[i + window_size]))

    return np.array(feature_list), np.array(label_list)


def split_data(data_length, ratio):
    """
    return index based on ratio
    --------------------------------------------------
    example

    >>> split_data(data_length = 20, ratio = [1,2,3])
    [3, 10]
    --------------------------------------------------
    """
    ratio = np.cumsum(np.array(ratio) / np.sum(ratio))

    idx = []
    for i in ratio[:-1]:
        idx.append(round(data_length * i))

    return idx


def main(params):
    con = connect("atmos_data")
    data = pd.read_sql("select tmp from stn108;", con)

    data = preprocess(data)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = base_dir.split(os.path.sep)[-1]

    train_feature, train_label = make_dataset(data, data, 72, 24)

    idx = split_data(train_feature.shape[0], [6, 3, 1])
    X_train, X_valid, X_test = (
        train_feature[: idx[0]],
        train_feature[idx[0] : idx[1]],
        train_feature[idx[1] :],
    )
    y_train, y_valid, y_test = (
        train_label[: idx[0]],
        train_label[idx[0] : idx[1]],
        train_label[idx[1] :],
    )

    model = Sequential()
    for layer in range(params["layer_n"]):
        if layer == params["layer_n"] - 1:
            model.add(GRU(params["cell"]))
        else:
            model.add(
                GRU(
                    params["cell"],
                    return_sequences=True,
                    input_shape=[None, train_feature.shape[2]],
                )
            )
    model.add(Dense(24))

    model_path = "./"
    model.compile(loss="mae", optimizer=keras.optimizers.Adam(lr=0.001))
    early_stop = EarlyStopping(monitor="val_loss", patience=30)
    expr_time = time.strftime("%y%m%d_%H%M%S")
    filename = os.path.join(model_path, 
                            f"./temp/{expr_time}_{parent_dir}")
    checkpoint = ModelCheckpoint(
        filename,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="auto",
    )

    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=128,
        validation_data=(X_valid, y_valid),
        callbacks=[early_stop, checkpoint],
    )

    result = (
        np.sum(
            np.abs(
                y_test.reshape(y_test.shape[0], y_test.shape[1]) - model.predict(X_test)
            )
        )
        / y_test.shape[0]
    )

    nni.report_final_result(result)


if __name__ == "__main__":
    params = nni.get_next_parameter()
    main(params)
