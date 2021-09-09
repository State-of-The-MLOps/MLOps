import pandas as pd


def preprocess(data):

    # missing data
    data = data.fillna(method="ffill")

    # etc.

    return data
