from functools import cache
import itertools

import pandas as pd
import numpy as np

from housing_prices.config import load_config
from housing_prices.paths import paths

model_config = load_config()


@cache
def load_train_data():
    return _load_train_data()


def _load_train_data():
    X, y = _load_raw_train_data()
    _pre_process(X)
    return X, y


def _load_raw_train_data():
    df = pd.read_csv(paths.DATA_DIR / 'kaggle_dataset/train.csv')

    # drop the target variable and move it to a separate vector
    X = df.drop('SalePrice', axis='columns')
    y = df['SalePrice']

    return X, y


@cache
def load_test_data():
    return _load_test_data()


def _load_test_data():
    X = _load_raw_test_data()
    _pre_process(X)
    return X


def _load_raw_test_data():
    return pd.read_csv(paths.DATA_DIR / 'kaggle_dataset/test.csv')


def _pre_process(X: pd.DataFrame):
    X['MSZoning'] = X['MSZoning'].replace('C (all)', 'C')

    X['BldgType'] = X['BldgType'].replace('2fmCon', '2FmCon')
    X['BldgType'] = X['BldgType'].replace('Duplex', 'Duplx')
    X['BldgType'] = X['BldgType'].replace('Twnhs', 'TwnhsE')

    for col_name in ['Exterior1st', 'Exterior2nd']:
        X[col_name] = X[col_name].replace('Brk Cmn', 'BrkComm')
        X[col_name] = X[col_name].replace('CmentBd', 'CemntBd')
        X[col_name] = X[col_name].replace('Wd Shng', 'WdShing')

    X['Neighborhood'] = X['Neighborhood'].replace('NAmes', 'Names')

    for feature in itertools.chain(model_config.ordinal_features(),
                                   model_config.categorical_features()):
        if feature.null_rep is not None:
            X[feature.name] = X[feature.name].replace(np.nan, feature.null_rep)
