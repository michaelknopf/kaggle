from functools import cache
import itertools

import pandas as pd
import numpy as np
from ml_soln.housing_prices import ctx


class DataPreparer:

    @cache
    def train_data(self):
        return self._load_train_data()

    def _load_train_data(self):
        X, y = self._load_raw_train_data()
        self._pre_process_x(X)
        return X, y

    @staticmethod
    def _load_raw_train_data():
        df = pd.read_csv(ctx().paths.input_dir / 'kaggle_dataset' / 'train.csv')
        # drop the target variable and move it to a separate vector
        X = df.drop('SalePrice', axis='columns')
        y = df['SalePrice']
        return X, y

    @cache
    def test_data(self):
        return self._load_test_data()

    def _load_test_data(self):
        X = self._load_raw_test_data()
        self._pre_process_x(X)
        return X

    @staticmethod
    def _load_raw_test_data():
        return pd.read_csv(ctx().paths.input_dir / 'kaggle_dataset' / 'test.csv')

    @staticmethod
    def _pre_process_x(X: pd.DataFrame):
        X['MSZoning'] = X['MSZoning'].replace('C (all)', 'C')

        X['BldgType'] = X['BldgType'].replace('2fmCon', '2FmCon')
        X['BldgType'] = X['BldgType'].replace('Duplex', 'Duplx')
        X['BldgType'] = X['BldgType'].replace('Twnhs', 'TwnhsE')

        for col_name in ['Exterior1st', 'Exterior2nd']:
            X[col_name] = X[col_name].replace('Brk Cmn', 'BrkComm')
            X[col_name] = X[col_name].replace('CmentBd', 'CemntBd')
            X[col_name] = X[col_name].replace('Wd Shng', 'WdShing')

        X['Neighborhood'] = X['Neighborhood'].replace('NAmes', 'Names')

        for feature in itertools.chain(ctx().model_config.ordinal_features(),
                                       ctx().model_config.categorical_features()):
            if feature.null_rep is not None:
                X[feature.name] = X[feature.name].replace(np.nan, feature.null_rep)
