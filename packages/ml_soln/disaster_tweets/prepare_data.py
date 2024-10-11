from functools import cache

import pandas as pd

from ml_soln.disaster_tweets import ctx


class DataPreparer:

    @cache
    def train_data(self):
        return self._load_train_data()

    def _load_train_data(self):
        X, y = self._load_raw_train_data()
        X = self._pre_process_x(X)
        y = self._pre_process_y(y)
        return X, y

    @cache
    def _load_raw_train_data(self):
        df = pd.read_csv(ctx().paths.input_dir / 'kaggle_dataset' / 'train.csv')
        y = df["target"]
        X = df.drop(labels=["target"], axis=1)
        return X, y

    @cache
    def test_data(self):
        return self._load_test_data()

    def _load_test_data(self):
        X = self._load_raw_test_data()
        return self._pre_process_x(X)

    @cache
    def _load_raw_test_data(self):
        return pd.read_csv(ctx().paths.input_dir / 'kaggle_dataset' / 'test.csv')

    @staticmethod
    def _pre_process_x(X):
        return X['text']

    @staticmethod
    def _pre_process_y(y):
        return y
