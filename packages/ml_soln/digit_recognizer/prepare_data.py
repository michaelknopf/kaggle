from functools import cache

import pandas as pd
from keras.api.utils import to_categorical

from ml_soln.digit_recognizer import ctx

N_DIGITS = 10
GRAYSCALE_MAX = 255.0
MAX_WIDTH = 28
MAX_HEIGHT = 28

class DataPreparer:

    @cache
    def train_data(self):
        return self._load_train_data()

    def _load_train_data(self):
        X, y = self._load_raw_train_data()
        X = self._pre_process_x(X)
        y = self._pre_process_y(y)
        return X, y

    @staticmethod
    def _load_raw_train_data():
        df = pd.read_csv(ctx().paths.input_dir / 'kaggle_dataset' / 'train.csv')
        y = df["label"]
        X = df.drop(labels=["label"], axis=1)
        return X, y

    @cache
    def test_data(self):
        return self._load_test_data()

    def _load_test_data(self):
        X = self._load_raw_test_data()
        return self._pre_process_x(X)

    @staticmethod
    def _load_raw_test_data():
        return pd.read_csv(ctx().paths.input_dir / 'kaggle_dataset' / 'test.csv')

    @staticmethod
    def _pre_process_x(X):
        # Normalize the data
        X = X / GRAYSCALE_MAX

        # Reshape image in 3 dimensions (height = 28px, width = 28px, canal = 1)
        X = X.values.reshape(-1, MAX_WIDTH, MAX_HEIGHT, 1)

        return X

    @staticmethod
    def _pre_process_y(y):
        return to_categorical(y, num_classes=N_DIGITS)
