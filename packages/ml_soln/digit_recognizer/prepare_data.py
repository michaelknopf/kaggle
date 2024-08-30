from functools import cache

import pandas as pd
from keras.api.utils import to_categorical

from ml_soln.digit_recognizer import ctx


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
        df = pd.read_csv(ctx().paths.data_dir / 'kaggle_dataset/train.csv')
        y = df["label"]
        X = df.drop(labels=["label"], axis=1)
        return X, y

    @staticmethod
    def _pre_process_x(X):
        # Normalize the data
        X = X / 255.0

        # Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
        X = X.values.reshape(-1, 28, 28, 1)

        return X

    @staticmethod
    def _pre_process_y(y):
        return to_categorical(y, num_classes=10)
