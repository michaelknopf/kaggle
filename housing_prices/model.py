from functools import cache, cached_property
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, MissingIndicator
from sklearn.pipeline import make_pipeline

from housing_prices.config import load_config, ModelConfig
from housing_prices.prepare_data import load_train_data, load_test_data

import warnings


warnings.filterwarnings("ignore", category=UserWarning, message=".*Found unknown categories in columns.*")

RANDOM_STATE = 0

class HousingPricesModel:

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    @cached_property
    def pipeline(self):
        return make_pipeline(
            self.create_column_transformer(),
            KNNImputer(n_neighbors=2, weights="uniform"),
            # IterativeImputer(max_iter=10, random_state=RANDOM_STATE),
            GradientBoostingRegressor(random_state=RANDOM_STATE)
        )

    def create_column_transformer(self):
        transformers = [
            self.create_categorical_transformer(),
            self.create_ordinal_transformer()
        ]
        return ColumnTransformer(transformers, remainder='passthrough', force_int_remainder_cols=False)

    def create_categorical_transformer(self):
        feature_configs = list(self.model_config.categorical_features())
        categories = [f.categories for f in feature_configs]
        feature_names = [f.name for f in feature_configs]
        one_hot_encoder = OneHotEncoder(dtype='int',
                                        drop='if_binary',
                                        categories=categories,
                                        handle_unknown='infrequent_if_exist')
        return 'Categorical Preprocessor', one_hot_encoder, feature_names

    def create_ordinal_transformer(self):
        feature_configs = list(self.model_config.ordinal_features())
        categories = [f.categories for f in feature_configs]
        feature_names = [f.name for f in feature_configs]
        encoder = OrdinalEncoder(categories=categories,
                                 handle_unknown='use_encoded_value',
                                 unknown_value=-1)
        return 'Ordinal Preprocessor', encoder, feature_names


def test(pipeline) -> np.ndarray:
    X_test = load_test_data()
    return pipeline.predict(X_test)

def train_and_test():
    model_config = load_config()
    model = HousingPricesModel(model_config)

    features, target = load_train_data()
    model.pipeline.fit(features, target)

    predictions = test(model.pipeline)
    print(predictions)

if __name__ == '__main__':
    train_and_test()
