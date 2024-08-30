from functools import cached_property

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from housing_prices.config import ModelConfig
from common.model_persistence import save_model

RANDOM_STATE = 0
SCORE_FUNCTION = 'neg_root_mean_squared_log_error'

class HousingPricesModel:

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    @cached_property
    def pipeline(self):
        return Pipeline([
            ('preprocess', self.create_column_transformer()),
            ('impute', KNNImputer(n_neighbors=2, weights="uniform")),
            ('regress', self.create_regressor()),
        ])

    @staticmethod
    def create_regressor():
        best_params = {
            'learning_rate': 10**-4,
            'n_estimators': 10**5,
            'max_leaf_nodes': 8,
            'max_features': 'sqrt',
        }
        gradient_boosted_regressor = GradientBoostingRegressor(
            random_state=RANDOM_STATE,
            **best_params
        )
        return TransformedTargetRegressor(
            regressor=gradient_boosted_regressor,
            func=np.log1p,              # log(y + 1)
            inverse_func=np.expm1,      # exp(y) - 1
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

    def persist(self, note=''):
        save_model(self.pipeline, note=note)
