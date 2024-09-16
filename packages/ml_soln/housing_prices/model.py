from functools import cached_property, cache

import numpy as np
from ml_soln.housing_prices import ctx
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

SCORE_FUNCTION = 'neg_root_mean_squared_log_error'

class HousingPricesModel:

    @cached_property
    def pipeline(self):
        return Pipeline([
            ('preprocess', self._create_column_transformer()),
            ('impute', KNNImputer(n_neighbors=2, weights="uniform")),
            ('regress', self.create_regressor()),
        ])

    @cache
    def fit(self):
        X, y = ctx().data_preparer.train_data()
        self.pipeline.fit(X, y)

    @staticmethod
    def create_regressor():
        hyperparams = ctx().hyperparams.to_dict()
        gradient_boosted_regressor = GradientBoostingRegressor(**hyperparams)
        return TransformedTargetRegressor(
            regressor=gradient_boosted_regressor,
            func=np.log1p,              # log(y + 1)
            inverse_func=np.expm1,      # exp(y) - 1
        )

    def _create_column_transformer(self):
        transformers = [
            self._create_categorical_transformer(),
            self._create_ordinal_transformer()
        ]
        return ColumnTransformer(transformers, remainder='passthrough', force_int_remainder_cols=False)

    def _create_categorical_transformer(self):
        feature_configs = list(ctx().model_config.categorical_features())
        categories = [f.categories for f in feature_configs]
        feature_names = [f.name for f in feature_configs]
        one_hot_encoder = OneHotEncoder(dtype='int',
                                        drop='if_binary',
                                        categories=categories,
                                        handle_unknown='infrequent_if_exist')
        return 'Categorical Preprocessor', one_hot_encoder, feature_names

    def _create_ordinal_transformer(self):
        feature_configs = list(ctx().model_config.ordinal_features())
        categories = [f.categories for f in feature_configs]
        feature_names = [f.name for f in feature_configs]
        encoder = OrdinalEncoder(categories=categories,
                                 handle_unknown='use_encoded_value',
                                 unknown_value=-1)
        return 'Ordinal Preprocessor', encoder, feature_names

    def persist(self, note=''):
        ctx().model_persistence.save_model(self.pipeline, note=note)
