import time
from functools import cached_property
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, root_mean_squared_log_error

from housing_prices.config import load_config, ModelConfig
from housing_prices.prepare_data import load_train_data, load_test_data
from housing_prices.model_persistence import persist_model


RANDOM_STATE = 0

class HousingPricesModel:

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    @cached_property
    def pipeline(self):
        return make_pipeline(
            self.create_column_transformer(),
            KNNImputer(n_neighbors=2, weights="uniform"),
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

    def persist(self, note=''):
        persist_model(self.pipeline, note=note)


def test(pipeline) -> np.ndarray:
    X_test = load_test_data()
    return pipeline.predict(X_test)

def train_and_test():
    model_config = load_config()
    model = HousingPricesModel(model_config)

    X, y = load_train_data()
    model.pipeline.fit(X, y)

    predictions = test(model.pipeline)
    print(predictions)

def train_and_cross_validate():
    model_config = load_config()
    model = HousingPricesModel(model_config)

    X, y = load_train_data()

    start_time = time.time()
    scores = cross_val_score(model.pipeline, X, y, cv=5, scoring=make_scorer(root_mean_squared_log_error))
    elapsed_time = time.time() - start_time

    for i, score in enumerate(scores):
        print(f'Split {i} score: {score:.2f}')

    print('')
    print(f'Mean: {scores.mean():.2f}')
    print(f'Standard deviation: {scores.std():.2f}')
    print(f'Training time: {elapsed_time:.2f}')

if __name__ == '__main__':
    train_and_cross_validate()
