from functools import cache, cached_property
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, MissingIndicator
from sklearn.pipeline import make_pipeline

from housing_prices.config import load_config, ModelConfig, FeatureConfig
from housing_prices.path_anchor import DATA_DIR

RANDOM_STATE = 0


@cache
def load_data():
    df = pd.read_csv(DATA_DIR / 'kaggle_dataset/train.csv')

    # drop the target variable and move it to a separate vector
    features = df.drop('SalePrice', axis='columns')
    target = df['SalePrice']

    return features, target

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
        categories = [self.get_categories(f) for f in feature_configs]
        feature_names = [f.name for f in feature_configs]
        one_hot_encoder = OneHotEncoder(dtype='int',
                                        drop='if_binary',
                                        categories=categories,
                                        handle_unknown='infrequent_if_exist')
        encoder = make_pipeline(
            self.create_replace_c_all_transformer(),
            one_hot_encoder
        )
        return 'Categorical Preprocessor', encoder, feature_names

    def create_ordinal_transformer(self):
        feature_configs = list(self.model_config.ordinal_features())
        categories = [self.get_categories(f) for f in feature_configs]
        feature_names = [f.name for f in feature_configs]
        encoder = OrdinalEncoder(categories=categories,
                                 handle_unknown='use_encoded_value',
                                 unknown_value=-1)
        return 'Ordinal Preprocessor', encoder, feature_names

    @classmethod
    def create_replace_c_all_transformer(cls):
        def replace_c_all(df: pd.DataFrame):
            if 'MSZoning' in df.columns:
                df['MSZoning'] = df['MSZoning'].replace('C (all)', 'C')
            return df
        return FunctionTransformer(replace_c_all)

    @classmethod
    def get_categories(cls, feature_config):
        categories = feature_config.categories
        if 'NA' in categories:
            categories = reversed(categories)
            categories = [x for x in categories if x != 'NA'] + [np.nan]
        return categories

def test(pipeline) -> np.ndarray:
    X_test = pd.read_csv(DATA_DIR / 'kaggle_dataset/test.csv')
    return pipeline.predict(X_test)

def main():
    feature_config = load_config()
    model = HousingPricesModel(feature_config)

    features, target = load_data()
    model.pipeline.fit(features, target)

    predictions = test(model.pipeline)
    print(predictions)

if __name__ == '__main__':
    main()
