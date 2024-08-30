from housing_prices.config import load_config
from housing_prices.model import HousingPricesModel
from housing_prices.paths import paths
from housing_prices.prepare_data import load_train_data, load_test_data
from common.predict import save_submission


def train_and_test():
    model_config = load_config()
    model = HousingPricesModel(model_config)

    X, y = load_train_data()
    model.pipeline.fit(X, y)

    X_test = load_test_data()
    predictions = model.pipeline.predict(X_test)
    X_test['SalePrice'] = predictions

    return save_submission(X_test, paths.submissions_dir)


if __name__ == '__main__':
    train_and_test()
