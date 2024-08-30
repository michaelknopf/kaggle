from ml_soln.housing_prices.config import load_config
from ml_soln.housing_prices.model import HousingPricesModel
from ml_soln.housing_prices.paths import paths
from ml_soln.housing_prices.prepare_data import load_train_data, load_test_data
from ml_soln.common.predict import save_submission


def train_and_test():
    model_config = load_config()
    model = HousingPricesModel(model_config)

    X, y = load_train_data()
    model.pipeline.fit(X, y)

    X_test = load_test_data()
    predictions = model.pipeline.predict(X_test)
    X_test['SalePrice'] = predictions

    return save_submission(X_test, paths.predictions_dir)


if __name__ == '__main__':
    train_and_test()
