from datetime import datetime

from housing_prices.config import load_config
from housing_prices.model import HousingPricesModel
from housing_prices.paths import paths
from housing_prices.prepare_data import load_train_data, load_test_data


def train_and_test():
    model_config = load_config()
    model = HousingPricesModel(model_config)

    X, y = load_train_data()
    model.pipeline.fit(X, y)

    X_test = load_test_data()
    predictions = model.pipeline.predict(X_test)
    X_test['SalePrice'] = predictions

    timestamp = datetime.now().replace(microsecond=0).isoformat()
    paths.SUBMISSIONS_DIR.mkdir(exist_ok=True)
    submission_file = paths.SUBMISSIONS_DIR / f'{timestamp}.csv'
    with open(submission_file, 'w') as f:
        X_test.to_csv(f, index=False, columns=['Id', 'SalePrice'])

    return submission_file


if __name__ == '__main__':
    train_and_test()
