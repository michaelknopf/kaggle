from datetime import datetime

import numpy as np

from housing_prices.config import load_config
from housing_prices.model import HousingPricesModel
from housing_prices.path_anchor import SUBMISSIONS_DIR
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
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    with open(SUBMISSIONS_DIR / f'{timestamp}.csv', 'w') as f:
        X_test.to_csv(f, index=False, columns=['Id', 'SalePrice'])


if __name__ == '__main__':
    train_and_test()
