import time
from datetime import datetime

import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV

from housing_prices.model import HousingPricesModel, SCORE_FUNCTION
from housing_prices.config import load_config
from housing_prices.prepare_data import load_train_data
from housing_prices.path_anchor import MODEL_SELECTION_DIR


def grid_search(param_grid, cv_splits=5):
    model_config = load_config()
    model = HousingPricesModel(model_config)

    X, y = load_train_data()

    cv = GridSearchCV(estimator=model.pipeline,
                      param_grid=param_grid,
                      scoring=SCORE_FUNCTION,
                      cv=cv_splits,
                      verbose=3)
    cv.fit(X, y)

    save_cv_results(cv)
    print(f'Best params: {cv.best_params_}')
    print(f'Best score: {cv.best_score_}')


def save_cv_results(search_cv: GridSearchCV):
    df = pd.DataFrame(search_cv.cv_results_)

    timestamp = datetime.now().replace(microsecond=0).isoformat()
    MODEL_SELECTION_DIR.mkdir(exist_ok=True)
    with open(MODEL_SELECTION_DIR / f'{timestamp}.json', 'w') as f:
        df.to_json(f, indent=2, orient='records')

    with open(MODEL_SELECTION_DIR / f'{timestamp}.csv', 'w') as f:
        df.to_csv(f, index=False)


def train_and_cross_validate(cv_splits=5):
    model_config = load_config()
    model = HousingPricesModel(model_config)

    X, y = load_train_data()

    start_time = time.time()
    scores = cross_val_score(model.pipeline, X, y, cv=cv_splits, scoring=SCORE_FUNCTION)
    elapsed_time = time.time() - start_time

    for i, score in enumerate(scores):
        print(f'Split {i} score: {score:.2f}')

    print('')
    print(f'Mean: {scores.mean():.2f}')
    print(f'Standard deviation: {scores.std():.2f}')
    print(f'Training time: {elapsed_time:.2f}')


if __name__ == '__main__':
    grid = {
        'learning_rate': [0.1],
        'n_estimators': [1000],
        'min_samples_leaf': [3, 5],
        'max_depth': [7],
    }
    grid = {'regress__regressor__' + k: v for k, v in grid.items()}
    grid_search(grid, cv_splits=2)
