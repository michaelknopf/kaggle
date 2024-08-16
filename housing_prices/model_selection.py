import time
from datetime import datetime

import pandas as pd

# noinspection PyUnresolvedReferences
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import cross_val_score, GridSearchCV, HalvingGridSearchCV
from sklearn.model_selection._search import BaseSearchCV

from housing_prices.model import HousingPricesModel, SCORE_FUNCTION
from housing_prices.config import load_config
from housing_prices.prepare_data import load_train_data
from housing_prices.path_anchor import MODEL_SELECTION_DIR


def grid_search(param_grid, cv_splits=5, use_halving=False):
    model_config = load_config()
    model = HousingPricesModel(model_config)

    X, y = load_train_data()

    if use_halving:
        grid_search_cls = HalvingGridSearchCV
    else:
        grid_search_cls = GridSearchCV

    cv = grid_search_cls(estimator=model.pipeline,
                         param_grid=param_grid,
                         scoring=SCORE_FUNCTION,
                         cv=cv_splits,
                         verbose=3)
    cv.fit(X, y)

    save_cv_results(cv)
    print(f'Best params: {cv.best_params_}')
    print(f'Best score: {cv.best_score_}')


def pre_process_grid(grid, prefix=None):
    if isinstance(grid, list):
        grids = grid
    else:
        grids = [grid]
    return [
        {prefix + k: v for k, v in g.items()}
        for g in grids
    ]


def save_cv_results(search_cv: BaseSearchCV):
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
    # grid = {
    #     'learning_rate': [10**-1, 10**-2],
    #     'n_estimators': [10**3, 10**4],
    #     'min_samples_leaf': [1, 3, 5, 9],
    #     'max_leaf_nodes': [4, 8, 13],
    #     'max_features': [None, 'sqrt', 0.8],
    # }
    # grid = pre_process_grid(grid, prefix='regress__regressor__')
    # grid_search(grid, cv_splits=5, use_halving=True)

    grid = [{
        'learning_rate': [10 ** -i],
        'n_estimators': [10 ** (i + 1)],
        'max_leaf_nodes': [8],
        'max_features': ['sqrt'],
    } for i in range(1, 5)]

    grid = pre_process_grid(grid, prefix='regress__regressor__')
    grid_search(grid, cv_splits=5, use_halving=False)
