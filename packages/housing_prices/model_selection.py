# noinspection PyUnresolvedReferences
from sklearn.experimental import enable_halving_search_cv

from common.model_selection import grid_search, simple_grid_search_cv, pre_process_grid
from housing_prices.config import load_config
from housing_prices.model import HousingPricesModel, SCORE_FUNCTION
from housing_prices.paths import paths
from housing_prices.prepare_data import load_train_data


model_config = load_config()
model = HousingPricesModel(model_config)
X, y = load_train_data()

def housing_grid_search_1():
    grid = {
        'learning_rate': [10**-1, 10**-2],
        'n_estimators': [10**3, 10**4],
        'min_samples_leaf': [1, 3, 5, 9],
        'max_leaf_nodes': [4, 8, 13],
        'max_features': [None, 'sqrt', 0.8],
    }
    grid = pre_process_grid(grid, prefix='regress__regressor__')
    cv = simple_grid_search_cv(estimator=model.pipeline,
                               param_grid=grid,
                               cv_splits=5,
                               scoring=SCORE_FUNCTION,
                               use_halving=True)
    grid_search(cv, X, y, model_selection_dir=paths.model_selection_dir)

def housing_grid_search_2():
    grid = [{
        'learning_rate': [10 ** -i],
        'n_estimators': [10 ** (i + 1)],
        'max_leaf_nodes': [8],
        'max_features': ['sqrt'],
    } for i in range(1, 5)]

    grid = pre_process_grid(grid, prefix='regress__regressor__')

    cv = simple_grid_search_cv(estimator=model.pipeline,
                               param_grid=grid,
                               cv_splits=5,
                               scoring=SCORE_FUNCTION,
                               use_halving=False)
    grid_search(cv, X, y, model_selection_dir=paths.model_selection_dir)


if __name__ == '__main__':
    housing_grid_search_2()
