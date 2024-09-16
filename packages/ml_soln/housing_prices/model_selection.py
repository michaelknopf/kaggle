from ml_soln.housing_prices import ctx
# noinspection PyUnresolvedReferences
from sklearn.experimental import enable_halving_search_cv

from ml_soln.common.model_selection import ModelSelection
from ml_soln.housing_prices.model import SCORE_FUNCTION

HYPERPARAM_PREFIX = 'regress__regressor__'

class HousingPricesModelSelection:

    def __init__(self):
        self.X, self.y = ctx().data_preparer.train_data()
        self.model_selection = ModelSelection(ctx().paths)

    def _grid_search(self, grid, cv_splits=5):
        grid = self.model_selection.pre_process_grid(grid, prefix=HYPERPARAM_PREFIX)
        cv = self.model_selection.simple_grid_search_cv(estimator=ctx().model.pipeline,
                                                        param_grid=grid,
                                                        cv_splits=cv_splits,
                                                        scoring=SCORE_FUNCTION,
                                                        use_halving=True)
        self.model_selection.grid_search(cv, self.X, self.y)

    def housing_grid_search_1(self):
        grid = {
            'learning_rate': [10**-1, 10**-2],
            'n_estimators': [10**3, 10**4],
            'min_samples_leaf': [1, 3, 5, 9],
            'max_leaf_nodes': [4, 8, 13],
            'max_features': [None, 'sqrt', 0.8],
        }
        self._grid_search(grid)

    def housing_grid_search_2(self):
        grid = [{
            'learning_rate': [10 ** -i],
            'n_estimators': [10 ** (i + 1)],
            'max_leaf_nodes': [8],
            'max_features': ['sqrt'],
        } for i in range(1, 5)]
        self._grid_search(grid)

if __name__ == '__main__':
    HousingPricesModelSelection().housing_grid_search_2()
