import pandas as pd
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
# noinspection PyUnresolvedReferences
from sklearn.experimental import enable_halving_search_cv
# noinspection PyProtectedMember
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from sklearn.model_selection._search import BaseSearchCV

from ml_soln.common.paths import Paths


class ModelSelection:

    def __init__(self, paths: Paths):
        self.paths = paths

    @staticmethod
    def simple_grid_search_cv(estimator: BaseEstimator,
                              param_grid,
                              cv_splits: int,
                              scoring=None,
                              use_halving=False):
        if use_halving:
            grid_search_cls = HalvingGridSearchCV
        else:
            grid_search_cls = GridSearchCV

        return grid_search_cls(estimator=estimator,
                               param_grid=param_grid,
                               scoring=scoring,
                               cv=cv_splits,
                               verbose=3)

    def grid_search(self,
                    cv: BaseSearchCV,
                    X: DataFrame,
                    y: Series):
        cv.fit(X, y)
        df = pd.DataFrame(cv.cv_results_)
        self.save_cv_results(df)
        print(f'Best params: {cv.best_params_}')
        print(f'Best score: {cv.best_score_}')
        return df

    @staticmethod
    def pre_process_grid(grid, prefix=None):
        if isinstance(grid, list):
            grids = grid
        else:
            grids = [grid]
        return [
            {prefix + k: v for k, v in g.items()}
            for g in grids
        ]

    def save_cv_results(self, cv_results: DataFrame):
        self.paths.output_data_dir.mkdir(exist_ok=True, parents=True)
        with open(self.paths.output_data_dir / f'selection.json', 'w') as f:
            cv_results.to_json(f, indent=2, orient='records')
        with open(self.paths.output_data_dir / f'selection.csv', 'w') as f:
            cv_results.to_csv(f, index=False)

    def train_and_cross_validate(self,
                                 estimator: BaseEstimator,
                                 X: DataFrame,
                                 y: Series,
                                 cv_splits=5,
                                 scoring=None):
        search_cv = self.simple_grid_search_cv(estimator,
                                               param_grid=[],
                                               cv_splits=cv_splits,
                                               scoring=scoring,
                                               use_halving=False)
        cv_results = self.grid_search(search_cv, X, y)

        print('')
        print(f'Score mean: {cv_results["mean_test_score"]:.2f}')
        print(f'Score standard deviation: {cv_results["std_test_score"]:.2f}')
        print(f'Total training time: {cv_results["mean_fit_time"] * cv_splits:.2f}')

        return cv_results
