from dataclasses import dataclass
from ml_soln.common.dataclass_utils import DictClassMixin

@dataclass
class HyperParams(DictClassMixin):
    """
    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    """

    random_state: int = 0
    learning_rate: float = 10**-4
    n_estimators: int = 10**5
    max_leaf_nodes: int = 8
    max_features: str | float = 'sqrt'
