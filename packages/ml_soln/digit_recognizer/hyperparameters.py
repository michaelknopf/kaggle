from dataclasses import dataclass
from ml_soln.common.dataclass_utils import DictClassMixin

@dataclass
class HyperParams(DictClassMixin):
    epochs: int = 30
    batch_size: int = 86
    train_size: int = None
    test_size: int = 0.1
    random_state: int = 2
