from dataclasses import dataclass
from typing import List

from ml_soln.common.dataclass_utils import DictClassMixin

@dataclass
class HyperParams(DictClassMixin):
    hidden_units: List[int] = (100, 200, 200, 100)
    gamma: float = 0.99
    episodes: int = 100
    max_experiences: int = 10000
    min_experiences: int = 100
    batch_size: int = 32
    lr: float = 1e-2
    start_epsilon: float = 0.99
    min_epsilon: float = 0.1
    decay: float = 0.99999
    copy_step: int = 25
