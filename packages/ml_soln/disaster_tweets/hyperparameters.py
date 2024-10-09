from dataclasses import dataclass

from ml_soln.common.dataclass_utils import DictClassMixin


@dataclass
class HyperParams(DictClassMixin):
    random_state: int = 0
    learning_rate: float = 1e-5
    epochs: int = 50
    batch_size: int = 32
    train_size: int = None
    test_size: int = 0.1

    # Learning Rate Reduction
    lrr_patience: int = 3
    lrr_factor: float = 0.5
    lrr_min_lr: float = 1e-5

    # Early Stopping
    es_patience: int = 5
    es_restore_best_weights: bool = True
