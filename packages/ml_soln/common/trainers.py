from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class Trainer:
    name: str
    func: Callable[[], None]
    package_name: str = None

    def __post_init__(self):
        if not self.package_name:
            self.package_name = self.name

def _train_digit_recognizer():
    from ml_soln.digit_recognizer import ctx
    model = ctx().model.model
    history = ctx().trainer.train()
    ctx().model_persistence.save_model(model, history)


TRAINERS: Dict[str, Trainer] = {
    t.name: t for t in (
        Trainer(
            name='digit_recognizer',
            func=_train_digit_recognizer
        ),
    )
}
