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

def _train_disaster_tweets():
    from ml_soln.disaster_tweets import ctx
    model = ctx().model.model
    history = ctx().trainer.train()
    ctx().model_persistence.save_model(model, history)

def _train_housing_prices():
    from ml_soln.housing_prices.model import ctx
    ctx().model.fit()
    ctx().model.persist()

def _housing_prices_grid_search():
    from ml_soln.housing_prices.model import ctx
    ctx().model_selection.housing_grid_search_1()

TRAINERS: Dict[str, Trainer] = {
    t.name: t for t in (
        Trainer(
            name='digit_recognizer',
            func=_train_digit_recognizer
        ),
        Trainer(
            name='disaster_tweets',
            func=_train_disaster_tweets
        ),
        Trainer(
            name='housing_prices',
            func=_train_housing_prices
        ),
        Trainer(
            name='housing_prices_search',
            package_name='housing_prices',
            func=_housing_prices_grid_search
        ),
    )
}

PACKAGES = list(sorted(set(t.package_name for t in TRAINERS.values())))
