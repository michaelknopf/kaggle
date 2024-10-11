from dataclasses import dataclass
from typing import Callable, Dict, Any


@dataclass
class Predictor:
    name: str
    func: Callable[[str], None]
    package_name: str = None

    def __post_init__(self):
        if not self.package_name:
            self.package_name = self.name

def _predict_digit_recognizer(job_name: str):
    from ml_soln.digit_recognizer.predict import predict_and_save
    predict_and_save(job_name=job_name)

def _predict_housing_prices(job_name: str):
    from ml_soln.housing_prices.predict import predict_and_save
    predict_and_save(job_name=job_name)

def _predict_disaster_tweets(job_name: str):
    from ml_soln.disaster_tweets.predict import predict_and_save
    predict_and_save(job_name=job_name)

PREDICTORS: Dict[str, Predictor] = {
    t.name: t for t in (
        Predictor(
            name='digit_recognizer',
            func=_predict_digit_recognizer
        ),
        Predictor(
            name='housing_prices',
            func=_predict_housing_prices
        ),
        Predictor(
            name='disaster_tweets',
            func=_predict_disaster_tweets
        )
    )
}
