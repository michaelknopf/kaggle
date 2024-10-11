from functools import cached_property

from ml_soln.common.context import BaseContext
from ml_soln.common.sagemaker_utils import sm_utils
from ml_soln.disaster_tweets.hyperparameters import HyperParams
from ml_soln.disaster_tweets.model import Model
from ml_soln.disaster_tweets.prepare_data import DataPreparer
from ml_soln.disaster_tweets.trainer import Trainer


class Context(BaseContext):

    @cached_property
    def data_preparer(self):
        return DataPreparer()

    @cached_property
    def model(self):
        return Model()

    @cached_property
    def trainer(self):
        return Trainer()

    @cached_property
    def hyperparams(self) -> HyperParams:
        return HyperParams.from_dict(sm_utils.hyperparams)
