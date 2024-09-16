from functools import cached_property

from ml_soln.common.context import BaseContext
from ml_soln.common.sagemaker_utils import sm_utils
from ml_soln.digit_recognizer.hyperparameters import HyperParams
from ml_soln.digit_recognizer.model import Model
from ml_soln.digit_recognizer.prepare_data import DataPreparer
from ml_soln.digit_recognizer.trainer import Trainer


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
    def hyperparams(self):
        return HyperParams.from_dict(sm_utils.hyperparams)
