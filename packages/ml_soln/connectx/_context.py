from functools import cached_property

from kaggle_environments import make

from ml_soln.common.context import BaseContext
from ml_soln.common.sagemaker_utils import sm_utils
from ml_soln.connectx.connect_x_gym import KaggleTrainer
from ml_soln.connectx.hyperparameters import HyperParams
from ml_soln.connectx.model import Model
from ml_soln.connectx.train import Trainer


class Context(BaseContext):

    @cached_property
    def kaggle_env(self):
        return make('connectx', debug=True)

    @cached_property
    def connect_x_gym(self) -> KaggleTrainer:
        return self.kaggle_env.train([None, 'random'])

    @cached_property
    def trainer(self):
        return Trainer()

    @cached_property
    def hyperparams(self) -> HyperParams:
        return HyperParams.from_dict(sm_utils.hyperparams)

    @cached_property
    def model(self):
        return Model()

    @cached_property
    def action_space_dim(self):
        return self.kaggle_env.configuration.columns

    @cached_property
    def observation_space_dim(self):
        return self.kaggle_env.configuration.columns * self.kaggle_env.configuration.rows

    @cached_property
    def num_states(self):
        # add 1 for the mark (whose turn it is)
        return self.observation_space_dim + 1

    @cached_property
    def num_actions(self):
        return self.action_space_dim
