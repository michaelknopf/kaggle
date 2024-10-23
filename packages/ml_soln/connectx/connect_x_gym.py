import random

from ml_soln.connectx import ctx
from ml_soln.connectx.agent import model_agent
from ml_soln.connectx.stubs import KaggleTrainer


class ConnectXGym:

    def __init__(self):
        self.player_order = 1
        self.opponent_index = -1

    def new_trainer(self) -> KaggleTrainer:
        self._roll_player_order()
        self._choose_opponent()
        agents = self._agent_pair()
        return ctx().kaggle_env.train(agents)

    @staticmethod
    def _new_model_agent():
        # create a fresh model to use as the opponent agent
        new_model = ctx().model.new_model()

        # copy parameter values from the model under training to the new model
        trained_model = ctx().model.model
        trained_vars = trained_model.trainable_variables
        new_vars = new_model.trainable_variables
        for trained_var, empty_var in zip(trained_vars, new_vars):
            empty_var.assign(trained_var.numpy())

        return model_agent(new_model)

    def _roll_player_order(self):
        if random.random() < ctx().hyperparams.switch_prob:
            self._switch_player_order()

    def _switch_player_order(self):
        self.player_order *= -1

    def _choose_opponent(self):
        r = random.random()

        # no opponent set yet
        if self.opponent_index == -1:
            switch_prob = 1

        # model
        elif self.opponent_index == 0:
            switch_prob = .2
        # negamax - don't stay on negamax for too long because it is computationally expensive
        elif self.opponent_index == 1:
            switch_prob = .75
        else:
            raise ValueError(f"Invalid opponent index: {self.opponent_index}")

        # keep same opponent
        if r > switch_prob:
            return

        self.opponent_index = (self.opponent_index + 1) % 2

        if self.opponent_index == 0:
            self.opponent = self._new_model_agent()
        else:
            self.opponent = 'negamax'

    def _agent_pair(self):
        return [None, self.opponent][::self.player_order]
