from functools import cached_property

from keras import Input
from keras.api.layers import Dense
from keras.api.models import Sequential

from ml_soln.connectx import ctx


class Model:

    @cached_property
    def model(self):
        return self.new_model()

    @staticmethod
    def new_model():
        return Sequential(
            [
                Input(shape=(ctx().num_states,))
            ] + [
                Dense(i, activation='sigmoid', kernel_initializer='RandomNormal')
                for i in ctx().hyperparams.hidden_units
            ] + [
                Dense(ctx().num_actions, activation='linear', kernel_initializer='RandomNormal')
            ]
        )
