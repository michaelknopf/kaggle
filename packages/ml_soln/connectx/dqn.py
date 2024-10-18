import random
from dataclasses import dataclass
from typing import List

import tensorflow as tf
import numpy as np
from keras import Model
from keras.src.optimizers import Adam

from ml_soln.common.dataclass_utils import DictClassMixin
from ml_soln.connectx import ctx
from ml_soln.connectx.connect_x_gym import ConnectXObservation


@dataclass
class Experience(DictClassMixin):

    state: ConnectXObservation
    """
    Previous state, right before taking action
    """

    action: int
    """
    The action to take (number of the column to play in)
    """

    reward: float
    """
    Reward resulting from the action
    """

    next_state: ConnectXObservation
    """
    Next state after taking action
    """

    done: bool
    """
    Whether the episode is complete
    """


class DQN:

    def __init__(self, model: Model):
        self.model = model
        self.num_actions = ctx().num_actions
        self.batch_size = ctx().hyperparams.batch_size
        self.optimizer = Adam(ctx().hyperparams.lr)
        self.gamma = ctx().hyperparams.gamma
        self.experiences: List[Experience] = []
        self.max_experiences = ctx().hyperparams.max_experiences
        self.min_experiences = ctx().hyperparams.min_experiences

    def predict(self, inputs):
        inputs = inputs.astype('float32')
        inputs = np.atleast_2d(inputs)
        return self.model(inputs)

    def train(self, target_dqn):
        # Only start the training process when we have enough experiences in the buffer
        if len(self.experiences) < self.min_experiences:
            return

        # Randomly select a batch of experiences from the buffer
        experiences: List[Experience] = random.sample(self.experiences, k=self.batch_size)

        states = np.array([self.pre_process(e.state) for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])

        # Prepare labels for training process
        next_states = np.array([self.pre_process(e.next_state) for e in experiences])
        dones = np.array([e.done for e in experiences])
        target_next_move_probs = np.max(target_dqn.predict(next_states), axis=1)
        actual_values = np.where(dones,
                                 rewards,
                                 rewards + self.gamma * target_next_move_probs)

        with tf.GradientTape() as tape:
            prediction = self.predict(states)
            action_vector = tf.one_hot(actions, self.num_actions)
            selected_actions = prediction * action_vector
            selected_action_values = tf.math.reduce_sum(selected_actions, axis=1)
            # sum of squared errors
            loss = tf.math.reduce_sum(
                tf.square(actual_values - selected_action_values)
            )

        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def get_action(self, state: ConnectXObservation, epsilon: float):
        """
        Use epsilon-greedy to decide next action.
        """
        if np.random.random() < epsilon:
            # explore new random move
            return self.random_move(state)
        else:
            # use the current model's best prediction
            return self.predict_move(state)

    def predict_move(self, state):
        board_and_mark = self.pre_process(state)
        prediction = self.predict(np.atleast_2d(board_and_mark))[0].numpy()

        # make illegal moves have the lowest value
        new_min = np.min(prediction) - 1
        for i in range(self.num_actions):
            if state.board[i] != 0:
                prediction[i] = new_min

        return int(np.argmax(prediction))

    def random_move(self, state):
        return random.choice(self.legal_moves(state))

    def legal_moves(self, state):
        return [c for c in range(self.num_actions) if state.board[c] == 0]

    def add_experience(self, exp):
        # pop items from back of buffer if it is full
        while len(self.experiences) >= self.max_experiences:
            self.experiences.pop(0)
        self.experiences.append(exp)

    def copy_weights(self, dqn):
        variables1 = self.model.trainable_variables
        variables2 = dqn.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)

    @staticmethod
    def pre_process(state: ConnectXObservation):
        """
        Add the mark to the board, indicating which player's turn it is.
        """
        return state.board + [state.mark]
