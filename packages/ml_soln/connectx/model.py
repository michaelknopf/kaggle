from functools import cached_property

import tensorflow as tf
from keras.api.layers import Layer, Input, Lambda, Reshape, Dense, Conv2D, Flatten, Concatenate, CategoryEncoding
from keras.api.models import Model as KerasModel

from ml_soln.connectx import ctx


class Model:

    @cached_property
    def model(self):
        return self.new_model()

    @classmethod
    def new_model(cls):
        n_rows = ctx().kaggle_env.configuration.rows
        n_cols = ctx().kaggle_env.configuration.columns

        board_input = Input(shape=(n_rows * n_cols,),
                            name='board')

        # Split into 2 channels, each containing one player's pieces only
        board_layers = SplitPlayerBoards()(board_input)
        board_layers = Reshape(target_shape=(n_rows, n_cols, 2))(board_layers)

        # Turn indicator input: shape (2,), a vector indicating whose turn it is
        turn_input = Input(shape=(1,), name='mark')
        turn_layers = CategoryEncoding(num_tokens=2,
                                       output_mode="one_hot")(turn_input - 1)

        # Convolutional layers to process the board and extract spatial features
        board_layers = Conv2D(64, kernel_size=(3, 3), activation='relu')(board_layers)
        board_layers = Conv2D(64, kernel_size=(3, 3), activation='relu')(board_layers)
        board_layers = Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same')(board_layers)
        board_layers = Flatten()(board_layers)

        # Concatenate the turn indicator with the flattened board features
        layers = Concatenate()([board_layers, turn_layers])

        # Fully connected layers
        layers = Dense(128, activation='relu')(layers)
        layers = Dense(64, activation='relu')(layers)

        # Output layer: n_cols possible actions (columns to drop a piece into)
        # Linear output is better for Q-values in DQN
        output = Dense(n_cols, activation='linear')(layers)

        # Build the model
        return KerasModel(inputs={'board': board_input, 'mark': turn_input}, outputs=output)


class SplitPlayerBoards(Layer):
    """
    Stack the board to create channels for each player (1, 2)
    """

    def call(self, inputs):
        return tf.stack([
            tf.cast(inputs == 1, dtype='float16'),
            tf.cast(inputs == 2, dtype='float16')
        ], axis=-1)

    def get_config(self):
        return super(SplitPlayerBoards, self).get_config()  # Required for serialization
