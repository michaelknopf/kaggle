import numpy as np
from keras import Model

from ml_soln.connectx import ctx
from ml_soln.connectx.connect_x_gym import ConnectXObservation, ConnectXConfiguration


def model_agent(model: Model):

    def agent(observation: ConnectXObservation,
              configuration: ConnectXConfiguration):
        """
        Test this with
        >>> ctx().kaggle_env.run([model_agent(model), 'random'])
        """

        # add dimension for the "mark" representing which player's turn it is
        state = observation.board + [observation.mark]

        x = np.array(state, dtype=np.float32)
        # add batch dimension of 1
        x = np.expand_dims(x, axis=0)

        y = model.predict(x)

        # Downgrade all illegal moves to be the lowest value in the array
        new_min = np.min(y) - 1
        for i in range(configuration.columns):
            # A move is illegal if that column is all the way filled up,
            # meaning its top row is already occupied (non-zero).
            # For example, board[0] is the left column of the top row.
            # If N=configuration.columns then board[N] is the last column of the top row.
            # board[N+1] is the first column of the 2nd row, so we don't need to check that.
            if observation.board[i] != 0:
                y[0][i] = new_min

        # return the next move (which column to play in)
        return int(np.argmax(y))

    return agent
