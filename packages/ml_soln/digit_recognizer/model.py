from functools import cached_property, cache

from keras.api.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input
from keras.api.models import Sequential
from keras.api.optimizers import RMSprop


class Model:

    LOSS = 'categorical_crossentropy'
    METRICS = ['accuracy']

    @cached_property
    def model(self) -> Sequential:
        """
        CNN model
        """
        model = Sequential([
            Input(shape=(28, 28, 1)),
            Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'),
            Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(0.25),

            Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Dropout(0.25),

            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(10, activation="softmax"),
        ])
        self._compile(model)
        return model

    @cached_property
    def optimizer(self):
        return RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08)

    @cache
    def _compile(self, model):
        model.compile(optimizer=self.optimizer,
                      loss=self.LOSS,
                      metrics=self.METRICS)
