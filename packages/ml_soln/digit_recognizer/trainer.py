from keras.src.callbacks import ReduceLROnPlateau, History
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from ml_soln.digit_recognizer import ctx


class Trainer:

    @classmethod
    def train(cls):
        X_train, X_val, Y_train, Y_val = cls.split_data()
        return cls._train(X_train, X_val, Y_train, Y_val)

    @staticmethod
    def split_data():
        X, y = ctx().data_preparer.train_data()

        # Split the train and the validation set for the fitting
        X_train, X_val, Y_train, Y_val = train_test_split(
            X,
            y,
            train_size=ctx().hyperparams.train_size,
            test_size=ctx().hyperparams.test_size,
            random_state=ctx().hyperparams.random_state
        )

        return X_train, X_val, Y_train, Y_val

    @staticmethod
    def _train(X_train, X_val, Y_train, Y_val) -> History:
        # Set a learning rate annealer
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                    patience=3,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=0.00001)

        epochs = ctx().hyperparams.epochs
        batch_size = ctx().hyperparams.batch_size

        # With data augmentation to prevent overfitting (accuracy 0.99286)
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.1,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False  # randomly flip images
        )

        datagen.fit(X_train)

        # Fit the model
        datagen_flow = datagen.flow(X_train, Y_train, batch_size=batch_size)
        return ctx().model.model.fit(datagen_flow,
                                     epochs=epochs,
                                     validation_data=(X_val, Y_val),
                                     verbose=2,
                                     # steps_per_epoch=X_train.shape[0] // (batch_size * epochs),
                                     callbacks=[learning_rate_reduction])
