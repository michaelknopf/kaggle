from keras.src.callbacks import History, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split

from ml_soln.disaster_tweets import ctx


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

        hp = ctx().hyperparams

        learning_rate_reduction = ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy',
                                                    patience=hp.lrr_patience,
                                                    verbose=1,
                                                    factor=hp.lrr_factor,
                                                    min_lr=hp.lrr_min_lr)

        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=hp.es_patience,
                                       restore_best_weights=hp.es_restore_best_weights,
                                       verbose=2)

        return ctx().model.model.fit(
            x=X_train,
            y=Y_train,
            batch_size=hp.batch_size,
            epochs=hp.epochs,
            validation_data=(X_val, Y_val),
            steps_per_epoch=X_train.shape[0] // (hp.batch_size * hp.epochs),
            callbacks=[learning_rate_reduction, early_stopping],
            verbose=2,
        )
