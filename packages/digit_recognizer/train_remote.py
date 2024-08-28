from keras.api.callbacks import ReduceLROnPlateau
from keras.api.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input
from keras.api.models import Sequential
from keras.api.optimizers import RMSprop
from keras.src.legacy.preprocessing.image import ImageDataGenerator


def train(X_train, X_val, Y_train, Y_val):
    # Set the CNN model
    # my CNN architecture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
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

    # Define the optimizer
    optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08)

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Set a learning rate annealer
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    epochs = 2  # Turn epochs to 30 to get 0.9967 accuracy
    batch_size = 86

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
        vertical_flip=False)  # randomly flip images

    datagen.fit(X_train)

    # Fit the model
    datagen_flow = datagen.flow(X_train, Y_train, batch_size=batch_size)
    return model.fit(datagen_flow,
                     epochs=epochs,
                     validation_data=(X_val, Y_val),
                     verbose=2,
                     steps_per_epoch=X_train.shape[0] // batch_size,
                     callbacks=[learning_rate_reduction])
