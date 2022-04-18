import tensorflow as tf

from keras import layers, models


class BoltClassifier:

    def __init__(self):
        self.model = models.Sequential()

    def create_model(self):
        self.model.add(layers.Conv2D(32, 3, padding="same", activation="relu", input_shape=(216, 288, 1)))
        self.model.add(layers.MaxPool2D())

        self.model.add(layers.Conv2D(32, 3, padding="same", activation="relu"))
        self.model.add(layers.MaxPool2D())

        self.model.add(layers.Conv2D(64, 3, padding="same", activation="relu"))
        self.model.add(layers.MaxPool2D())
        self.model.add(layers.Dropout(0.4))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(128, activation="relu"))
        self.model.add(layers.Dense(11, activation="softmax"))
        return self.model
