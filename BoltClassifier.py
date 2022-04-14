import tensorflow as tf

from keras import layers, models


class BoltClassifier:

    def __init__(self):
        self.model = models.Sequential()

    def create_model(self):
        self.model.add(layers.Conv2D(288, (3, 3), activation='relu', input_shape=(288, 216, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(288, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(288, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(288, activation='relu'))
        self.model.add(layers.Dense(11))
        return self.model
