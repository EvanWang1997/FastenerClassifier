import tensorflow as tf

from keras import layers, models


class BoltClassifier:

    def __init__(self):
        self.model = models.Sequential()

    def create_model(self):
        self.model = models.Sequential()
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

    def create_hyper_model(self, hp):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(hp.Int('num_of_neurons', min_value=32, max_value=128, step=32), 3,
                                     padding="same", activation="relu", input_shape=(216, 288, 1)))
        self.model.add(layers.MaxPool2D())

        # providing the range for hidden layers
        for i in range(hp.Int('num_of_layers', 1, 10)):
            # providing range for number of neurons in hidden layers
            self.model.add(layers.Conv2D(hp.Int('num_of_neurons' + str(i), min_value=32, max_value=128, step=32), 3,
                                         padding="same", activation="relu"))
            self.model.add(layers.MaxPool2D())

        self.model.add(layers.Dropout(0.4))

        self.model.add(layers.Flatten())
        # providing the range for hidden layers
        for i in range(hp.Int('num_of_layers', 1, 10)):
            # providing range for number of neurons in hidden layers
            self.model.add(layers.Dense(units=hp.Int('num_of_neurons' + str(i), min_value=32, max_value=128, step=32),
                                        activation='relu'))
        self.model.add(layers.Dense(11, activation="softmax"))

        self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return self.model
