import numpy as np
import tensorflow as tf

from keras import layers, models, Model
from keras.applications.densenet import DenseNet121
from keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Dense, Conv2D, MaxPool2D


class BoltClassifier:

    def __init__(self):
        self.x = None
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

    def create_dense_model(self, hp):
        densemodel = DenseNet121(weights='imagenet', include_top=False, input_shape=(216, 288, 3))
        x = densemodel.output

        # providing the range for hidden layers
        # for i in range(hp.Int('num_of_layers', 1, 10)):
        #     # providing range for number of neurons in hidden layers
        #     x = Conv2D(hp.Int('num_of_neurons' + str(i), min_value=32, max_value=128, step=32), 3,
        #                padding="same", activation="relu")(x)
        #     x = MaxPool2D()(x)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        # providing the range for hidden layers
        for i in range(hp.Int('num_of_layers', 1, 10)):
            # providing range for number of neurons in hidden layers
            x = Dense(units=hp.Int('num_of_neurons' + str(i), min_value=32, max_value=1024, step=124),
                      activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        preds = Dense(11, activation='softmax')(x)
        self.model = Model(inputs=densemodel.input, outputs=preds)
        print(np.shape(self.model.layers))
        trainable_layers = hp.Int('num_of_layers', min_value=8, max_value=408, step=40)
        for layer in self.model.layers[:-trainable_layers]:
            layer.trainable = False

        for layer in self.model.layers[-trainable_layers:]:
            layer.trainable = True
        self.model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        return self.model
