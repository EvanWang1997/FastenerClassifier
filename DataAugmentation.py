import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from copy import deepcopy


class DataAugmentation:

    def __init__(self):
        self.da = ImageDataGenerator(rotation_range=90, horizontal_flip=True, vertical_flip=True)

    # Function for augmenting data
    # Params:
    # X: numpy_ndarray containing samples to augment
    # y: numpr_array containing labels of samples
    # batch_size: amount of samples to augment at a time in X
    # returns:
    # X_concat: concatenation of original and augmented samples
    # y_concat: concatenation of labels of original and augmented samples
    def aug_data(self, X, y, batch_size):
        X_copy = deepcopy(X)
        data_aug = self.da.flow(X_copy, y, batch_size=batch_size)
        X_batch, y_batch = data_aug.next()
        X_concat = np.concatenate((X, X_batch), axis=0)
        y_concat = np.concatenate((y, y_batch))
        return X_concat, y_concat
