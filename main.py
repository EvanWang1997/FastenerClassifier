import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sklearn
import cv2
import tensorflow as tf
import keras.utils as ku

from ImageResize import ImageResizer
from BoltClassifier import BoltClassifier

if __name__ == '__main__':
    np.random.seed(69420)
    IR = ImageResizer()
    # IR.greyscale_all("./Data/Classes25%/", "./Data/Classesgrey25%/")
    # IR.resize_all("./Data/Classes/", "./Data/Classes25%/", 25)
    # IR.convert_folder_classes("./Data/Classes25%/", "25%.pkl")
    data = IR.load_data("./Data/grey10%.pkl")
    print(data[0])
    v, h = np.shape(data)
    np.random.shuffle(data)
    train, test, validate = np.split(data, [int(.6 * v), int(.9 * v)])
    X = data[:, :h - 1].reshape(v, 288, 216, 1)
    y = np.core.defchararray.replace(data[:, h - 1], '-', '')
    y = np.array([int(x) for x in y])
    print(type(X[1][1][1][0]))
    print(type(y[1]))
    BC = BoltClassifier()
    model = BC.create_model()
    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(X, y, epochs=10, validation_data=(X, y))
