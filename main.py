import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as pyplot
import sklearn as sklearn
import cv2
import tensorflow as tf
from keras import layers, models
from sklearn.metrics import classification_report

from ImageResize import ImageResizer
from BoltClassifier import BoltClassifier
from DataAugmentation import DataAugmentation

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
    X_train = (train[:, :h - 1]) / 255
    X_train = X_train.reshape(np.shape(X_train)[0], 216, 288, 1)
    y_train = train[:, h - 1]
    X_test = (test[:, :h - 1]) / 255
    X_test = X_test.reshape(np.shape(X_test)[0], 216, 288, 1)
    y_test = test[:, h - 1]
    X_validate = (validate[:, :h - 1]) / 255
    X_validate = X_validate.reshape(np.shape(X_validate)[0], 216, 288, 1)
    y_validate = validate[:, h - 1]
    BC = BoltClassifier()
    DA = DataAugmentation()
    X_aug, y_aug = DA.aug_data(X_train, y_train, np.shape(X_train)[0])

    # model = tf.keras.models.load_model("./Models/prelim_model")

    model = BC.create_model()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(X_aug, y_aug, epochs=10, validation_data=(X_test, y_test))
    model.save("./Models/prelim_model")

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_test)
    # model.evaluate(X_test, y_test)
    print(classification_report(y_test, y_pred))
