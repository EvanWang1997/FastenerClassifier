import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sklearn
import cv2
import tensorflow as tf
import keras.utils as ku
import keras
from sklearn.metrics import classification_report

from ImageResize import ImageResizer
from BoltClassifier import BoltClassifier

if __name__ == '__main__':
    np.random.seed(69420)
    IR = ImageResizer()
    # IR.greyscale_all("./Data/Classes25%/", "./Data/Classesgrey25%/")
    # IR.resize_all("./Data/Classes/", "./Data/Classes25%/", 25)
    # IR.convert_folder_classes("./Data/Classes25%/", "25%.pkl")
    data = IR.load_data("./Data/grey10%parallel.pkl")
    print(data[0])
    v, h = np.shape(data)
    np.random.shuffle(data)
    train, test, validate = np.split(data, [int(.6 * v), int(.9 * v)])
    X_train = (train[:, :h - 1]) / 255
    X_train = X_train.reshape(np.shape(X_train)[0], 288, 216, 1)
    y_train = train[:, h - 1]
    X_test = (test[:, :h - 1]) / 255
    X_test = X_test.reshape(np.shape(X_test)[0], 288, 216, 1)
    y_test = test[:, h - 1]
    X_validate = (validate[:, :h - 1]) / 255
    X_validate = X_validate.reshape(np.shape(X_validate)[0], 288, 216, 1)
    y_validate = validate[:, h - 1]
    # X = data[:, :h - 1].astype(float)
    # X = X.reshape(v, 288, 216, 1)
    # y = np.core.defchararray.replace(data[:, h - 1], '-', '')
    # y = y.astype(int)
    print(type(X_train[1][1][1][0]))
    print(type(y_train[1]))
    BC = BoltClassifier()

    # model = keras.models.load_model("./Models/prelim_model_parallel")

    model = BC.create_model()
    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    model.save("./Models/prelim_model_parallel")

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_test)
    # model.evaluate(X_test, y_test)
    print(classification_report(y_test, y_pred))
