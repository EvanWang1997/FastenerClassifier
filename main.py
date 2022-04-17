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
from RandomForestModels import RandomForestModels

if __name__ == '__main__':
    np.random.seed(69420)
    IR = ImageResizer()
    data = IR.load_data("./Data/grey10%parallel.pkl")
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
    BC = BoltClassifier()

    FM = RandomForestModels()
    FM.loadModels("./Models")

    # model1 = keras.models.load_model("./Models/prelim_model_1")
    # model2 = keras.models.load_model("./Models/prelim_model_2")
    # model3 = keras.models.load_model("./Models/prelim_model_3")
    # model4 = keras.models.load_model("./Models/prelim_model_4")
    # model5 = keras.models.load_model("./Models/prelim_model_5")

    y_pred = FM.predictValues(X_validate)

    # model = BC.create_model()
    # model.summary()
    #
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    #
    # history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    # model.save("./Models/prelim_model_5")

    print(y_test)
    # model.evaluate(X_test, y_test)
    print(classification_report(y_validate, y_pred))
