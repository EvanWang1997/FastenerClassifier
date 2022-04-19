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
from DataRectifier import DataRectifier
from BoltClassifier import BoltClassifier
from EvanClassifier import EvanClassifier
from RandomForestModels import RandomForestModels

if __name__ == '__main__':
    np.random.seed(69420)
    IR = ImageResizer()
    DR = DataRectifier()
    DR.rectifier_data_float("./ValidationSet10%Grey/ValidationSet")
    # data = IR.load_data("./Data/grey10%parallel.pkl")
    # v, h = np.shape(data)
    # np.random.shuffle(data)
    # train, test, validate = np.split(data, [int(.6 * v), int(.9 * v)])
    # X_train = (train[:, :h - 1]) / 255
    # X_train = X_train.reshape(np.shape(X_train)[0], 288, 216, 1)
    # y_train = train[:, h - 1]
    # X_test = (test[:, :h - 1]) / 255
    # X_test = X_test.reshape(np.shape(X_test)[0], 288, 216, 1)
    # y_test = test[:, h - 1]
    # X_validate = (validate[:, :h - 1]) / 255
    # X_validate = X_validate.reshape(np.shape(X_validate)[0], 288, 216, 1)
    # y_validate = validate[:, h - 1]
    # BC = BoltClassifier()
    # EC = EvanClassifier()
    # FM = RandomForestModels()


    # FM.createModels("./Models/AveragePooling5", 5, EC.create_model, X_train, y_train, 20, (X_test, y_test))
    # FM.loadModels("./Models/AveragePooling")
    # y_pred = FM.predictValues(X_validate)

    # model = BC.create_model()
    # model.summary()
    #
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    #
    # history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    # model.save("./Models/prelim_model_5")

    # print(y_test)
    # # model.evaluate(X_test, y_test)
    # print(classification_report(y_validate, y_pred))
