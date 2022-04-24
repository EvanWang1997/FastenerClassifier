import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as pyplot
import sklearn as sklearn
import cv2
import tensorflow as tf
from keras import layers, models
import utils
from sklearn.metrics import classification_report

from ImageResize import ImageResizer
from BoltClassifier import BoltClassifier
from RandomForestModels import RandomForestModels
from DataAugmentation import DataAugmentation


if __name__ == '__main__':
    np.random.seed(69420)
    tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    IR = ImageResizer()
    BC = BoltClassifier()
    DA = DataAugmentation()
    FM = RandomForestModels()
    FIM = RandomForestModels()
    data = IR.load_data("./Data/grey10%.pkl")
    X_train, y_train, X_test, y_test, X_validate, y_validate = utils.process_and_split_data(data)
    # X_validate, y_validate = utils.return_all_validation_data(data)
    X_aug, y_aug = DA.aug_data(X_train, y_train, np.shape(X_train)[0])
    #
    # ret, threshTrain = cv2.threshold(X_train, 150, 255, cv2.THRESH_BINARY)
    # ret, threshTest = cv2.threshold(X_test, 150, 255, cv2.THRESH_BINARY)
    # ret, threshValidate = cv2.threshold(X_validate, 150, 255, cv2.THRESH_BINARY)

    # print(np.shape(threshTrain))

    # model = tf.keras.models.load_model("./Models/prelim_model")

    FM.createHyperModels("./Models/1models", 1, BC.create_hyper_model, X_aug, y_aug, 10, (X_test, y_test))
    # FM.loadModels("./Models/5models")
    # FIM.createIndModels("./Models/5indmodels", 5, BC.create_model, X_aug, y_aug, 10, (X_test, y_test))
    # FIM.loadModels("./Models/5indmodels")
    y_pred = FM.predictValues(X_validate)
    # y_ind_pred = FIM.predictValues(X_validate)

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
    # model.evaluate(X_test, y_test)
    print(classification_report(y_validate, y_pred))
    # print(classification_report(y_validate, y_ind_pred))
