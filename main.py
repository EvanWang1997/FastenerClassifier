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
from ThresholdContour import ThresholdContour

if __name__ == '__main__':
    np.random.seed(42069)
    IR = ImageResizer()
    BC = BoltClassifier()
    DA = DataAugmentation()
    FM = RandomForestModels()
    data = IR.load_data("./Data/grey10%.pkl")
    datathresh = utils.thresh_all(data, 125)
    X_train, y_train, X_test, y_test, X_validate, y_validate = utils.process_and_split_data(data)
    # data = utils.thresh_all(data, 120)
    # X_validate, y_validate = utils.return_all_validation_data(data)
    # X_aug, y_aug = DA.aug_data(X_train, y_train, np.shape(X_train)[0])
    #
    # ret, threshTrain = cv2.threshold(X_train, 150, 255, cv2.THRESH_BINARY)
    # ret, threshTest = cv2.threshold(X_test, 150, 255, cv2.THRESH_BINARY)
    # ret, threshValidate = cv2.threshold(X_validate, 150, 255, cv2.THRESH_BINARY)

    FM.createRandomDataModels("./Models/5modelThreshOriginalData", 5, BC.create_model, data, 20)
    # FM.loadModels("./Models/5modelThresh")
    y_pred = FM.predictValues(X_validate)

    # model.evaluate(X_test, y_test)
    print(classification_report(y_validate, y_pred))
