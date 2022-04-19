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
    np.random.seed(42069)
    IR = ImageResizer()
    BC = BoltClassifier()
    DA = DataAugmentation()
    FM = RandomForestModels()
    data = IR.load_data("./Data/grey10%.pkl")
    X_train, y_train, X_test, y_test, X_validate, y_validate = utils.process_and_split_data(data)
    # X_aug, y_aug = DA.aug_data(X_train, y_train, np.shape(X_train)[0])

    # model = tf.keras.models.load_model("./Models/prelim_model")

    FM.createModels("./Models/8models", 8, BC.create_model, X_train, y_train, 20, (X_test, y_test))
    # FM.loadModels("./Models/5models")
    y_pred = FM.predictValues(X_validate)


    # model.evaluate(X_test, y_test)
    print(classification_report(y_validate, y_pred))
