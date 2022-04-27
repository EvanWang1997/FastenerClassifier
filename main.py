import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as pyplot
import sklearn as sklearn
import cv2
import tensorflow as tf
from keras import layers, models, Model
from keras.applications.densenet import DenseNet121
from keras.callbacks import ReduceLROnPlateau
from keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Dense

import utils
from sklearn.metrics import classification_report

from ImageResize import ImageResizer
from BoltClassifier import BoltClassifier
from RandomForestModels import RandomForestModels
from DataAugmentation import DataAugmentation


if __name__ == '__main__':
    np.random.seed(69420)
    # tf.config.list_physical_devices('GPU')
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    IR = ImageResizer()
    BC = BoltClassifier()
    DA = DataAugmentation()
    FM = RandomForestModels()
    FIM = RandomForestModels()
    data = IR.load_data("./Data/10%.pkl")
    data, X_validate, y_validate = utils.color_data_validation_split(data)
    X_train, y_train, X_test, y_test = utils.color_train_test_split(data)
    X_aug, y_aug = DA.aug_data(X_train, y_train, np.shape(X_train)[0])
    #
    # ret, threshTrain = cv2.threshold(X_train, 150, 255, cv2.THRESH_BINARY)
    # ret, threshTest = cv2.threshold(X_test, 150, 255, cv2.THRESH_BINARY)
    # ret, threshValidate = cv2.threshold(X_validate, 150, 255, cv2.THRESH_BINARY)

    # print(np.shape(threshTrain))

    # model = tf.keras.models.load_model("./Models/prelim_model")
    # densemodel = DenseNet121(weights='imagenet', include_top=False, input_shape=(216, 288, 3))
    # x = densemodel.output
    #
    # x = GlobalAveragePooling2D()(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    # x = Dense(1024, activation='relu')(x)
    # x = Dense(512, activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    #
    # preds = Dense(11, activation='softmax')(x)
    # model = Model(inputs=densemodel.input, outputs=preds)
    # for layer in model.layers[:-12]:
    #     layer.trainable = False
    #
    # for layer in model.layers[-12:]:
    #     layer.trainable = True
    # model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # metrics=['accuracy'])
    # anne = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3)
    # model.fit(X_aug, y_aug, epochs=10, callbacks=[anne], validation_data=(X_test, y_test))
    # model.save("./Models/dense1models")
    # y_pred = model.predict(X_validate)

    FM.createDenseModels("./Models/dense1models", 1, BC.create_dense_model, X_aug, y_aug, 10, (X_test, y_test))
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
