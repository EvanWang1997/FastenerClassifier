import tensorflow as tf

import numpy as np
import os
import keras
import random
import cv2
import math
import utils
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Convolution2D,BatchNormalization
from tensorflow.keras.layers import Flatten,MaxPooling2D,Dropout

from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import warnings
warnings.filterwarnings("ignore")

class DensenetClassifier:

    def __init__(self, outputclasses = 11):
        self.outputclasses = outputclasses

    def train_model(self, data, epochs, outputclasses=2):
        model_d = DenseNet201(weights='imagenet', include_top=False, input_shape=(216, 288, 3))

        x = model_d.output

        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        preds = Dense(outputclasses, activation='softmax')(x)  # FC-layer

        # Prepares model
        model = Model(inputs=model_d.inputs, outputs=preds)
        model.summary()

        #Specified that training is limited to certain layers
        for layer in model.layers[:-8]:
            layer.trainable = False

        for layer in model.layers[-8:]:
            layer.trainable = True

        model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        model.summary()

        (xtrain, ytrain, xtest, ytest) = utils.color_train_test_split(data)

        anne = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3)
        checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)

        datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True, shear_range=0.2)

        datagen.fit(xtrain)
        # Fits-the-model
        history = model.fit_generator(datagen.flow(xtrain, ytrain, batch_size=128),
                                      steps_per_epoch=xtrain.shape[0] // 128,
                                      epochs=epochs,
                                      verbose=2,
                                      callbacks=[anne, checkpoint],
                                      validation_data=(xtrain, ytrain))

        ypred = model.predict(xtest)

        total = 0
        accurate = 0
        accurateindex = []
        wrongindex = []

        for i in range(len(ypred)):
            if np.argmax(ypred[i]) == np.argmax(ytest[i]):
                accurate += 1
                accurateindex.append(i)
            else:
                wrongindex.append(i)

            total += 1

        print('Total-test-data;', total, '\taccurately-predicted-data:', accurate, '\t wrongly-predicted-data: ',
              total - accurate)
        print('Accuracy:', round(accurate / total * 100, 3), '%')

        Ypred = model.predict(xtest)

        n = np.shape(ytest)[0]
        ytest = np.reshape(ytest, (n,1))
        print(np.shape(Ypred))
        print(np.shape(ytest))
        Ypred = np.argmax(Ypred, axis=1)
        Ytrue = np.argmax(ytest, axis=1)


        cm = confusion_matrix(Ytrue, Ypred)
        plt.figure(figsize=(12, 12))
        ax = sns.heatmap(cm, cmap="rocket_r", fmt=".01f", annot_kws={'size': 16}, annot=True, square=True,
                         xticklabels="ypred", yticklabels="ytest")
        ax.set_ylabel('Actual', fontsize=20)
        ax.set_xlabel('Predicted', fontsize=20)
        plt.show()

        return model