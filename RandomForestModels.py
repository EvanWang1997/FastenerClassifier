import tensorflow as tf
import keras as ks
import os
import numpy as np
import BoltClassifier

class RandomForestModels:

    def __init__(self):
        self.models = []
        self.nummodels = 0
        return

    # Function for creating models from keras
    # Params:
    # modelfolder: Name of folder to persistently store all of the models desired
    # nummodels: Number of models desired for the array of models
    # kmfunction: Keras model creation function
    # xtrain: training input data for each of the models (images)
    # ytrain: training output data for each of the models (bolt class)
    # epochs: Number of epochs to be trained for, for each model
    # testeingdata: testing data used to validate each model during the epochs
    def createModels(self, modelsfolder, nummodels, kmfunction, xtrain, ytrain, epochs, testingdata):

        self.nummodels = nummodels

        if not os.path.exists(modelsfolder):
            os.makedirs(modelsfolder)

        for i in range(nummodels):
            model = kmfunction()
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            model.fit(xtrain, ytrain, epochs=epochs, validation_data=testingdata)
            self.models[i] = model
            model.save(modelsfolder + "/" + i)


    # Loads models from the specified models folder
    # Stores them all into the models array for an instance of this class
    # Params:
    # modelsfolder: string specifying the folder containing the models to be loaded in
    def loadModels(self, modelsfolder):
        self.models = []
        self.nummodels = 0

        for folder in os.listdir(modelsfolder):
            print(self.nummodels)
            self.models.append(ks.models.load_model(modelsfolder + "/" + folder))
            self.nummodels += 1


    # Gathers predictions based on each of the models stored in the class
    # Params:
    # X: Data whose classes we want to predict on
    def predictValues(self, X):
        n = np.shape(X)[0]

        print(X.shape)

        if (self.nummodels > 0):
            y1 = self.models[0].predict(X)
            k, c = np.shape(y1)
        else:
            return

        predprob = np.zeros((n, c))

        for i in range(self.nummodels):
            predprob += self.models[i].predict(X)

        y_pred = np.argmax(predprob, axis=1)
        return y_pred