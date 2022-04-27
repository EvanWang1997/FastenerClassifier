import tensorflow as tf
import keras as ks
import os
import numpy as np
import BoltClassifier
import keras_tuner as kt

import utils


class EnsembleModels:

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
        self.models = []
        if not os.path.exists(modelsfolder):
            os.makedirs(modelsfolder)

        for i in range(nummodels):
            model = kmfunction()
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            model.fit(xtrain, ytrain, epochs=epochs, validation_data=testingdata)
            self.models.append(model)
            model.save(modelsfolder + "/" + str(i))
            del model

    def saveModel(self, modelsfolder, model):
        if not os.path.exists(modelsfolder):
            os.makedirs(modelsfolder)

        model.save(modelsfolder)


    def createRandomDataModels(self, modelsfolder, nummodels, kmfunction, data, epochs):
        self.nummodels = nummodels
        self.models = []

        if not os.path.exists(modelsfolder):
            os.makedirs(modelsfolder)

        for i in range(nummodels):
            xtrain, ytrain, xtest, ytest = utils.train_and_test_split(data)
            model = kmfunction()
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            model.fit(xtrain, ytrain, epochs=epochs, validation_data=(xtest, ytest))
            self.models.append(model)
            model.save(modelsfolder + "/" + str(i))
            del model

    def createRandomColorModels(self, modelsfolder, nummodels, kmfunction, data, epochs):
        self.nummodels = nummodels
        self.models = []

        if not os.path.exists(modelsfolder):
            os.makedirs(modelsfolder)

        for i in range(nummodels):
            xtrain, ytrain, xtest, ytest = utils.color_train_test_split(data)
            model = kmfunction()
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            model.fit(xtrain, ytrain, epochs=epochs, validation_data=(xtest, ytest))
            self.models.append(model)
            model.save(modelsfolder + "/" + str(i))
            del model

    # Function for creating models from keras
    # Params:
    # modelfolder: Name of folder to persistently store all of the models desired
    # nummodels: Number of models desired for the array of models
    # kmfunction: Keras model creation function
    # xtrain: training input data for each of the models (images)
    # ytrain: training output data for each of the models (bolt class)
    # epochs: Number of epochs to be trained for, for each model
    # testeingdata: testing data used to validate each model during the epochs
    def createIndModels(self, modelsfolder, nummodels, kmfunction, xtrain, ytrain, epochs, testingdata):

        self.nummodels = nummodels
        self.models = []
        sample_ratio = 0.6
        if not os.path.exists(modelsfolder):
            os.makedirs(modelsfolder)

        for i in range(nummodels):
            print("start")
            random_indices = np.random.choice(xtrain.shape[0], size=int(sample_ratio * xtrain.shape[0]),
                                              replace=False)
            x_rand_sample = xtrain[random_indices, :]
            y_rand_sample = ytrain[random_indices]
            model = kmfunction()
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            model.fit(x_rand_sample, y_rand_sample, epochs=epochs, validation_data=testingdata)
            self.models.append(model)
            model.save(modelsfolder + "/" + str(i))
            print("completed ", i + 1, " models out of ", nummodels)
            del model

    # Function for creating models from keras
    # Params:
    # modelfolder: Name of folder to persistently store all of the models desired
    # nummodels: Number of models desired for the array of models
    # kmfunction: Keras model creation function
    # xtrain: training input data for each of the models (images)
    # ytrain: training output data for each of the models (bolt class)
    # epochs: Number of epochs to be trained for, for each model
    # testeingdata: testing data used to validate each model during the epochs
    def createHyperModels(self, modelsfolder, nummodels, kmfunction, xtrain, ytrain, epochs, testingdata):

        self.nummodels = nummodels
        self.models = []
        if not os.path.exists(modelsfolder):
            os.makedirs(modelsfolder)

        for i in range(nummodels):
            print("start")
            tuner = kt.RandomSearch(kmfunction,
                                 objective='val_accuracy',
                                 max_trials=5,
                                 executions_per_trial=3,
                                 directory='tuner1',
                                 project_name='Clothing')
            tuner.search_space_summary()
            stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            tuner.search(xtrain, ytrain, epochs=10, validation_data=testingdata, callbacks=[stop_early])
            tuner.results_summary()
            best_hps = tuner.get_best_hyperparameters()[0]
            model = tuner.hypermodel.build(best_hps)
            model.fit(xtrain, ytrain, epochs=epochs, validation_data=testingdata)
            self.models.append(model)
            model.save(modelsfolder + "/" + str(i))
            print("completed ", i + 1, " models out of ", nummodels)
            del model

    # Loads models from the specified models folder
    # Stores them all into the models array for an instance of this class
    # Params:
    # modelsfolder: string specifying the folder containing the models to be loaded in
    def loadModels(self, modelsfolder):
        self.models = []
        self.nummodels = 0

        for folder in os.listdir(modelsfolder):
            print(self.nummodels, "models loaded")
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
