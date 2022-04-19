import numpy as np


def process_and_split_data(data):
    v, h = np.shape(data)
    np.random.shuffle(data)
    train, test, validate = np.split(data, [int(.6 * v), int(.9 * v)])
    X_train = (train[:, :h - 1]) / 255
    X_train = X_train.reshape(np.shape(X_train)[0], 216, 288, 1)
    y_train = train[:, h - 1]
    X_test = (test[:, :h - 1]) / 255
    X_test = X_test.reshape(np.shape(X_test)[0], 216, 288, 1)
    y_test = test[:, h - 1]
    X_validate = (validate[:, :h - 1]) / 255
    X_validate = X_validate.reshape(np.shape(X_validate)[0], 216, 288, 1)
    y_validate = validate[:, h - 1]
    return X_train, y_train, X_test, y_test, X_validate, y_validate

def train_and_test_split(data):
    v, h = np.shape(data)
    np.random.shuffle(data)
    train, test = np.split(data, [int(.7 * v)])
    X_train = (train[:, :h - 1]) / 255
    X_train = X_train.reshape(np.shape(X_train)[0], 216, 288, 1)
    y_train = train[:, h - 1]
    X_test = (test[:, :h - 1]) / 255
    X_test = X_test.reshape(np.shape(X_test)[0], 216, 288, 1)
    y_test = test[:, h - 1]

    return X_train, y_train, X_test, y_test

def return_all_validation_data(data):
    v, h = np.shape(data)
    X_validate = (data[:, :h - 1]) / 255
    X_validate = X_validate.reshape(np.shape(X_validate)[0], 216, 288, 1)
    y_validate = data[:, h - 1]

    return X_validate, y_validate