import cv2
import os
import numpy as np
import pickle
from ImageResize import ImageResizer

BoltClassMapping = {'11': 0,
                    '12': 1,
                    '13': 2,
                    '14': 3,
                    '15': 4,
                    '16': 5,
                    '21': 6,
                    '22': 7,
                    '23': 8,
                    '24': 9,
                    '25': 10}

MetricImperialMapping = {
    0:0,
    1:0,
    2:0,
    3:0,
    4:0,
    5:0,
    6:1,
    7:1,
    8:1,
    9:1,
    10:1
}

class DataRectifier:

    def __init__(self):
        self

    # Changes all data in a pickle filw to be
    def rectifier_data_float(self, datafilepath):

        IR = ImageResizer()
        data = IR.load_data(datafilepath)
        v, h = np.shape(data)
        X = data[:, :h - 1]
        y = np.core.defchararray.replace(data[:, h - 1], '-', '')
        y = np.array([self.class_map(yi) for yi in y])
        y = y.reshape(v,1)
        y = y.astype(np.float)
        X = X.astype(np.float)

        data = np.hstack((X, y))
        IR.store_data(datafilepath, data)

    def imperial_metric_datamap(self, data):
        v, h = np.shape(data)
        y = data[:, h-1]
        y = np.array([self.mi_map(yi) for yi in y])
        data[:, h-1] = y

        return data

    def class_map(self, bolt_number):
        return BoltClassMapping[bolt_number]

    def mi_map(self, bolt_number):
        return MetricImperialMapping[bolt_number]