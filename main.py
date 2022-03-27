import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sklearn
import cv2

from ImageResize import ImageResizer


if __name__ == '__main__':

    IR = ImageResizer()
    # IR.resize_all("./Data/TestClasses/", "./Data/TestClasses10%/", 10)
    IR.convert_folder_classes("./Data/TestClasses/", "group_data.pkl")
    data = IR.load_data("./Data/TestClasses/group_data.pkl")
    print(data.shape)


