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
    IR.convert_folder_classes("./Data/TestClasses/", "group_data.csv")


