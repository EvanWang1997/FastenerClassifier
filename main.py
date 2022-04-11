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
    # IR.greyscale_all("./Data/Classes25%/", "./Data/Classesgrey25%/")
    # IR.resize_all("./Data/Classes/", "./Data/Classes25%/", 25)
    IR.convert_folder_classes("./Data/Classes25%/", "25%.pkl")
    # data = IR.load_data("./Data/Classes10%/10%.pkl")



