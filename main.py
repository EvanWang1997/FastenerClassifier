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

    IR.resize_percent("Test2.jpg", "Test2_10%.jpg", 10)
