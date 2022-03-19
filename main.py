import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sklearn
import cv2



if __name__ == '__main__':

    print("Running image conversion")
    img = cv2.imread('/Users/Evan Wang/Fasteners/Data/Test1.jpg', cv2.IMREAD_UNCHANGED)

    print('Original Dimensions : ', img.shape)

    scale_percent = 10  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print('Resized Dimensions : ', resized.shape)

    cv2.imshow("Resized image", resized)
    cv2.waitKey(0)
    cv2.imwrite('/Users/Evan Wang/Fasteners/Data/Test1Resized(10%).jpg', resized)
    cv2.destroyAllWindows()
