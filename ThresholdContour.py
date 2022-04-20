
import numpy as np
MaxPointValue = 255
MinPointValue = 0


class ThresholdContour:

    def __init__(self):
        return

    def ThresholdAllData(self, x, thresh):
        contourvector = np.vectorize(self.ContourPoint)
        return contourvector(x, thresh);


    def ContourPoint(self, point, thresh):
        if point >= thresh:
            return MaxPointValue
        else:
            return MinPointValue