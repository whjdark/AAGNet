import numpy as np
from sklearn.cluster import MeanShift

class ClusterMethod():
    def __init__(self, X):
        self.X=X
    def MeanShift(self, Band_Width):
        ms = MeanShift(bandwidth=Band_Width)
        ms.fit(self.X)
        pred_ms = ms.predict(self.X)
        # print(ms)
        # print('mean-shift:', ms.labels_)
        # print('mean-shift:', np.unique(ms.labels_))
        return pred_ms
