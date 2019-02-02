import pandas
import numpy as np

import dataProc

class extraPhie(dataProc.Phie):
    def __init__(self, path, timeWindow, coordiantes, *args, **kwargs):
        csvData = pandas.read_csv(path, skiprows=12, header=None)
        self.raw = np.empty((csvData.shape[0]-1, csvData.shape[1]-1), dataProc.DATA_TYPE)
        self.raw = csvData[:-1][timeWindow[0]:timeWindow[1]]
        super(extraPhie, self).__init__(coordiantes, 1, timeWindow[1]-timeWindow[0], *args, **kwargs)