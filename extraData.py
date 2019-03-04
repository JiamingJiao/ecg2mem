import pandas
import numpy as np
import scipy.interpolate

import dataProc

class extraPhie(dataProc.Phie):
    def __init__(self, path, timeWindow, coordiantes, *args, **kwargs):
        csvData = pandas.read_csv(path, skiprows=12, header=None)
        self.raw = np.empty((timeWindow[1]-timeWindow[0], csvData.shape[1]-1), dataProc.DATA_TYPE)
        self.raw[:, :] = csvData.iloc[timeWindow[0]: timeWindow[1], 0:-1]
        super(extraPhie, self).__init__(coordiantes, 1, timeWindow[1]-timeWindow[0], *args, **kwargs)
        
        # set data
        firstRowIndex = np.linspace(0, self.height, num=self.height, endpoint=False, dtype=dataProc.DATA_TYPE)
        firstColIndex = np.linspace(0, self.width, num=self.width, endpoint=False, dtype=dataProc.DATA_TYPE)
        colIndex, rowIndex = np.meshgrid(firstRowIndex, firstColIndex)
        grid = (rowIndex, colIndex)
        for i, frame in enumerate(self.twoD[0]):
            self.twoD[0, i, :, :, 0] = scipy.interpolate.griddata(self.coordinates, self.raw[i], grid, method='nearest')
