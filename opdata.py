import numpy as np
import glob
import os

import dataProc

class OpVmem(dataProc.Vmem):
    def __init__(self, srcDir, rawSize, roi, *args, **kwargs):
        pathList = sorted(glob.glob(os.path.join(srcDir, '*.raww')))
        self.raw = np.zeros((len(pathList), rawSize[0]*rawSize[1]), np.uint16)
        for i, path in enumerate(pathList):
            self.raw[i, :] = np.fromfile(path, np.uint16)
        self.raw = self.raw.reshape(((len(pathList),) + rawSize + (1,)))
        self.raw = self.raw[:, roi[0][0]:roi[1][0], roi[0][1]:roi[1][1]]
        self.raw = self.raw.astype(np.float64)

        super(OpVmem, self).__init__(groups=1, length=len(pathList), **kwargs)

    def calcVmem(self):
        # map to [0, 1], different from that in opmap
        rawMax = np.max(self.raw, axis=0)
        rawMin = np.min(self.raw, axis=0)
        rawRange = (rawMax - rawMin) + (rawMax == rawMin)*1
        self.raw = (rawMax - self.raw) / rawRange
    
    def reduceNoise(self):
        pass
        