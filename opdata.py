import numpy as np
import glob
import os
import matplotlib.cm

import dataProc

class OpVmem(dataProc.Vmem):
    def __init__(self, srcDir, rawSize, roiOrigin, roiSize, *args, **kwargs):
        pathList = sorted(glob.glob(os.path.join(srcDir, '*.raww')))
        self.raw = np.zeros((len(pathList), rawSize[0]*rawSize[1]), np.uint16)
        for i, path in enumerate(pathList):
            self.raw[i, :] = np.fromfile(path, np.uint16)
        self.raw = self.raw.reshape(((len(pathList),) + rawSize + (1,)))
        self.raw = self.raw[:, roiOrigin[0]:roiOrigin[0]+roiSize[0], roiOrigin[1]:roiOrigin[1]+roiSize[1]]
        self.raw = self.raw.astype(np.float64)
        self.vmem = np.zeros_like(self.raw)

        # map to [0, 1], different from that in opmap
        self.rawMax = np.amax(self.raw, axis=0)
        self.rawMin = np.amin(self.raw, axis=0)
        self.rawRange = (self.rawMax - self.rawMin) + (self.rawMax == self.rawMin)*1
        self.vmem = (self.rawMax - self.raw) / self.rawRange

        self.colorMap = None

        super(OpVmem, self).__init__(groups=1, length=len(pathList), height=roiSize[0], width=roiSize[1], **kwargs)
    
    def reduceNoise(self):
        pass
    
    def setColor(self, cmap='inferno'):
        mapper = matplotlib.cm.ScalarMappable(cmap=cmap)
        self.colorMap = np.empty((self.length, self.height, self.width, 4), np.float32)
        for i, frame in enumerate(self.vmem[:, :, :, 0]):
            self.colorMap[i] = mapper.to_rgba(frame, norm=False)
