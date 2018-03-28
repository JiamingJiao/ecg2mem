#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import dataProc
import numpy as np
import glob
import math

electrodesNum = 25
imgRows = 200
imgCols = 200
channels = 1
srcPath = '/mnt/recordings/SimulationResults/mapping/2D/train/extra/*.jpg'
dstPath = '/mnt/recordings/SimulationResults/mapping/2D/train/extra_sparse'
dstStartNum = 100
fileName = glob.glob(srcPath)
src = np.ndarray((len(fileName), imgRows, imgCols, channels), dtype = np.float64)
src = dataProc.loadData(inputPath = srcPath)
kernelSize = 3
gaussianKernel = np.array([[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]])
temp = np.zeros((kernelSize, kernelSize, channels), np.float64)
stride = int(imgRows/(round(math.sqrt(electrodesNum))-1)-1+0.5)
dst = np.ndarray((len(fileName), imgRows, imgCols, channels), dtype = np.float64)
for imgNum in range(len(fileName)):
    for row in range(0, int(math.sqrt(electrodesNum)+0.5), stride):
        for col in range(0, int(math.sqrt(electrodesNum)+0.5), stride):
            temp = src[imgNum, row:row+kernelSize, col:col+kernelSize, :]
            cv.filter2D(temp, cv.CV_64F, gaussianKernel)
            temp[:, :, 0] = temp[int(kernelSize/2)+1, int(kernelSize/2)+1, 0]
            dst[imgNum, row:row+kernelSize, col:col+kernelSize, :] = temp
dst = dst.astype('float64')
for i in range(0, len(fileName)):
    dstFileName = dstPath + '/%d'%electrodesNum + '/%04d'%dstStartNum + '.jpg'
    cv.imwrite(dstFileName, dst[i])
    dstStartNum += 1
print('finished')
