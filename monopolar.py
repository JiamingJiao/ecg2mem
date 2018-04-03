#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import dataProc
import numpy as np
import glob
import math

samplingMethod = 'Gaussian'
electrodesNum = 25
imgRows = 200
imgCols = 200
channels = 1
srcPath = '/mnt/recordings/SimulationResults/mapping/2D/train/extra_for_fake/'
dstPath = '/mnt/recordings/SimulationResults/mapping/2D/train/sparse/25/extra_for_fake/'
startNum = 200
fileName = glob.glob(srcPath + '*.jpg')
src = np.ndarray((len(fileName), imgRows, imgCols, channels), dtype = np.float64)
src = dataProc.loadData(inputPath = srcPath, startNum = startNum, cvtDataType = 1)
kernelSize = 3
#gaussianKernel = cv.getGaussianKrnel(3, 0.2, cv.CV_64F)
temp = np.zeros((kernelSize, kernelSize, channels), np.float64)
stride = int(imgRows/(round(math.sqrt(electrodesNum))-1)-1+0.5)
dst = np.zeros((len(fileName), imgRows, imgCols, channels), dtype = np.float64)
for imgNum in range(len(fileName)):
    for row in range(0, imgRows, stride):
        for col in range(0, imgCols, stride):
            temp = src[imgNum, row:row+kernelSize, col:col+kernelSize, :]
            #cv.filter2D(temp, cv.CV_64F, gaussianKernel)
            if samplingMethod == 'Gaussian':
                temp = cv.GaussianBlur(temp, (kernelSize, kernelSize), 0.2, temp, 0.2)
            #temp[:, :, 0] = temp[int(kernelSize/2), int(kernelSize/2), 0]
            dst[imgNum, row:row+kernelSize, col:col+kernelSize, :] = temp[int(kernelSize/2), int(kernelSize/2), 0]
dst *= 255
dst = dst.astype('uint8')
for i in range(0, len(fileName)):
    dstFileName = dstPath + '%04d'%startNum + '.png'
    cv.imwrite(dstFileName, dst[i])
    startNum += 1
print('finished')
