#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import glob
import math

import dataProc

samplingMethod = 'Gaussian'
interpolation = 1
electrodesNum = 25
imgRows = 200
imgCols = 200
channels = 1
srcPath = '/mnt/recordings/SimulationResults/mapping/simulation_data/20171225-9/extra/'
dstPath = '/mnt/recordings/SimulationResults/mapping/2D/train/sparse/20180416/extra/'
startNum = 0
fileName = glob.glob(srcPath + '*.png')
src = np.ndarray((len(fileName), imgRows, imgCols, channels), dtype = np.float64)
src = dataProc.loadData(inputPath = srcPath, startNum = startNum, cvtDataType = 1)
kernelSize = 3
sqrtElectrodesNum = round(math.sqrt(electrodesNum))
samplingSite = np.zeros((sqrtElectrodesNum, sqrtElectrodesNum, len(fileName), kernelSize, kernelSize, channels), np.float64)
stride = int(imgRows/(round(math.sqrt(electrodesNum))-1)-1+0.5)
dst = np.zeros((len(fileName), imgRows, imgCols, channels), dtype = np.float64)
'''
for imgNum in range(len(fileName)):
    for row in range(0, imgRows, stride):
        for col in range(0, imgCols, stride):
            samplingSite = src[imgNum, row:row+kernelSize, col:col+kernelSize, :]
            if samplingMethod == 'Gaussian':
                samplingSite = cv.GaussianBlur(samplingSite, (kernelSize, kernelSize), 0.2, samplingSite, 0.2)
            dst[imgNum, row:row+kernelSize, col:col+kernelSize, :] = samplingSite[int(kernelSize/2), int(kernelSize/2), 0]
'''

downSamplingSize = kernelSize*math.sqrt(electrodesNum)
downSampling = np.ndarray((len(fileName), downSamplingSize, downSamplingSize, channels), dtype = np.float64)
for siteNum in range(0, electrodesNum):
    row = floor(siteNum/math.sqrt(electrodesNum)+0.1
    col = (electrodeNum%siteNum)
    samplingSite[row, col, :, :, :, :] = src[:, stride*row:stride*row+3, stride*col:stride*col+3, :]
dst *= 255
dst = dst.astype('uint8')
for i in range(0, len(fileName)):
    dstFileName = dstPath + '%04d'%startNum + '.png'
    cv.imwrite(dstFileName, dst[i])
    startNum += 1
print('completed')
