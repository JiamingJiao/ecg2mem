#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
import glob
import shutil
import math
import random

def npyToPng(srcPath, dstPath):
    npy = loadData(srcPath)
    min = np.amin(npy)
    max = np.amax(npy)
    npy = 255*(npy-min)/(max-min)
    for i in range(0, npy.shape[0]):
        cv.imwrite(dstPath + "%06d"%i+".png", npy[i, :, :])
    print('completed')

def loadData(srcPath, resize = 0, rawRows = 200, rawCols = 200, imgRows = 256, imgCols = 256, normalization = 0, normalizationRange = [0., 1.],
approximateData = True):
    fileName = sorted(glob.glob(srcPath + '*.npy'))
    lowerBound = normalizationRange[0]
    upperBound = normalizationRange[1]
    if resize == 0:
        mergeImg = np.ndarray((len(fileName), rawRows, rawCols), dtype = np.float32)
    else:
        mergeImg = np.ndarray((len(fileName), imgRows, imgCols), dtype = np.float32)
        tempImg = np.ndarray((imgRows, imgCols), dtype = np.float32)
    rawImg = np.ndarray((rawRows, rawCols), dtype = np.float32)
    index = 0
    for i in fileName:
        rawImg = np.load(i)
        if resize == 1:
            mergeImg[index] = cv.resize(rawImg, (imgRows, imgCols), mergeImg[index], 0, 0, cv.INTER_NEAREST)
        else:
            mergeImg[index] = rawImg
        index += 1
    if approximateData == True:
        min = np.amin(mergeImg)
        max = np.amax(mergeImg)
        mergeImg = 255*(mergeImg-min)/(max-min)
        mergeImg = np.around(mergeImg)
        normalization = 1
    if normalization == 1:
        min = np.amin(mergeImg)
        max = np.amax(mergeImg)
        mergeImg = lowerBound + ((mergeImg-min)*(upperBound-lowerBound))/(max-min)
    return mergeImg

def generatePseudoECG(srcPath, dstPath):
    src = loadData(srcPath = srcPath, resize = 0)
    dst = np.ndarray(src.shape, dtype = np.float64)
    diffVKernel = np.zeros((3, 3, 1), dtype = np.float64)
    diffVKernel[1, :, 0] = 1
    diffVKernel[:, 1, 0] = 1
    diffVKernel[1, 1, 0] = -4
    diffV = np.ndarray((src.shape[1], src.shape[2]), dtype = np.float64)
    distance = np.ndarray((src.shape[1], src.shape[2]), dtype = np.float64)
    pseudoECG = np.ndarray((src.shape[1], src.shape[2]), dtype = np.float64)
    firstRowIndex = np.linspace(0, src.shape[1], num = src.shape[1], endpoint = False)
    firstColIndex = np.linspace(0, src.shape[2], num = src.shape[2], endpoint = False)
    colIndex, rowIndex = np.meshgrid(firstRowIndex, firstColIndex)
    for i in range(0, src.shape[0]):
        diffV = cv.filter2D(src = src[i], ddepth = -1, kernel = diffVKernel, dst = diffV, anchor = (-1, -1), delta = 0, borderType = cv.BORDER_REPLICATE)
        for row in range(0, src.shape[1]):
            for col in range(0, src.shape[2]):
                distance = cv.magnitude((rowIndex-row), (colIndex-col))
                pseudoECG[row,col] = cv.sumElems(cv.divide(diffV, distance))[0]
        dstFileName = dstPath + '%06d'%i
        np.save(dstFileName, pseudoECG)
    print('completed')

def downSample(srcPath, dstPath, samplePoints = (5, 5), interpolationSize = (200, 200)):
    src = loadData(srcPath)
    rowStride = math.floor(src.shape[1]/samplePoints[0])
    colStride = math.floor(src.shape[2]/samplePoints[1])
    multipleOfStride = ((samplePoints[0]-1)*rowStride+1, (samplePoints[1]-1)*colStride+1)
    temp = np.ndarray(multipleOfStride, dtype = np.float64) #Its size is a multiple of stride + 1
    sample = np.ndarray(samplePoints, dtype = np.float64)
    interpolated = np.ndarray(interpolationSize, dtype = np.float64)
    for i in range(0, src.shape[0]):
        temp = cv.resize(src[i, :, :], multipleOfStride)
        for j in range(0, samplePoints[0]):
            for k in range(0, samplePoints[1]):
                sample[j, k] = temp[j*rowStride, k*colStride]
        interpolated = cv.resize(sample, interpolationSize, interpolated, 0, 0, cv.INTER_NEAREST)
        dstFileName = dstPath + '%06d'%i
        np.save(dstFileName, interpolated)
    print('down sampling completed')

def generateSparsePseudoECG(srcPath, dstPath, samplePoints = (10, 10)):
    src = loadData(srcPath)
    rowStride = math.floor(src.shape[1]/samplePoints[0])
    colStride = math.floor(src.shape[2]/samplePoints[1])
    multipleOfStride = ((samplePoints[0]-1)*rowStride+1, (samplePoints[1]-1)*colStride+1)
    temp = np.ndarray(multipleOfStride, dtype = np.float64) #Its size is a multiple of stride + 1
    sample = np.ndarray(samplePoints, dtype = np.float64)
    diffVKernel = np.zeros((3, 3, 1), dtype = np.float64)
    diffVKernel[1, :, 0] = 1
    diffVKernel[:, 1, 0] = 1
    diffVKernel[1, 1, 0] = -4
    diffV = np.ndarray((src.shape[1], src.shape[2]), dtype = np.float64)
    firstRowIndex = np.linspace(0, src.shape[1], num = src.shape[1], endpoint = False)
    firstColIndex = np.linspace(0, src.shape[2], num = src.shape[2], endpoint = False)
    colIndex, rowIndex = np.meshgrid(firstRowIndex, firstColIndex)
    pseudoECG = np.ndarray(samplePoints, dtype = np.float64)
    interpolated = np.ndarray(src.shape, dtype = np.float64)
    for i in range(0, src.shape[0]):
        diffV = cv.filter2D(src = src[i], ddepth = -1, kernel = diffVKernel, dst = diffV, anchor = (-1, -1), delta = 0, borderType = cv.BORDER_REPLICATE)
        temp = cv.resize(src[i, :, :], multipleOfStride)
        for row in range(0, temp.shape[0]):
            for col in range(0, temp.shape[1]):
                distance = cv.magnitude((rowIndex-row*stride), (colIndex-col*stride))
                pseudoECG[row, col] = cv.sumElems(cv.divide(diffV, distance))[0]
        interpolated[i, :, :] = cv.resize(pseudoECG, (src.shape[1], src.shape[2]))
    return interpolated

def loadImage(srcPath, resize = 0, rawRows = 200, rawCols = 200, imgRows = 256, imgCols = 256, normalization = 0):
    fileName = glob.glob(srcPath + '*.png')
    if resize == 0:
        mergeImg = np.ndarray((len(fileName), rawRows, rawCols), dtype = np.float32)
    else:
        mergeImg = np.ndarray((len(fileName), imgRows, imgCols), dtype = np.float32)
        tempImg = np.ndarray((imgRows, imgCols), dtype = np.float32)
    rawImg = np.ndarray((rawRows, rawCols), dtype = np.float32)
    for i in range(0, len(fileName)):
        localName = srcPath + '%06d'%i + ".png"
        rawImg = cv.imread(localName, -1)
        if resize == 1:
            mergeImg[i] = cv.resize(rawImg, (imgRows, imgCols))
        else:
            mergeImg[i] = rawImg
    if normalization == 1:
        min = np.amin(mergeImg)
        max = np.amax(mergeImg)
        mergeImg = (mergeImg-min)/(max-min)
    return mergeImg

def create3DData(src, temporalDepth):
    framesNum = src.shape[0]
    paddingDepth = math.floor((temporalDepth-1)/2 + 0.1)
    dst = np.zeros((framesNum, temporalDepth, src.shape[1], src.shape[2]), dtype = np.float32)
    for i in range(0, paddingDepth):
        dst[i, paddingDepth-i:temporalDepth, :, :] = src[0:temporalDepth-paddingDepth+i, :, :]
        dst[framesNum-1-i, 0:temporalDepth-paddingDepth+i, :, :] = src[framesNum-(temporalDepth-paddingDepth)-i:framesNum, :, :]
    for i in range(paddingDepth, framesNum-paddingDepth):
        dst[i, :, :, :] = src[i-paddingDepth:i+paddingDepth+1, :, :]
    return dst

def clipData(srcPath, dstPath, bounds = [0., 1.]):
    src = loadData(srcPath)
    dst = np.clip(src, bounds[0], bounds[1])
    for i in range(0, src.shape[0]):
        dstFileName = dstPath + '%06d'%i
        np.save(dstFileName, dst[i])

def splitTrainAndVal(src1, src2, valSplit):
    srcLength = src1.shape[0]
    dataType = src1.dtype
    dimension1 = src1.ndim
    dimension2 = src2.ndim
    valNum = math.floor(valSplit*srcLength+0.1)
    randomIndexes = random.sample(range(0, srcLength), valNum)
    trainDataShape1 = np.ndarray((dimension1), dtype = np.uint8)
    valDataShape1 = np.ndarray((dimension1), dtype = np.uint8)
    trainDataShape1[0] = srcLength - valNum
    valDataShape1[0] = valNum
    trainDataShape1[1:dimension1] = src1.shape[1:dimension1]
    valDataShape1[1:dimension1] = src1.shape[1:dimension1]
    trainDataShape2 = np.ndarray((dimension2), dtype = np.uint8)
    valDataShape2 = np.ndarray((dimension2), dtype = np.uint8)
    trainDataShape2[0] = srcLength - valNum
    valDataShape2[0] = valNum
    trainDataShape2[1:dimension2] = src2.shape[1:dimension2]
    valDataShape2[1:dimension2] = src2.shape[1:dimension2]
    dst = [np.ndarray((trainDataShape1), dtype = dataType), np.ndarray((valDataShape1), dtype = dataType),
    np.ndarray((trainDataShape2), dtype = dataType), np.ndarray((valDataShape2), dtype = dataType)]
    dst[1] = np.take(src1, randomIndexes, 0)
    dst[0] = np.delete(src1, randomIndexes, 0)
    dst[3] = np.take(src2, randomIndexes, 0)
    dst[2] = np.delete(src2, randomIndexes, 0)
    return dst

def copyMassiveData(srcPathList, dstPath, potentialName):
    startNum = 0
    for srcPath in srcPathList:
        fileName = sorted(glob.glob(srcPath + potentialName + '*.npy'))
        for srcName in fileName:
            dst = np.load(srcName)
            dstFileName = dstPath + '%06d'%startNum
            np.save(dstFileName, dst)
            startNum += 1

def copyData(srcPath, dstPath, startNum = 0, endNum = None, shiftNum = 0):
    fileName = sorted(glob.glob(srcPath + '*.npy'))
    del fileName[endNum+1:len(fileName)]
    del fileName[0:startNum]
    for srcName in fileName:
        dst = np.load(srcName)
        dstFileName = dstPath + '%06d'%(startNum+shiftNum)
        np.save(dstFileName, dst)
        startNum += 1

'''
def annealingDownSample(srcPath, dst, maskPath)
'''
