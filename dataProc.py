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
        cv.imwrite(dstPath + "%06d"%i+".png",npy[i, :, :])
    print('completed')

def loadData(srcPath, startNum = 0, resize = 0, rawRows = 200, rawCols = 200, imgRows = 256, imgCols = 256, normalization = 0, normalizationRange = [0., 1.],
approximateData = False):
    fileName = glob.glob(srcPath + '*.npy')
    lowerBound = normalizationRange[0]
    upperBound = normalizationRange[1]
    if resize == 0:
        mergeImg = np.ndarray((len(fileName), rawRows, rawCols), dtype = np.float64)
    else:
        mergeImg = np.ndarray((len(fileName), imgRows, imgCols), dtype = np.float64)
        tempImg = np.ndarray((imgRows, imgCols), dtype = np.float64)
    rawImg = np.ndarray((rawRows, rawCols), dtype = np.float64)
    for i in range(0, len(fileName)):
        localName = srcPath + '%06d'%startNum + ".npy"
        rawImg = np.load(localName)
        if resize == 1:
            mergeImg[i] = cv.resize(rawImg, (imgRows, imgCols))
        else:
            mergeImg[i] = rawImg
        startNum += 1
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

# generate full size pseudo-ECG maps, and downsample them if it is necessary
def generatePseudoECG(srcPath, dstPath):
    src = loadData(srcPath = srcPath, startNum = 0, resize = 0)
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
        dstFileName = dstPath + '%04d'%i
        np.save(dstFileName, pseudoECG)
    print('completed')

def downSample(srcPath, dstPath, samplePoints = (5, 5), interpolationSize = (200, 200)):
    src = loadData(srcPath)
    rowStride = math.floor(src.shape[1]/samplePoints[0])
    colStride = math.floor(src.shape[2]/samplePoints[1])
    multipleOfStride = ((samplePoints[0]-1)*rowStride+1, (samplePoints[1]-1)*colStride+1)
    temp = np.ndarray(multipleOfStride, dtype = np.float64) #Its size is a multiple of stride + 1
    sample = np.ndarray(samplePoints, dtype = np.float64)
    for i in range(0, src.shape[0]):
        temp = cv.resize(src[i, :, :], multipleOfStride)
        for j in range(0, samplePoints[0]):
            for k in range(0, samplePoints[1]):
                sample[j, k] = temp[j*rowStride, k*colStride]
        interpolated = cv.resize(sample, interpolationSize)
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

def loadImage(srcPath, startNum = 0, resize = 0, rawRows = 200, rawCols = 200, imgRows = 256, imgCols = 256, normalization = 0):
    fileName = glob.glob(srcPath + '*.png')
    if resize == 0:
        mergeImg = np.ndarray((len(fileName), rawRows, rawCols), dtype = np.float64)
    else:
        mergeImg = np.ndarray((len(fileName), imgRows, imgCols), dtype = np.float64)
        tempImg = np.ndarray((imgRows, imgCols), dtype = np.float64)
    rawImg = np.ndarray((rawRows, rawCols), dtype = np.float64)
    for i in range(0, len(fileName)):
        localName = srcPath + '%06d'%startNum + ".png"
        rawImg = cv.imread(localName, -1)
        if resize == 1:
            #tempImg = cv.resize(rawImg, (imgRows, imgCols))
            #mergeImg[i] = img_to_array(tempImg)
            mergeImg[i] = cv.resize(rawImg, (imgRows, imgCols))
        else:
            #mergeImg[i] = img_to_array(rawImg)
            mergeImg[i] = rawImg
        startNum += 1
    if normalization == 1:
        min = np.amin(mergeImg)
        max = np.amax(mergeImg)
        mergeImg = (mergeImg-min)/(max-min)
    return mergeImg

def create3DData(src, temporalDepth):
    framesNum = src.shape[0]
    paddingDepth = math.floor((temporalDepth-1)/2 + 0.1)
    dst = np.zeros((framesNum, temporalDepth, src.shape[1], src.shape[2]), dtype = np.float64)
    for i in range(0, paddingDepth):
        dst[i, paddingDepth-i:temporalDepth, :, :] = src[0:temporalDepth-paddingDepth+i, :, :]
        dst[framesNum-1-i, 0:temporalDepth-paddingDepth+i, :, :] = src[framesNum-(temporalDepth-paddingDepth)-i:framesNum, :, :]
    for i in range(paddingDepth, framesNum-paddingDepth):
        dst[i, :, :, :] = src[i-paddingDepth:i+paddingDepth+1, :, :]
    return dst

def renameData(*srcPath,  dstPath, potentialName = 'phie'):
    dstDataID = 0
    for i in range(0, len(srcPath)):
        fileName = glob.glob(srcPath[i] + potentialName + '*.npy')
        fileNum = len(fileName)
        for j in range(0, fileNum):
            shutil.copy(srcPath[i] + potentialName + '_%04d'%j + '.npy', dstPath + '%06d'%dstDataID + '.npy', follow_symlinks = False)
            dstDataID += 1

def clipData(srcPath, dstPath, bounds = [0., 1.]):
    src = loadData(srcPath)
    dst = np.clip(src, bounds[0], bounds[1])
    for i in range(0, src.shape[0]):
        dstFileName = dstPath + '%06d'%i
        np.save(dstFileName, dst[i])
'''
def splitTrainAndVal(src, valSplit):
    valNum = math.floor(valSplit*src[0].shape[0]+0.1)
    randomIndices = random.sample(np.arrange(0, src[0].shape[0]-1), valNum)
    for i in range(0,len(src)):
        valArray[0] =
'''
