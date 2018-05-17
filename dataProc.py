#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
import glob
import os
import math
#import keras
#from keras.preprocessing.image import array_to_img, img_to_array

def rename(srcPath, dstPath, deletePart = 'phie_'):
    fileName = glob.glob(srcPath + deletePart + '*.npy')
    for i in range(0, len(fileName)):
        os.rename(srcPath + deletePart + '%04d.npy'%i, dstPath + '%04d.npy'%i)
    print(delePart + 'completed')

def npyToPng(srcPath, dstPath):
    npy = loadData(srcPath)
    min = np.amin(npy)
    max = np.amax(npy)
    for i in range(0, 1000, 1):
        npy[i, :, :] = 255*(npy[i, :, :]-min)/(max-min)
        cv.imwrite(dstPath + "%04d"%i+".png",npy[i, :, :])
    print('completed')

def loadData(inputPath, startNum = 0, resize = 0, rawRows = 200, rawCols = 200, imgRows = 256, imgCols = 256, channels = 1):
    fileName = glob.glob(inputPath + '*.npy')
    if resize == 0:
        mergeImg = np.ndarray((len(fileName), rawRows, rawCols), dtype = np.float64)
    else:
        mergeImg = np.ndarray((len(fileName), imgRows, imgCols), dtype = np.float64)
        tempImg = np.ndarray((imgRows, imgCols), dtype = np.float64)
    rawImg = np.ndarray((rawRows, rawCols), dtype = np.float64)
    for i in range(0, len(fileName)):
        localName = inputPath + '%04d'%startNum + ".npy"
        rawImg = np.load(localName)
        if resize == 1:
            #tempImg = cv.resize(rawImg, (imgRows, imgCols))
            #mergeImg[i] = img_to_array(tempImg)
            mergeImg[i] = cv.resize(rawImg, (imgRows, imgCols))
        else:
            #mergeImg[i] = img_to_array(rawImg)
            mergeImg[i] = rawImg
        startNum += 1
    return mergeImg

def generatePseudoECG(srcPath, dstPath, electrodesNum, zDistance = 100, interpolationSize = (200, 200)):
    src = loadData(inputPath = srcPath, startNum = 0, resize = 0)
    dst = np.ndarray(src.shape, dtype = np.float64)
    diffVKernel = np.zeros((3, 3, 1), dtype = np.float64)
    diffVKernel[1, :, 0] = 1
    diffVKernel[:, 1, 0] = 1
    diffVKernel[1, 1, 0] = -4
    diffV = np.ndarray(src.shape, dtype = np.float64)
    distance = np.ndarray((src.shape[1], src.shape[2]), dtype = np.float64)
    rowDistance = np.ndarray((src.shape[1], src.shape[2]), dtype = np.float64)
    colDistance = np.ndarray((src.shape[1], src.shape[2]), dtype = np.float64)
    firstRowIndex = np.linspace(0, src.shape[1], num = src.shape[1], endpoint = False)
    firstColIndex = np.linspace(0, src.shape[2], num = src.shape[2], endpoint = False)
    colIndex, rowIndex = np.meshgrid(firstRowIndex, firstColIndex)
    pseudoECG = np.ndarray((math.floor(math.sqrt(electrodesNum)+0.1), math.floor(math.sqrt(electrodesNum)+0.1)), dtype = np.float64)
    stride = int(src.shape[1]/(round(math.sqrt(electrodesNum))-1)-1+0.5)
    zDistanceSquare = zDistance*zDistance
    for i in range(0, src.shape[0]):
        diffV = cv.filter2D(src = src[i], ddepth = -1, kernel = diffVKernel, dst = diffV, anchor = (-1, -1), delta = 0, borderType = cv.BORDER_REPLICATE)
        for siteNum in range(0, electrodesNum):
            cellRow = math.floor(siteNum/math.sqrt(electrodesNum)+0.1)
            cellCol = siteNum%math.floor(math.sqrt(electrodesNum)+0.1)
            rowDistance = rowIndex - cellRow * stride
            colDistance = colIndex - cellCol * stride
            distance = cv.sqrt(cv.magnitude(rowDistance, rowDistance) + cv.magnitude(colDistance, colDistance) + zDistanceSquare)
            pseudoECG[cellRow, cellCol] = cv.sumElems(cv.divide(diffV, distance))[0]
        interpolatedECG = cv.resize(pseudoECG, interpolationSize)
        dstFileName = dstPath + '%04d'%i
        np.save(dstFileName, interpolatedECG)
    print('completed')
