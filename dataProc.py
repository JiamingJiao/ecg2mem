#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
import glob
import os
import shutil
import math
import random
import scipy.interpolate

import time

DATA_TYPE = np.float32
CV_DATA_TYPE = cv.CV_32F
INTERPOLATION_METHOD = cv.INTER_NEAREST
NORM_RANGE = (0, 1)
VIDEO_ENCODER = 'XVID'
VIDEO_FPS = 50
IMG_SIZE = (200, 200)
PSEUDO_ECG_CONV_KERNEL = np.zeros((3, 3, 1), dtype=DATA_TYPE)
PSEUDO_ECG_CONV_KERNEL[1, :, 0] = 1
PSEUDO_ECG_CONV_KERNEL[:, 1, 0] = 1
PSEUDO_ECG_CONV_KERNEL[1, 1, 0] = -4
ECG_FOLDER_NAME = 'pseudo_ecg'

class SparsePecg(object):
    def __init__(self, shape, coordinates, roi=-1):
        self.coordinates = coordinates
        self.shape = shape
        firstRowIndex = np.linspace(0, shape[0], num=shape[0], endpoint=False, dtype=DATA_TYPE)
        firstColIndex = np.linspace(0, shape[1], num=shape[1], endpoint=False, dtype=DATA_TYPE)
        colIndex, rowIndex = np.meshgrid(firstRowIndex, firstColIndex)
        self.grid = (rowIndex, colIndex)
        self.distance = np.ndarray(((coordinates.shape[0],) + shape), dtype=DATA_TYPE)
        for i in range(0, coordinates.shape[0]):
            cv.magnitude((rowIndex - coordinates[i, 0]), (colIndex - coordinates[i, 1]), self.distance[i])
        self.pseudoEcg = np.ndarray(coordinates.shape[0], dtype=DATA_TYPE)
        self.diffV = np.ndarray(shape, dtype=DATA_TYPE)
        self.quotient = np.ndarray(shape, dtype=DATA_TYPE)
        self.dst = np.ndarray(shape, dtype=DATA_TYPE)
        if roi == -1:
            self.roi = [[0, 0], [shape[0]-1, shape[1]-1]]
        else:
            self.roi = roi
        self.dstInRoi = np.ndarray((self.roi[1][0]-self.roi[0][0], self.roi[1][1]-self.roi[0][1]), dtype=DATA_TYPE)

    def calcPecg(self, src):
        cv.filter2D(src=src, ddepth=CV_DATA_TYPE, kernel=PSEUDO_ECG_CONV_KERNEL, dst=self.diffV, anchor=(-1, -1), delta=0, borderType=cv.BORDER_REPLICATE)
        for i in range(0, self.coordinates.shape[0]):
            cv.divide(self.diffV, self.distance[i], dst=self.quotient)
            self.pseudoEcg[i] = cv.sumElems(self.quotient)[0]
        self.dst = scipy.interpolate.griddata(self.coordinates, self.pseudoEcg, self.grid, method='nearest')
        self.dstInRoi = np.copy(self.dst[self.roi[0][0]:self.roi[1][0], self.roi[0][1]:self.roi[1][1]])

    def callCalc(self, srcDirList, dstDirList):
        length = len(srcDirList)
        for i in range(0, length):
            srcPath = srcDirList[i]
            src = loadData(srcPath)
            dstPath = dstDirList[i]
            if not os.path.exists(dstPath):
                os.mkdir(dstPath)
            for i in range(0, src.shape[0]):
                self.calcPecg(src[i])
                np.save(dstPath+'%06d'%i, self.dstInRoi)
            print('%s completed'%dstDirList[i])

class AccurateSparsePecg(SparsePecg):
    def __init__(self, srcShape, removeNum, fullCoordinatesShape, parentCoordinates, coordinatesDir):
        self.srcShape = srcShape
        rowStride = math.floor(srcShape[0]/fullCoordinatesShape[0])
        colStride = math.floor(srcShape[1]/fullCoordinatesShape[1])
        self.multipleOfStride = ((fullCoordinatesShape[0]-1)*rowStride+1, (fullCoordinatesShape[1]-1)*colStride+1)
        coordinates = removePoints(parentCoordinates, removeNum)
        super(AccurateSparsePecg, self).__init__(self.multipleOfStride, coordinates)
        if not coordinatesDir == 'None':
            np.save(coordinatesDir, self.coordinates)
        else:
            print('coordinates were not saved!')
    
    def resizeAndCalc(self, srcDirList, dstDirList):
        resizedSrc = np.ndarray((self.multipleOfStride+(1,)), dtype=DATA_TYPE)
        resizedDst = np.ndarray((self.srcShape+(1,)), dtype=DATA_TYPE)
        length = len(srcDirList)
        for i in range(0, length):
            srcPath = srcDirList[i]
            src = loadData(srcPath)
            dstPath = dstDirList[i]
            if not os.path.exists(dstPath):
                os.mkdir(dstPath)
            for i in range(0, src.shape[0]):
                cv.resize(src[i], self.multipleOfStride, resizedSrc, 0, 0, cv.INTER_CUBIC)
                self.calcPecg(resizedSrc)
                cv.resize(self.dst, self.srcShape, resizedDst, 0, 0, INTERPOLATION_METHOD)
                np.save(dstPath+'%06d'%i, resizedDst)
            print('%s completed'%dstPath)

class Data(object):
    def __init__(self, groups, length, height, width, channels):
        self.groups = groups
        self.length = length
        self.twoD = np.zeros((groups, length, height, width, channels), dtype=DATA_TYPE)
        self.threeD = None
    
    def set2dData(self, dirList):
        for i in range(0, len(dirList)):
            self.twoD[i, :, :, :, :] = loadData(dirList[i])

    def set3dData(self, temporalDepth):
        validLength = self.length - temporalDepth + 1
        shape = ((self.groups, validLength, temporalDepth) + self.twoD.shape[2:])
        strides = (self.twoD.strides[0:2] + (self.twoD.strides[1],) + self.twoD.strides[2:]) # Expand dim_of_length to dim_of_length * dim_of_temporalDepth
        expanded = np.lib.stride_tricks.as_strided(self.twoD, shape=shape, strides=strides, writeable=False) # (groups, validLength, temporalDepth, height, width, channels)

        shape = ((self.groups*validLength,) + expanded.shape[2:]) # Merge the dim of groups and the dim of length
        #strides = ((expanded.strides[2],) + expanded.strides[2:])
        #self.threeD = np.lib.stride_tricks.as_strided(expanded, shape=shape, strides=strides, writeable=False) # This is wrong
        self.threeD = expanded.reshape(shape) # This is a view (?)

def createDirList(dataDir, nameList, potentialName=''):
    length = len(nameList)
    dst = [None]*length
    for i in range(0, length):
        dst[i] = dataDir + nameList[i] + '/' + potentialName
    return dst

def randomCoordinates(pointsNum, limit):
    sampledPoints = random.sample(range(0, limit[0]*limit[1]), pointsNum)
    dst = np.ndarray((pointsNum, 2), dtype=np.uint16)
    dst[:,0], dst[:,1] = np.divmod(sampledPoints, limit[1])
    return dst

def uniformCoordinates(shape, limit):
    firstRowIndex = np.linspace(0, limit[0]-1, num=shape[0], dtype=DATA_TYPE)
    firstColIndex = np.linspace(0, limit[1]-1, num=shape[1], dtype=DATA_TYPE)
    colIndex, rowIndex = np.meshgrid(firstRowIndex, firstColIndex)
    dst = np.ndarray((shape[0]*shape[1], 2), dtype=np.uint16)
    dst[:,0] = rowIndex.flatten()
    dst[:,1] = colIndex.flatten()
    return dst

def removePoints(src, num=0):
    deletedPoints = random.sample(range(0, src.shape[0]), num)
    dst = np.delete(src, deletedPoints, 0)
    return dst

def clipData(src, bounds=(0, 1)):
    dst = np.clip(src, bounds[0], bounds[1])
    return dst

def npyToPng(srcDir, dstDir):
    if not os.path.exists(dstDir):
        os.mkdir(dstDir)
    src = loadData(srcDir=srcDir)
    src, _, _ = normalize(src, (0, 255))
    for i in range(0, src.shape[0]):
        cv.imwrite(dstDir + "%06d"%i+".png", src[i, :, :])
    print('convert .npy to .png completed')

def loadData(srcDir, resize=False, dstSize=(256, 256), withChannel=True): # doesn't work with multi-channel image
    srcPathList = sorted(glob.glob(srcDir + '*.npy'))
    length = len(srcPathList)
    sample = np.load(srcPathList[0])
    src = np.ndarray(sample.shape, dtype=DATA_TYPE)
    if resize == False:
        dstSize = (sample.shape[0], sample.shape[1])
    if sample.ndim == 2:
        dst = np.ndarray((length,)+dstSize, dtype=DATA_TYPE)
    elif sample.ndim == 3:
        if resize == True:
            dst = np.ndarray(((length,)+dstSize), dtype=DATA_TYPE)
        else:
            dst = np.ndarray(((length,)+sample.shape), dtype=DATA_TYPE)
    for i in range(0, length):
        src = np.load(srcPathList[i])
        src = src.astype(DATA_TYPE)
        if resize == True:
            cv.resize(src, dstSize, dst[i], 0, 0, INTERPOLATION_METHOD)
        else:
            dst[i] = src
    if withChannel==True:
        dst = dst.reshape((length,)+dstSize+(1,))
    else:
        dst = dst.reshape((length,)+dstSize)
    return dst

def create3dSequence(srcEcg, srcMem, temporalDepth, netGName):
    dstEcg = create3dEcg(srcEcg, temporalDepth, netGName)
    if netGName == 'convLstm':
        startFrame = temporalDepth
    if netGName == 'uNet3d':
        startFrame = math.floor(temporalDepth/2 + 0.1)
    dstMem = srcMem[startFrame:dstEcg.shape[0]+startFrame]
    return dstEcg, dstMem

def create3dEcg(src, temporalDepth, netGName):
    if netGName == 'convLstm':
        framesNum = src.shape[0]-temporalDepth
    elif netGName == 'uNet3d':
        framesNum = src.shape[0] - 2*math.floor(temporalDepth/2 + 0.1)
    dst = np.zeros((framesNum, temporalDepth, src.shape[1], src.shape[2], src.shape[3]), dtype=DATA_TYPE)
    for i in range(0, framesNum):
        dst[i] = src[i:i+temporalDepth]
    return dst

def mergeSequence(pecgDirList, memDirList, temporalDepth, netGName=None, srcSize=(200, 200), dstSize=(256, 256), normalizationRange=NORM_RANGE):
    length = len(pecgDirList)
    ecg = np.empty(length, dtype=object)
    mem = np.empty(length, dtype=object)
    for i in range(0, length):
        srcEcg = loadData(srcDir=pecgDirList[i], resize=True, dstSize=dstSize)
        srcMem = loadData(srcDir=memDirList[i], resize=True, dstSize=dstSize)
        if netGName=='convLstm' or netGName=='uNet3d':
            ecg[i], mem[i] = create3dSequence(srcEcg, srcMem, temporalDepth, netGName)
        if netGName=='uNet':
            ecg[i] = srcEcg
            mem[i] = srcMem
    del srcEcg, srcMem
    ecg = np.concatenate(ecg)
    mem = np.concatenate(mem)
    ecg, minEcg, maxEcg = normalize(ecg, normalizationRange)
    mem, minMem, maxMem = normalize(mem, normalizationRange)
    dataRange = [minEcg, maxEcg, minMem, maxMem]
    print('min ecg is %.8f'%minEcg)
    print('max ecg is %.8f'%maxEcg)
    print('min mem is %.8f'%minMem)
    print('max mem is %.8f'%maxMem)
    return [ecg, mem, dataRange]

def splitTrainAndVal(src1, src2, valSplit):
    srcLength = src1.shape[0]
    valNum = math.floor(valSplit*srcLength + 0.1)
    randomIndices = random.sample(range(0, srcLength), valNum)
    val1 = np.take(src1, randomIndices, 0)
    train1 = np.delete(src1, randomIndices, 0)
    val2 = np.take(src2, randomIndices, 0)
    train2 = np.delete(src2, randomIndices, 0)
    return [train1, val1, train2, val2]

def copyMassiveData(srcDirList, dstDir, potentialName):
    startNum = 0
    for srcDir in srcDirList:
        fileName = sorted(glob.glob(srcDir + potentialName + '*.npy'))
        for srcName in fileName:
            dst = np.load(srcName)
            np.save(dstDir + '%06d'%startNum, dst)
            startNum += 1

def copyData(srcPath, dstPath, startNum=0, endNum=None, shiftNum=0):
    fileName = sorted(glob.glob(srcPath + '*.npy'))
    del fileName[endNum+1:len(fileName)]
    del fileName[0:startNum]
    for srcName in fileName:
        dst = np.load(srcName)
        dstFileName = dstPath + '%06d'%(startNum+shiftNum)
        np.save(dstFileName, dst)
        startNum += 1
    
def normalize(src, normalizationRange=NORM_RANGE):
    minValue = np.amin(src)
    maxValue = np.amax(src)
    np.add(src, -minValue, src)
    factor = (normalizationRange[1]-normalizationRange[0]) / (maxValue-minValue)
    np.multiply(src, factor, src)
    return [src, minValue, maxValue]

def scale(src, priorRange=None, dstRange=(0, 1)):
    dst = dstRange[0] + ((src-priorRange[0])*(dstRange[1]-dstRange[0])) / (priorRange[1]-priorRange[0])
    return dst

def makeVideo(srcDir, dstPath, frameRange=(-1, -1), padding=(0, 0)):
    srcPathList = sorted(glob.glob(srcDir+'*.png'))
    if not frameRange[0] == -1:
        srcPathList = srcPathList[frameRange[0]:]
    if not frameRange[1] == -1:
        srcPathList = srcPathList[:frameRange[1]-frameRange[0]]
    sample = cv.imread(srcPathList[0], -1)
    paddingArray = np.zeros_like(sample)
    writer = cv.VideoWriter(filename=dstPath, fourcc=cv.VideoWriter_fourcc(*VIDEO_ENCODER), fps=VIDEO_FPS, frameSize=IMG_SIZE, isColor=True)
    for i in range(0, padding[0]):
        writer.write(paddingArray)
    for i in srcPathList:
        src = cv.imread(i, -1)
        writer.write(src)
    for i in range(0, padding[1]):
        writer.write(paddingArray)
    writer.release
