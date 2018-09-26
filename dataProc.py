#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
import glob
import os
import shutil
import math
import random

DATA_TYPE = np.float32
INTERPOLATION_METHOD = cv.INTER_NEAREST # Use nearest interpolation if it is the last step, otherwise use cubic
NORM_RANGE = (0, 1)
VIDEO_ENCODER = 'H264'
VIDEO_FPS = 50
IMG_SIZE = (200, 200)
PSEUDO_ECG_CONV_KERNEL = np.zeros((3, 3, 1), dtype=DATA_TYPE)
PSEUDO_ECG_CONV_KERNEL[1, :, 0] = 1
PSEUDO_ECG_CONV_KERNEL[:, 1, 0] = 1
PSEUDO_ECG_CONV_KERNEL[1, 1, 0] = -4
ECG_FOLDER_NAME = 'pseudo_ecg'
MEM_FOLDER_NAME = 'mem'

class Calculation(object):
    def __init__(self, dst=None, shape=None):
        self.dst = dst
        if not type(self.dst) is np.ndarray:
            self.dst = np.ndarray(shape, dtype=DATA_TYPE)

    def calcPseudoEcg(self, src): # src: extra cellular potential map, dst: pseudo-ECG map
        diffV = np.ndarray(src.shape, dtype=DATA_TYPE)
        diffV = cv.filter2D(src=src, ddepth =-1, kernel=PSEUDO_ECG_CONV_KERNEL, dst=diffV, anchor=(-1, -1), delta=0, borderType=cv.BORDER_REPLICATE)
        distance = np.ndarray(src.shape, dtype=DATA_TYPE)
        firstRowIndex = np.linspace(0, src.shape[0], num=src.shape[1], endpoint=False)
        firstColIndex = np.linspace(0, src.shape[1], num=src.shape[1], endpoint=False)
        colIndex, rowIndex = np.meshgrid(firstRowIndex, firstColIndex)
        for row in range(0, src.shape[0]):
            for col in range(0, src.shape[1]):
                distance = cv.magnitude((rowIndex-row), (colIndex-col))
                self.dst[row,col] = cv.sumElems(cv.divide(diffV, distance))[0]
        return self.dst

    def downSample(self, src, samplePoints=(20, 20)):
        rowStride = math.floor(src.shape[0]/samplePoints[0])
        colStride = math.floor(src.shape[1]/samplePoints[1])
        multipleOfStride = ((samplePoints[0]-1)*rowStride+1, (samplePoints[1]-1)*colStride+1)
        resized = cv.resize(src, multipleOfStride)
        sample = np.ndarray(samplePoints, dtype=DATA_TYPE)
        for row in range(0, samplePoints[0]):
            for col in range(0, samplePoints[1]):
                sample[row, col] = resized[row*rowStride, col*colStride]
        self.dst = cv.resize(sample, src.shape, self.dst, 0, 0, INTERPOLATION_METHOD)
        return self.dst

    def calcSparsePseudoEcg(self, src, samplePoints = (20, 20)):
        rowStride = math.floor(src.shape[0]/samplePoints[0])
        colStride = math.floor(src.shape[1]/samplePoints[1])
        multipleOfStride = ((samplePoints[0]-1)*rowStride+1, (samplePoints[1]-1)*colStride+1)
        resized = np.ndarray(multipleOfStride, dtype=DATA_TYPE) #Its size is a multiple of stride
        diffV = np.ndarray(multipleOfStride, dtype=DATA_TYPE)
        firstRowIndex = np.linspace(0, resized.shape[0], num=resized.shape[0], endpoint=False)
        firstColIndex = np.linspace(0, resized.shape[1], num=resized.shape[1], endpoint=False)
        colIndex, rowIndex = np.meshgrid(firstRowIndex, firstColIndex)
        distance = np.ndarray(multipleOfStride, dtype=DATA_TYPE)
        pseudoEcg = np.ndarray(samplePoints, dtype=DATA_TYPE)
        resized = cv.resize(src, multipleOfStride, resized, 0, 0, cv.INTER_CUBIC)
        diffV = cv.filter2D(src=resized, ddepth=-1, kernel=PSEUDO_ECG_CONV_KERNEL, dst=diffV, anchor=(-1, -1), delta=0, borderType=cv.BORDER_REPLICATE)
        for row in range(0, samplePoints[0]):
            for col in range(0, samplePoints[1]):
                distance = cv.magnitude((rowIndex - row*rowStride), (colIndex - col*colStride))
                distance = distance.astype(DATA_TYPE)
                pseudoEcg[row, col] = cv.sumElems(cv.divide(diffV, distance))[0]
        self.dst = cv.resize(pseudoEcg, (src.shape[0], src.shape[1]), self.dst, 0, 0, INTERPOLATION_METHOD)
        return self.dst

    def clipData(self, src, bounds=(0, 1)):
        self.dst = np.clip(src, bounds[0], bounds[1])
        return self.dst

def callCalc(srcDir, dstDir, methodName, **kwargs):
    if not os.path.exists(dstDir):
        os.makedirs(dstDir)
    src = loadData(srcDir)
    dst = np.ndarray(src[0].shape, dtype=DATA_TYPE)
    calc = Calculation(dst=dst)
    method = getattr(calc, methodName)
    for i in range(0, src.shape[0]):
        dst = method(src=src[i], **kwargs)
        np.save(dstDir + '%06d'%i, dst)
    print('%s completed'%methodName)

def npyToPng(srcDir, dstDir):
    if not os.path.exists(dstDir):
        os.mkdir(dstDir)
    src = loadData(srcDir=srcDir, normalization=True, normalizationRange=(0, 255))
    for i in range(0, src.shape[0]):
        cv.imwrite(dstDir + "%06d"%i+".png", src[i, :, :])
    print('convert .npy to .png completed')

def loadData(srcDir, resize=False, srcSize=(200, 200), dstSize=(256, 256), normalization=False, normalizationRange=NORM_RANGE, addChannel=True):
    srcPathList = sorted(glob.glob(srcDir + '*.npy'))
    lowerBound = normalizationRange[0]
    upperBound = normalizationRange[1]
    if resize == False:
        dstSize = srcSize
    dst = np.ndarray((len(srcPathList), dstSize[0], dstSize[1]), dtype=DATA_TYPE)
    src = np.ndarray(srcSize, dtype=DATA_TYPE)
    index = 0
    for srcPath in srcPathList:
        src = np.load(srcPath)
        if resize == True:
            dst[index] = cv.resize(src, dstSize, dst[index], 0, 0, INTERPOLATION_METHOD)
        else:
            dst[index] = src
        index += 1
    if normalization == True:
        minValue = np.amin(dst)
        maxValue = np.amax(dst)
        dst = lowerBound + ((dst-minValue)*(upperBound-lowerBound))/(maxValue-minValue)
    if addChannel == True:
        dst = dst.reshape((dst.shape[0], dst.shape[1], dst.shape[2], 1))
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
    for i in range(0, framesNum+1):
        dst[i] = src[i:i+temporalDepth]
    return dst

def mergeSequence(srcDirList, electrodesNum, temporalDepth, netGName=None, srcSize=(200, 200), dstSize=(256, 256), normalizationRange=NORM_RANGE):
    ecgList = np.empty(len(srcDirList), dtype=object)
    memList = np.empty(len(srcDirList), dtype=object)
    i = 0
    for srcDir in srcDirList:
        srcEcg = loadData(srcDir=srcDir+ECG_FOLDER_NAME+'_%d/'%electrodesNum, resize=True, srcSize=srcSize, dstSize=dstSize, normalization=False)
        srcMem = loadData(srcDir=srcDir+MEM_FOLDER_NAME+'/', resize=True, srcSize=srcSize, dstSize=dstSize, normalization=False)
        if netGName=='convLstm' or netGName=='uNet3d':
            ecgList[i], memList[i] = create3dSequence(srcEcg, srcMem, temporalDepth, netGName)
        if netGName=='uNet':
            ecgList[i] = srcEcg
            memList[i] = srcMem
        i += 1
    ecg = np.concatenate(ecgList)
    mem = np.concatenate(memList)
    ecg, minEcg, maxEcg = normalize(ecg, normalizationRange)
    mem, minMem, maxMem = normalize(mem, normalizationRange)
    print('min ecg is %.8f'%minEcg)
    print('max ecg is %.8f'%maxEcg)
    print('min mem is %.8f'%minMem)
    print('max mem is %.8f'%maxMem)
    return [ecg, mem]

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
    dst = normalizationRange[0] + ((src-minValue)*(normalizationRange[1]-normalizationRange[0])) / (maxValue-minValue)
    return [dst, minValue, maxValue]

def scale(src, priorRange=None, dstRange=(0, 1)):
    dst = dstRange[0] + ((src-priorRange[0])*(dstRange[1]-dstRange[0])) / (priorRange[1]-priorRange[0])
    return dst

def makeVideo(srcDir, dstPath, frameRange=(-1, -1)):
    srcPathList = sorted(glob.glob(srcDir+'*.png'))
    if not frameRange[0] == -1:
        srcPathList = srcPathList[frameRange[0] : frameRange[1]+1]
    writer = cv.VideoWriter(filename=dstPath, fourcc=cv.VideoWriter_fourcc(*VIDEO_ENCODER), fps=VIDEO_FPS, frameSize=IMG_SIZE, isColor=False)
    for i in srcPathList:
        src = cv.imread(i, -1)
        writer.write(src)
    writer.release
    