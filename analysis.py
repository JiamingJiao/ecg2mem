#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import math
import os
import glob
import matplotlib.pyplot as plt
import keras.backend as K

import opmap.videoData
import opmap.phaseMap
import opmap.phaseVarianceMap
import opmap.PhaseVariancePeakMap

import dataProc
import model

BLUE = '#0070c1'

def calculateMae(src1, src2):
    difference = np.subtract(src1, src2)
    absDifference = np.abs(difference)
    mae = np.mean(absDifference)
    std = np.std(difference, ddof=1)
    return [mae, std]

def histEqualizationArray(src, depth = 8):
    maxIntensity = 2**depth - 1
    temp = 0.
    dst = np.zeros((maxIntensity+1), dtype = np.float64)
    pixNum = src.shape[0]*src.shape[1]
    histogram = cv.calcHist([src], [0], None, [maxIntensity+1], [0, maxIntensity+1])
    histogram = histogram.astype(np.float64)
    for i in range(0, maxIntensity+1):
        dst[i] = temp + maxIntensity*histogram[i]/(pixNum)
        temp = dst[i]
    dst = dst.astype(np.int32)
    return dst

def calculateHistTransArray(srcHist, dstHist, depth = 8):
    maxIntensity = 2**depth - 1
    dst = np.zeros((maxIntensity+1), dtype = np.int32)
    accumulation = 0
    for i in range(0, maxIntensity+1):
        diff = maxIntensity+1
        for j in range(0, maxIntensity + 1):
            if abs(dstHist[j] - srcHist[i])<diff:
                diff = abs(dstHist[j] - srcHist[i])
                dst[i] = j
    return dst

def histSpecification(src, transArray):
    dst = np.zeros(src.shape, dtype = np.int32)
    for i in range(0, src.shape[0]):
        for j in range(0, src.shape[1]):
            dst[i, j] = transArray[src[i, j]]
    return dst

def plotEcg(srcDir, dstDir, coordinate=(100, 100), xrange=(-1, -1), ylimit=(-100, 0), xlabel=('Time / ms'), ylabel=('Membrane potential / mV')):
    src = dataProc.loadData(srcDir)
    ecg = src[:, coordinate[0], coordinate[1], :]
    y = ecg.flatten()
    if not xrange[0] == -1:
        y = y[xrange[0]:]
    if not xrange[1] == -1:
        y = y[:xrange[1]-xrange[0]]
    length = y.shape[0]
    x = np.linspace(1, length, num=length)
    figure = plt.figure(figsize=(9, 3), dpi=300)
    plt.plot(x, y, linewidth=1, color=BLUE, figure=figure)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylimit)
    plt.show()
    if not os.path.exists(dstDir):
        os.mkdir(dstDir)
    figure.savefig(dstDir+'%d_%d.png'%coordinate, bbox_inches='tight')

class IntermediateLayers(model.Networks):
    def __init__(self, src, netName = 'uNet', weightsPath = None, **kwargs):
        super(IntermediateLayers, self).__init__(**kwargs)
        self.netName = netName
        if self.netName == 'uNet':
            resizedSrc = np.ndarray((self.imgRows, self.imgCols), dtype = np.float32)
            resizedSrc = cv.resize(src, (self.imgRows, self.imgCols), resizedSrc, 0, 0, cv.INTER_NEAREST)
            self.input = resizedSrc[np.newaxis, :, :, np.newaxis]
            self.network = super(IntermediateLayers, self).uNet()
        self.network.load_weights(weightsPath)
        self.inputTensor = self.network.layers[0].input

    def intermediateFeatures(self, layerName):
        layer = self.network.get_layer(layerName)
        layerFunc = K.function([self.inputTensor, K.learning_phase()], [layer.output])
        dst = layerFunc([self.input, 0])[0] # output in test mode, learning_phase = 0
        return dst

    # save all features
    def saveFeatures(self, dstPath, encoderStartLayer = 1, encoderEndLayer = 5, decoderStartLayer = 3, decoderEndLayer = 9, normalizationMode = 'layer', resize = 0):
        features = np.empty(encoderEndLayer-encoderStartLayer+1+decoderEndLayer-decoderStartLayer+1, dtype = object)
        layersList = list()
        for encoderNum in range(encoderStartLayer, encoderEndLayer+1, 1):
            layersList.append('encoder' + '%d'%encoderNum)
        for decoderNum in range(decoderStartLayer, decoderEndLayer+1, 1):
            layersList.append('decoder' + '%d'%decoderNum)
        # calculate intermediate layers
        layerNum = 0
        maxValue = -np.inf
        minValue = np.inf
        minMax = open(dstPath + 'min_max.txt', 'w+')
        for layerName in layersList:
            features[layerNum] = self.intermediateFeatures(layerName)
            layerMax = np.amax(features[layerNum])
            layerMin = np.amin(features[layerNum])
            minMax.write('%s: %f, %f\n'%(layerName, layerMin, layerMax))
            if normalizationMode == 'layer':
                features[layerNum] = 255*(features[layerNum] - layerMin)/(layerMax - layerMin)
            if normalizationMode == 'all':
                if layerMax > maxValue:
                    maxValue = layerMax
                if layerMin < minValue:
                    minValue = layerMin
            layerNum += 1
        minMax.close()
        if normalizationMode == 'all':
            features = 255*(features - minValue)/(maxValue - minValue)
        # resize and save
        resizedDst = np.ndarray((self.imgRows, self.imgCols), dtype = np.uint8)
        layerNum = 0
        for layerName in layersList:
            #encoderFeatures[encoderNum-startLayer] = 255*(encoderFeatures[encoderNum-startLayer]-min)/(max-min)
            layerPath = dstPath + layerName
            if not os.path.exists(layerPath):
                os.makedirs(layerPath)
            featuresNum = features[layerNum].shape[3]
            for feature in range(0, featuresNum, 1):
                if resize == 0:
                    cv.imwrite(layerPath + "/%d"%(feature+1)+".png", features[layerNum][0, :, :, feature])
                if resize == 1:
                    resizedDst = cv.resize(features[layerNum][0, :, :, feature], (self.imgRows, self.imgCols), resizedDst, 0, 0, cv.INTER_NEAREST)
                    cv.imwrite(layerPath + "/%d"%(feature+1)+".png", resizedDst)
            layerNum += 1

class MemStream(opmap.videoData.VideoData):
    def __init__(self, srcDir, threshold, **videoDataArgs):
        tempData = dataProc.loadData(srcDir, addChannel=False)
        print(tempData.shape)
        super(MemStream, self).__init__(length=tempData.shape[0], height=tempData.shape[1], width=tempData.shape[2], **videoDataArgs)
        #fileList = sorted(glob.glob(srcDir+'mem/*.npy'))
        self.camp = 'grey'
        self.data = tempData
        '''
        self.phase = opmap.phaseMap.PhaseMap(self.data, width=self.data.shape[1])
        self.phaseVariance = opmap.phaseVarianceMap.PhaseVarianceMap(self.phase)
        self.phaseVariancePeak = opmap.PhaseVariancePeakMap.PhaseVariancePeakMap(self.phaseVariance, threshold=threshold)
        '''

class Phase(object):
    def __init__(self, srcDir, spatialSigma=32, temporalSigma=5, threshold=0.8):
        src = MemStream(srcDir=srcDir, threshold=threshold)
        print('data loaded')
        self.phase = opmap.phaseMap.PhaseMap(src, width=src.data.shape[1], sigma_mean=spatialSigma, sigma_t=temporalSigma)
        print('phase')
        self.phaseVariance = opmap.phaseVarianceMap.PhaseVarianceMap(self.phase)
        print('pv')
        self.phaseVariancePeak = opmap.PhaseVariancePeakMap.PhaseVariancePeakMap(self.phaseVariance, threshold=threshold)
        print('pvp')
