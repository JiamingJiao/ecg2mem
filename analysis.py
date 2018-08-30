#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import math
import os
import dataProc
import model
import keras.backend as K

def calculateMAE(src1, src2):
    difference = np.subtract(src1, src2)
    absDifference = np.abs(difference)
    mae = np.mean(absDifference)
    return mae

def histEqualizationArray(src, depth = 8):
    maxIntensity = 2**depth - 1
    temp = 0.
    dst = np.zeros((maxIntensity+1), dtype = np.float64)
    pixNum = src.shape[0]*src.shape[1]
    histogram = cv.calcHist([src], [0], None, [maxIntensity+1], [0, maxIntensity+1])
    hitogram = histogram.astype(np.float64)
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

class intermediateLayers(model.networks):
    def __init__(self, src, netName = 'uNet', weightsPath = None, **kwargs):
        super(intermediateLayers, self).__init__(**kwargs)
        self.netName = netName
        if self.netName == 'uNet':
            resizedSrc = np.ndarray((self.imgRows, self.imgCols), dtype = np.float32)
            resizedSrc = cv.resize(src, (self.imgRows, self.imgCols), resizedSrc, 0, 0, cv.INTER_NEAREST)
            self.input = resizedSrc[np.newaxis, :, :, np.newaxis]
            self.network = super(intermediateLayers, self).uNet()
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
        max = -np.inf
        min = np.inf
        minMax = open(dstPath + 'min_max.txt', 'w+')
        for layerName in layersList:
            features[layerNum] = self.intermediateFeatures(layerName)
            layerMax = np.amax(features[layerNum])
            layerMin = np.amin(features[layerNum])
            minMax.write('%s: %f, %f\n'%(layerName, layerMin, layerMax))
            if normalizationMode == 'layer':
                features[layerNum] = 255*(features[layerNum] - layerMin)/(layerMax - layerMin)
            if normalizationMode == 'all':
                if layerMax > max:
                    max = layerMax
                if layerMin < min:
                    min = layerMin
            layerNum += 1
        minMax.close()
        if normalizationMode == 'all':
            features = 255*(features - min)/(max - min)
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
