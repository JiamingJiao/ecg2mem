#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import math
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import keras.backend as K

import opmap.videoData
import opmap.phaseMap
import opmap.phaseVarianceMap
import opmap.PhaseVariancePeakMap

import dataProc
import model

BLUE = (0, 0.439, 0.756)
RED = (0.996, 0.380, 0.380)
GREEN = (0.102, 0.655, 0.631)

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
    dst = dst.astype(np.int16)
    return dst

def calculateHistTransArray(srcHist, dstHist, depth = 8):
    maxIntensity = 2**depth - 1
    dst = np.zeros((maxIntensity+1), dtype = np.int16)
    accumulation = 0
    for i in range(0, maxIntensity+1):
        diff = maxIntensity+1
        for j in range(0, maxIntensity + 1):
            if abs(dstHist[j] - srcHist[i])<diff:
                diff = abs(dstHist[j] - srcHist[i])
                dst[i] = j
    return dst

def histSpecification(src, transArray):
    dst = np.zeros(src.shape, dtype = np.int16)
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
        tempData = dataProc.loadData(srcDir, withChannel=False)
        super(MemStream, self).__init__(length=tempData.shape[0], height=tempData.shape[1], width=tempData.shape[2], **videoDataArgs)
        #fileList = sorted(glob.glob(srcDir+'mem/*.npy'))
        self.camp = 'grey'
        self.data = tempData

class Phase(object):
    def __init__(self, srcDir, threshold=0.8):
        self.mem = MemStream(srcDir=srcDir, threshold=threshold)
        self.phase = opmap.phaseMap.PhaseMap(self.mem, width=self.mem.data.shape[1])
        self.phaseVariance = opmap.phaseVarianceMap.PhaseVarianceMap(self.phase)
        self.phaseVariancePeak = opmap.PhaseVariancePeakMap.PhaseVariancePeakMap(self.phaseVariance, threshold=threshold)

def drawRemovedElectrodes(parent, child, dstPath, **drawElectrodesArgs):
    parentImg = drawElectrodes(parent, color=BLUE, dstPath=-1, **drawElectrodesArgs)
    childImg = drawElectrodes(child, dstPath=-1, **drawElectrodesArgs)
    removedAlpha = cv.subtract(parentImg[:, :, 3], childImg[:, :, 3])
    removedBgr = np.zeros((parentImg.shape[:-1]+(3, )), dtype = np.float64)
    parentImg = parentImg.astype(np.float64)
    for i in range(0, 3):
        removedBgr[:, :, i] = np.multiply(np.divide(removedAlpha, 255), parentImg[:, :, i])
    removedBgr = removedBgr.astype(np.uint8)
    dst = childImg[:, :, 0:3] + removedBgr
    dst = cv.merge((dst[:, :, 0], dst[:, :, 1], dst[:, :, 2], parentImg[:, :, 0].astype(np.uint8)))
    if not dstPath == -1:
        cv.imwrite(dstPath, dst)
    return dst

def drawElectrodes(coordinates, radius=2, thickness=-1, color=RED, mapSize=(191, 191), dstPath=-1):
    alpha = np.zeros((mapSize[0]+(radius+thickness+1)*2, mapSize[1]+(radius+thickness+1)*2, 1), dtype=np.uint8)
    coordinates = np.around(coordinates)
    coordinates = coordinates.astype(np.uint16)
    # Convert coordinate system of array to coordinate system of image (OpenCV style)
    coordinates = np.flip(coordinates, 1)
    markerCenter = coordinates + radius + thickness + 1
    for i in markerCenter:
        center = tuple(i)
        cv.circle(alpha, center, radius, (255,255,255), thickness, cv.LINE_AA)
    cv.threshold(alpha, 127, 255, cv.THRESH_BINARY, alpha)
    b = alpha*color[2]
    b = b.astype(np.uint8)
    g = alpha*color[1]
    g = g.astype(np.uint8)
    r = alpha*color[0]
    r = r.astype(np.uint8)
    dst = cv.merge((b, g, r, alpha))
    if not dstPath == -1:
        cv.imwrite(dstPath, dst)
    return dst

def markPhaseSingularity(srcDir1, srcDir2, dstDir, priorMemRange, truncate=10, **drawMarkersArgs):
    
    def centersList(srcDir):
        phase = Phase(srcDir)
        phaseVariancePeak = phase.phaseVariancePeak.data[truncate:-truncate]
        length = phaseVariancePeak.shape[0]
        centersList = np.empty((length), dtype=object)
        for i in range(0, length):
            centersList[i] = centers(phaseVariancePeak[i])
        return centersList
    
    centersList1 = centersList(srcDir1)
    centersList2 = centersList(srcDir2)
    length = centersList1.shape[0]
    distance = np.empty((length), dtype=object)
    background = dataProc.loadData(srcDir1, withChannel=False)[truncate:-truncate] # Markers are drawn on estimation
    background = dataProc.scale(background, priorMemRange, (0, 255))
    canvas = np.ndarray(background.shape+(3,), dtype=np.uint8)
    statistics = np.zeros((5), dtype=np.float32) # mean, standard error, matching points, false postive, false negative
    # Temporally, save video in the future
    if not os.path.exists(dstDir):
        os.makedirs(dstDir)
    for i in range(0, length):
        distance[i], matching, fp, fn = matchPoints(centersList1[i], centersList2[i])
        canvas[i] = drawMarkers(background[i], (matching[0], fp, fn), **drawMarkersArgs)
        statistics[2] += matching.shape[0]
        statistics[3] += fp.shape[0]
        statistics[4] += fn.shape[0]
    if not dstDir == 'None':
         for i in range(0, length):
             # Temporally, save video in the future
            cv.imwrite(dstDir+'%06d.png'%i, canvas[i])
    else:
        print('Images were not saved!')
    distance = np.concatenate(distance)
    statistics[0] = np.mean(distance)
    statistics[1] = np.std(distance, ddof=1)
    return canvas, statistics

def drawMarkers(src, coordinatesList, colors=(BLUE, GREEN, RED), markerType=cv.MARKER_SQUARE, markerSize=20, thickness=2):
    # coordinatesList is a list of n*2 arrays
    # colors is a list of tupples
    # IMPORTANT: src data type: uint8, range: [0, 255]
    src = src.astype(np.uint8)
    canvas = np.ndarray(src.shape+(3,), dtype=np.uint8)
    cv.cvtColor(src, cv.COLOR_GRAY2BGR, canvas, 3)
    for i in range(0, len(coordinatesList)):
        coordinates = coordinatesList[i]
        coordinates = np.around(coordinates)
        coordinates = coordinates.astype(np.uint16)
        bgrColor = (colors[i][2]*255, colors[i][1]*255, colors[i][0]*255)
        for j in range(0, coordinates.shape[0]):
            cv.drawMarker(canvas, tuple(coordinates[j]), bgrColor, markerType, markerSize, thickness, cv.LINE_4)
    return canvas

def centers(src):
    src = src.astype(np.uint8)
    _, contours, _ = cv.findContours(src, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contoursNum = len(contours)
    dst = np.ndarray((contoursNum, 1, 2), dtype=np.float32)
    for i in range(0, contoursNum):
        dst[i, 0, :] = np.mean(contours[i], 0)
    dst = dst.reshape((contoursNum, 2))
    return dst

def matchPoints(src1, src2): #src2 is groundtruth
    num1 = src1.shape[0]
    num2 = src2.shape[0]
    allDistance = np.ndarray((num1, num2), dtype=np.float32)
    for i in range(0, num1):
        cv.magnitude(src2[: ,0]-src1[i, 0], src2[:, 1]-src1[i, 1], allDistance[i, :])
    matchingNum = min(num1, num2)
    matching = np.ndarray((2, matchingNum, 2), dtype=np.float32)
    distance = np.ndarray((matchingNum), dtype=np.float32)
    location = np.ndarray((matchingNum, 2), dtype = np.uint8)
    for i in range(0, matchingNum):
        location[i] = np.unravel_index(allDistance.argmin(), (num1, num2))
        distance[i] = allDistance[tuple(location[i])]
        allDistance[location[i, 0], :] = np.inf
        allDistance[:, location[i, 1]] = np.inf
    matching[0, :, :] = src1[location[:, 0]] # advanced indexing
    matching[1, :, :] = src2[location[:, 1]]
    if num1 == num2:
        fp = np.ndarray((0))
        fn = np.ndarray((0))
    elif num1 > num2:
        fp = np.delete(src1, location[:, 0], 0)
        fn = np.ndarray((0))
    else:
        fp = np.ndarray((0))
        fn = np.delete(src2, location[:, 1], 0)
    return distance, matching, fp, fn
