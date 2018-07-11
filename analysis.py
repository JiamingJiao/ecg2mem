#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import math
import dataProc

'''
def histogramMatchingArray(src1, src2):
    return dst

def calculateHistogram(src):
    return dst
'''

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
