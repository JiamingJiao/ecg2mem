#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
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

def histEqualizationArray(src, dst):
    return dst

def calculateHistTransArray(src1, src2, depth = 8):
    dst = np.zeros((src1.shape[0]), dtype = np.int32)
    accumulation = 0
    for i in range(0, 2**depth):
        dst[i] = Temp
    return dst
