#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import glob
import os
import keras
from keras.preprocessing.image import array_to_img, img_to_array

def npyToJpg(src = "20171225-10", dst = "20180228"):
    max_a = 19.766789
    min_a = -11.055192
    max_b = 17.245272
    min_b = -88.138232
    for i in range(0, 1000, 1):
        npy = numpy.load("/mnt/recordings/SimulationResults/2D/"+src+"/phie_"+"%04d"%i+".npy")
        npy = 255*(npy-min_a)/(max_a-min_a)
        cv2.imwrite("/mnt/recordings/SimulationResults/mapping/"+dst+"/data/A/val/%04d"%i+".jpg",npy)
        npy = numpy.load("/mnt/recordings/SimulationResults/2D/"+src+"/vmem_"+"%04d"%i+".npy")
        npy = 255*(npy-min_b)/(max_b-min_b)
        cv2.imwrite("/mnt/recordings/SimulationResults/mapping/"+dst+"/data/B/val/%04d"%i+".jpg",npy)

def loadData(inputPath, startNum = 0, cvtDataType = 0, resize = 0, rawRows = 200, rawCols = 200, imgRows = 256, imgCols = 256, channels = 1):
    fileName = glob.glob(inputPath + '*.jpg')
    if resize == 0:
        mergeImg = np.ndarray((len(fileName), rawRows, rawCols, channels), dtype=np.uint8)
    else:
        mergeImg = np.ndarray((len(fileName), imgRows, imgCols, channels), dtype=np.uint8)
        tempImg = np.ndarray((imgRows, imgCols, channels), dtype=np.uint8)
    rawImg = np.ndarray((rawRows, rawCols, channels), dtype=np.uint8)

    j = 0
    for i in range(0, len(fileName)):
        localName = inputPath + '%04d'%startNum + ".jpg"
        rawImg = cv2.imread(localName, 0)
        if resize == 1:
            tempImg = vc2.resize(rawImg, imgRows, imgCols)
            mergeImg[i] = img_to_array(tempImg)
        else:
            mergeImg[i] = img_to_array(rawImg)
        startNum += 1
    if cvtDataType == 1:
        mergeImg = mergeImg.astype('float64')
        mergeImg /= 255
    return mergeImg
