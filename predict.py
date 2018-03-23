#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import keras
from keras.preprocessing.image import array_to_img, img_to_array
#from keras.models import *
from model import *

rawRows = 200
rawCols = 200
imgRows = 256
imgCols = 256
channel = 1
extraTest = glob.glob('/mnt/recordings/SimulationResults/mapping/2/test/src/extra/*.jpg')
extraTestMerge = np.ndarray((len(extraTest),imgRows, imgCols, channel), dtype=np.uint8)
rawSizeImg = np.ndarray((rawRows, rawCols, channel), dtype=np.uint8)
resizedImg = np.ndarray((imgRows, imgCols, channel), dtype=np.uint8)
for j in range(0, len(extraTest)):
    srcFileName = '/mnt/recordings/SimulationResults/mapping/2/test/src/extra/' + '%04d'%j + '.jpg'
    rawImg = cv2.imread(srcFileName,0)
    resizedImg = cv2.resize(rawImg, (imgRows, imgCols))
    extraTestMerge[j] = img_to_array(resizedImg)
extraTestMerge = extraTestMerge.astype('float32')
extraTestMerge /= 255
print('data loaded')
network = networks()
model = network.uNet()
model.load_weights('/mnt/recordings/SimulationResults/mapping/2/checkpoints/20180322/netG_epoch_99.h5')
print('model loaded')
resultArray = model.predict(extraTestMerge, verbose = 0, batch_size = 10)
#maxPix = np.max(resultArray)
#minPix = np.min(resultArray)
#print(maxPix)
#print(minPix)
#resultArray = (resultArray - minPix)/(maxPix - minPix)
resultArray *= 255
resultArray =resultArray.astype('uint8')

for j in range(0, len(extraTest)):
    rawSizeImg = cv2.resize(resultArray[j], (rawRows, rawCols))
    dstFileName = '/mnt/recordings/SimulationResults/mapping/2/test/dst/mem/' + '%04d'%j + '.jpg'
    cv2.imwrite(dstFileName, rawSizeImg)
print('finished')
