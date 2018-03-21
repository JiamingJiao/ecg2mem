#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import glob
import keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend
from keras.layers.advanced_activations import LeakyReLU

class netG(object):
    def __init__(self, imgRows = 256, imgCols = 256, rawRows = 200, rawCols = 200, channel = 1, kernels = 64):
        self.imgRows = imgRows
        self.imgCols = imgCols
        self.rawRows = rawRows
        self.rawCols = rawCols
        self.channel = channel
        self.kernels = kernels

    #Unet
    def uNet(self):
        inputs = Input((self.imgRows, self.imgCols,1)) # single channel

        encoder1 = Conv2D(self.kernels, 4, activation = LeakyReLU(alpha = 0.2), strides = 2, padding = 'same', kernel_initializer = 'he_normal')(inputs)
        encoder2 = Conv2D(self.kernels*2, 4, activation = LeakyReLU(alpha = 0.2), strides = 2, padding = 'same', kernel_initializer = 'he_normal')(encoder1)
        encoder3 = Conv2D(self.kernels*4, 4, activation = LeakyReLU(alpha = 0.2), strides = 2, padding = 'same', kernel_initializer = 'he_normal')(encoder2)
        encoder4 = Conv2D(self.kernels*8, 4, activation = LeakyReLU(alpha = 0.2), strides = 2, padding = 'same', kernel_initializer = 'he_normal')(encoder3)
        encoder5 = Conv2D(self.kernels*8, 4, activation = LeakyReLU(alpha = 0.2), strides = 2, padding = 'same', kernel_initializer = 'he_normal')(encoder4)
        encoder6 = Conv2D(self.kernels*8, 4, activation = LeakyReLU(alpha = 0.2), strides = 2, padding = 'same', kernel_initializer = 'he_normal')(encoder5)
        encoder7 = Conv2D(self.kernels*8, 4, activation = LeakyReLU(alpha = 0.2), strides = 2, padding = 'same', kernel_initializer = 'he_normal')(encoder6)
        encoder8 = Conv2D(self.kernels*8, 4, activation = LeakyReLU(alpha = 0.2), strides = 2, padding = 'same', kernel_initializer = 'he_normal')(encoder7)

        decoder1 = Conv2D(self.kernels*8, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(encoder8))
        merge1 = merge([decoder1, encoder7], mode = 'concat', concat_axis = 3)
        decoder2 = Conv2D(self.kernels*8, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(merge1))
        merge2 = merge([decoder2, encoder6], mode = 'concat', concat_axis = 3)
        decoder3 = Conv2D(self.kernels*8, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(merge2))
        merge3 = merge([decoder3, encoder5], mode = 'concat', concat_axis = 3)
        decoder4 = Conv2D(self.kernels*8, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(merge3))
        merge4 = merge([decoder4, encoder4], mode = 'concat', concat_axis = 3)
        decoder5 = Conv2D(self.kernels*8, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(merge4))
        merge5 = merge([decoder5, encoder3], mode = 'concat', concat_axis = 3)
        decoder6 = Conv2D(self.kernels*4, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(merge5))
        merge6 = merge([decoder6, encoder2], mode = 'concat', concat_axis = 3)
        decoder7 = Conv2D(self.kernels*2, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(merge6))
        merge7 = merge([decoder7, encoder1], mode = 'concat', concat_axis = 3)
        decoder8 = Conv2D(self.kernels, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(merge7))
        decoder9 = Conv2D(1, 1, activation = 'tanh')(decoder8)

        model = Model(input = inputs, output = decoder9)
        model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_absolute_percentage_error', metrics = ['accuracy'])
        return model

    def train(self):
        # load data
        extraTrain = glob.glob('/mnt/recordings/SimulationResults/mapping/2/train/extra/*.jpg')
        memTrain = glob.glob('/mnt/recordings/SimulationResults/mapping/2/train/mem/*.jpg')
        extraTrainMerge = np.ndarray((len(extraTrain), self.imgRows, self.imgCols, self.channel), dtype=np.uint8)
        memTrainMerge = np.ndarray((len(memTrain), self.imgRows, self.imgCols, self.channel), dtype=np.uint8)
        rawImg = np.ndarray((self.rawRows, self.rawCols, self.channel), dtype=np.uint8)
        tempImg = np.ndarray((self.rawRows, self.rawCols, self.channel), dtype=np.uint8)
        j = 0
        for i in extraTrain:
            #rawImg = load_img(i)
            rawImg = cv2.imread(i,0)
            tempImg = cv2.resize(rawImg, (self.imgRows, self.imgCols))
            extraTrainMerge[j] = img_to_array(tempImg)
            j += 1
        extraTrainMerge = extraTrainMerge.astype('float32')
        extraTrainMerge /= 255
        j = 0
        for i in memTrain:
            #rawImg = load_img(i)
            rawImg = cv2.imread(i,0)
            tempImg = cv2.resize(rawImg, (self.imgRows, self.imgCols))
            memTrainMerge[j] = img_to_array(tempImg)
            j += 1
        memTrainMerge = memTrainMerge.astype('float32')
        memTrainMerge /= 255
        # train model
        model = self.uNet()
        checkpoints = ModelCheckpoint('/mnt/recordings/SimulationResults/mapping/2/checkpoints/20180314.hdf5', monitor = 'loss', verbose = 2,
        save_best_only = True, save_weights_only = False, mode = 'auto', period = 20)
        model.fit(extraTrainMerge, memTrainMerge, batch_size = 10, epochs = 100, verbose = 2, validation_split = 0.4, shuffle = True,
        callbacks=[checkpoints])
