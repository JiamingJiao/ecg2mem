#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import glob
import keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import *
from keras.layers import Input, merge, Conv2D, UpSampling2D, Dropout, BatchNormalization
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend
from keras.layers.advanced_activations import LeakyReLU

class netG(object):
    def __init__(self, imgRows = 256, imgCols = 256, rawRows = 200, rawCols = 200, channels = 1, gKernels = 64, dKernels = 64):
        self.imgRows = imgRows
        self.imgCols = imgCols
        self.rawRows = rawRows
        self.rawCols = rawCols
        self.channels = channels
        self.gKernels = gKernels
        self.dKernels = dKernels

    #Unet
    def uNet(self):
        inputs = Input((self.imgRows, self.imgCols,1)) # single channel

        encoder1 = Conv2D(self.gKernels, 4, strides = 2, padding = 'same', kernel_initializer = 'he_normal')(inputs)
        encoder2 = Conv2D(self.gKernels*2, 4, strides = 2, padding = 'same', kernel_initializer = 'he_normal')(encoder1)
        encoder2 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(encoder2)
        encoder2 = LeakyReLU(alpha = 0.2)(encoder2)
        encoder3 = Conv2D(self.gKernels*4, 4, strides = 2, padding = 'same', kernel_initializer = 'he_normal')(encoder2)
        encoder3 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(encoder3)
        encoder3 = LeakyReLU(alpha = 0.2)(encoder3)
        encoder4 = Conv2D(self.gKernels*8, 4, strides = 2, padding = 'same', kernel_initializer = 'he_normal')(encoder3)
        encoder4 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(encoder4)
        encoder4 = LeakyReLU(alpha = 0.2)(encoder4)
        encoder5 = Conv2D(self.gKernels*8, 4, strides = 2, padding = 'same', kernel_initializer = 'he_normal')(encoder4)
        encoder5 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(encoder5)
        encoder5 = LeakyReLU(alpha = 0.2)(encoder5)
        encoder6 = Conv2D(self.gKernels*8, 4, strides = 2, padding = 'same', kernel_initializer = 'he_normal')(encoder5)
        encoder6 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(encoder6)
        encoder6 = LeakyReLU(alpha = 0.2)(encoder6)
        encoder7 = Conv2D(self.gKernels*8, 4, strides = 2, padding = 'same', kernel_initializer = 'he_normal')(encoder6)
        encoder7 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(encoder7)
        encoder7 = LeakyReLU(alpha = 0.2)(encoder7)
        encoder8 = Conv2D(self.gKernels*8, 4, strides = 2, padding = 'same', kernel_initializer = 'he_normal')(encoder7)
        encoder8 = LeakyReLU(alpha = 0.2)(encoder8)

        decoder1 = Conv2D(self.gKernels*8, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(encoder8))
        decoder1 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder1)
        decoder1 = Dropout(0.5)(decoder1)
        merge1 = merge([decoder1, encoder7], mode = 'concat', concat_axis = -1)
        decoder2 = Conv2D(self.gKernels*8, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(merge1))
        decoder2 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder2)
        decoder2 = Dropout(0.5)(decoder2)
        merge2 = merge([decoder2, encoder6], mode = 'concat', concat_axis = -1)
        decoder3 = Conv2D(self.gKernels*8, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(merge2))
        decoder3 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder3)
        decoder3 = Dropout(0.5)(decoder3)
        merge3 = merge([decoder3, encoder5], mode = 'concat', concat_axis = -1)
        decoder4 = Conv2D(self.gKernels*8, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(merge3))
        decoder4 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder4)
        merge4 = merge([decoder4, encoder4], mode = 'concat', concat_axis = -1)
        decoder5 = Conv2D(self.gKernels*8, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(merge4))
        decoder5 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder5)
        merge5 = merge([decoder5, encoder3], mode = 'concat', concat_axis = -1)
        decoder6 = Conv2D(self.gKernels*4, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(merge5))
        decoder6 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder6)
        merge6 = merge([decoder6, encoder2], mode = 'concat', concat_axis = -1)
        decoder7 = Conv2D(self.gKernels*2, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(merge6))
        decoder7 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder7)
        merge7 = merge([decoder7, encoder1], mode = 'concat', concat_axis = -1)
        decoder8 = Conv2D(self.gKernels, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(merge7))
        decoder9 = Conv2D(1, 1, activation = 'tanh')(decoder8)

        model = Model(input = inputs, output = decoder9)
        model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_absolute_percentage_error', metrics = ['accuracy'])
        return model

    def netD(self):
        inputs = Input((self.imgRows, self.imgCols,1))
        conv1 = Conv2D(dKernels, 1, padding = 'same', kernel_initializer = 'he_normal')
        conv1 = LeakyReLU(alpha = 0.2)(conv1)
        conv2 = Conv2D(dKernels*2, 1, padding = 'same', kernel_initializer = 'he_normal')
        conv2 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(conv2)
        conv2 = LeakyReLU(alpha = 0.2)(conv2)
        conv3 = Conv2D(1, 1, padding = 'same', kernel_initializer = 'he_normal')(conv2)
        conv4 = Conv2D(1, 1, activation = 'sigmoid')(conv3)

        model = Model(input = inputs, output = conv4)
        model.compile(optimizer = Adam(lr = 1e-4), )

    def train(self, extraPath, memPath, modelPath):
        # load data
        #extraTrain = glob.glob('/mnt/recordings/SimulationResults/mapping/2/train/extra/*.jpg')
        #memTrain = glob.glob('/mnt/recordings/SimulationResults/mapping/2/train/mem/*.jpg')
        extraTrain = glob.glob(extraPath)
        memTrain = glob.glob(memPath)
        extraTrainMerge = np.ndarray((len(extraTrain), self.imgRows, self.imgCols, self.channels), dtype=np.uint8)
        memTrainMerge = np.ndarray((len(memTrain), self.imgRows, self.imgCols, self.channels), dtype=np.uint8)
        rawImg = np.ndarray((self.rawRows, self.rawCols, self.channels), dtype=np.uint8)
        tempImg = np.ndarray((self.rawRows, self.rawCols, self.channels), dtype=np.uint8)
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
        #checkpoints = ModelCheckpoint('/mnt/recordings/SimulationResults/mapping/2/checkpoints/20180314.hdf5', monitor = 'loss', verbose = 2,
        #save_best_only = True, save_weights_only = True, mode = 'auto', period = 20)
        checkpoints = ModelCheckpoint(modelPath, monitor = 'loss', verbose = 2, save_best_only = True, save_weights_only = True, mode = 'auto', period = 20)
        model.fit(extraTrainMerge, memTrainMerge, batch_size = 10, epochs = 100, verbose = 2, validation_split = 0.4, shuffle = True,
        callbacks=[checkpoints])
