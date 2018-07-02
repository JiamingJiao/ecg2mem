#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
import os
import glob
import keras
import math
import tensorflow as tf
import keras.backend as K
from keras.models import *
from keras.layers import Input, Concatenate, Conv2D, UpSampling2D, Dropout, BatchNormalization, Flatten, Dense, MaxPooling2D
from keras.layers import Conv3D, UpSampling3D, MaxPooling3D, Reshape, Permute, Lambda, ZeroPadding3D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend
from keras.layers.advanced_activations import LeakyReLU
import dataProc

class networks(object):
    def __init__(self, imgRows = None, imgCols = None, channels = None, gKernels = 64, dKernels = 64, temporalDepth = None, activationG = None):
        self.imgRows = imgRows
        self.imgCols = imgCols
        self.channels = channels
        self.gKernels = gKernels
        self.dKernels = dKernels
        self.temporalDepth = temporalDepth
        self.activationG = activationG

    def uNet(self, connections):
        inputs = Input((self.imgRows, self.imgCols, self.channels))
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
        if connections == 0:
            decoder1 = Conv2D(self.gKernels*8, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(encoder8))
            decoder1 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder1)
            decoder1 = Dropout(0.5)(decoder1)
            decoder2 = Conv2D(self.gKernels*8, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(decoder1))
            decoder2 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder2)
            decoder2 = Dropout(0.5)(decoder2)
            decoder3 = Conv2D(self.gKernels*8, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(decoder2))
            decoder3 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder3)
            decoder3 = Dropout(0.5)(decoder3)
            decoder4 = Conv2D(self.gKernels*8, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(decoder3))
            decoder4 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder4)
            decoder5 = Conv2D(self.gKernels*8, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(decoder4))
            decoder5 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder5)
            decoder6 = Conv2D(self.gKernels*4, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(decoder5))
            decoder6 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder6)
            decoder7 = Conv2D(self.gKernels*2, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(decoder6))
            decoder7 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder7)
            decoder8 = Conv2D(self.gKernels, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(decoder7))
        if connections == 1:
            decoder1 = Conv2D(self.gKernels*8, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(encoder8))
            decoder1 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder1)
            decoder1 = Dropout(0.5)(decoder1)
            connection1 = Concatenate(axis = -1)([decoder1, encoder7])
            decoder2 = Conv2D(self.gKernels*8, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(connection1))
            decoder2 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder2)
            decoder2 = Dropout(0.5)(decoder2)
            connection2 = Concatenate(axis = -1)([decoder2, encoder6])
            decoder3 = Conv2D(self.gKernels*8, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(connection2))
            decoder3 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder3)
            decoder3 = Dropout(0.5)(decoder3)
            connection3 = Concatenate(axis = -1)([decoder3, encoder5])
            decoder4 = Conv2D(self.gKernels*8, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(connection3))
            decoder4 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder4)
            connection4 = Concatenate(axis = -1)([decoder4, encoder4])
            decoder5 = Conv2D(self.gKernels*8, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(connection4))
            decoder5 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder5)
            connection5 = Concatenate(axis = -1)([decoder5, encoder3])
            decoder6 = Conv2D(self.gKernels*4, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(connection5))
            decoder6 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder6)
            connection6 = Concatenate(axis = -1)([decoder6, encoder2])
            decoder7 = Conv2D(self.gKernels*2, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(connection6))
            decoder7 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder7)
            connection7 = Concatenate(axis = -1)([decoder7, encoder1])
            decoder8 = Conv2D(self.gKernels, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(connection7))
        decoder9 = Conv2D(1, 1, activation = self.activationG)(decoder8)
        model = Model(input = inputs, output = decoder9, name = 'uNet')
        return model

    def uNet3D(self):
        inputs = Input((self.temporalDepth, self.imgRows, self.imgCols, self.channels))
        encoder1 = Conv3D(filters = self.gKernels, kernel_size = (self.temporalDepth, 4, 4), strides = (1, 2, 2), \
        padding = 'same', kernel_initializer = 'he_normal')(inputs)
        encoder2 = Conv3D(filters = self.gKernels*2, kernel_size = (self.temporalDepth, 4, 4), strides = (1, 2, 2), \
        padding = 'same', kernel_initializer = 'he_normal')(encoder1)
        encoder2 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(encoder2)
        encoder2 = LeakyReLU(alpha = 0.2)(encoder2)
        encoder3 = Conv3D(filters = self.gKernels*4, kernel_size = (self.temporalDepth, 4, 4), strides = (1, 2, 2), \
        padding = 'same', kernel_initializer = 'he_normal')(encoder2)
        encoder3 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(encoder3)
        encoder3 = LeakyReLU(alpha = 0.2)(encoder3)
        encoder4 = Conv3D(filters = self.gKernels*8, kernel_size = (self.temporalDepth, 4, 4), strides = (1, 2, 2), \
        padding = 'same', kernel_initializer = 'he_normal')(encoder3)
        encoder4 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(encoder4)
        encoder4 = LeakyReLU(alpha = 0.2)(encoder4)
        encoder5 = Conv3D(filters = self.gKernels*8, kernel_size = (self.temporalDepth, 4, 4), strides = (1, 2, 2), \
        padding = 'same', kernel_initializer = 'he_normal')(encoder4)
        encoder5 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(encoder5)
        encoder5 = LeakyReLU(alpha = 0.2)(encoder5)
        encoder6 = Conv3D(filters = self.gKernels*8, kernel_size = (self.temporalDepth, 4, 4), strides = (1, 2, 2), \
        padding = 'same', kernel_initializer = 'he_normal')(encoder5)
        encoder6 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(encoder6)
        encoder6 = LeakyReLU(alpha = 0.2)(encoder6)
        encoder7 = Conv3D(filters = self.gKernels*8, kernel_size = (self.temporalDepth, 4, 4), strides = (1, 2, 2), \
        padding = 'same', kernel_initializer = 'he_normal')(encoder6)
        encoder7 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(encoder7)
        encoder7 = LeakyReLU(alpha = 0.2)(encoder7)
        encoder8 = Conv3D(filters = self.gKernels*8, kernel_size = (self.temporalDepth, 4, 4), strides = (1, 2, 2), \
        padding = 'same', kernel_initializer = 'he_normal')(encoder7)
        encoder8 = LeakyReLU(alpha = 0.2)(encoder8)
        decoder1 = Conv3D(self.gKernels*8, kernel_size = (self.temporalDepth, 4, 4), activation = 'relu', \
        padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (1,2,2))(encoder8))
        decoder1 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder1)
        decoder1 = Dropout(0.5)(decoder1)
        connection1 = Concatenate(axis = -1)([decoder1, encoder7])
        decoder2 = Conv3D(self.gKernels*8, kernel_size = (self.temporalDepth, 4, 4), activation = 'relu', \
        padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (1,2,2))(connection1))
        decoder2 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder2)
        decoder2 = Dropout(0.5)(decoder2)
        connection2 = Concatenate(axis = -1)([decoder2, encoder6])
        decoder3 = Conv3D(self.gKernels*8, kernel_size = (self.temporalDepth, 4, 4), activation = 'relu', \
        padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (1,2,2))(connection2))
        decoder3 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder3)
        decoder3 = Dropout(0.5)(decoder3)
        connection3 = Concatenate(axis = -1)([decoder3, encoder5])
        decoder4 = Conv3D(self.gKernels*8, kernel_size = (self.temporalDepth, 4, 4), activation = 'relu', \
        padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (1,2,2))(connection3))
        decoder4 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder4)
        connection4 = Concatenate(axis = -1)([decoder4, encoder4])
        decoder5 = Conv3D(self.gKernels*8, kernel_size = (self.temporalDepth, 4, 4), activation = 'relu', \
        padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (1,2,2))(connection4))
        decoder5 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder5)
        connection5 = Concatenate(axis = -1)([decoder5, encoder3])
        decoder6 = Conv3D(self.gKernels*4, kernel_size = (self.temporalDepth, 4, 4), activation = 'relu', \
        padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (1,2,2))(connection5))
        decoder6 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder6)
        connection6 = Concatenate(axis = -1)([decoder6, encoder2])
        decoder7 = Conv3D(self.gKernels*2, kernel_size = (self.temporalDepth, 4, 4), activation = 'relu', \
        padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (1,2,2))(connection6))
        decoder7 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(decoder7)
        connection7 = Concatenate(axis = -1)([decoder7, encoder1])
        decoder8 = Conv3D(self.gKernels, kernel_size = (self.temporalDepth, 4, 4), activation = 'relu', \
        padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (1,2,2))(connection7))
        #decoder9 = Conv3D(1, 1, activation = 'sigmoid')(decoder8)
        decoder9 = Conv3D(1, kernel_size = (self.temporalDepth, 4, 4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(decoder8)
        decoder10 = Conv3D(1, kernel_size = (self.temporalDepth, 1, 1), activation = self.activationG, padding = 'valid', kernel_initializer = 'he_normal')(decoder9)
        squeezed10 = Lambda(squeeze, output_shape = (self.imgRows, self.imgCols, self.channels))(decoder10)
        model = Model(input = inputs, output = squeezed10, name = 'uNet3D')
        model.summary()
        return model

    def straight3(self):
        inputs = Input((self.imgRows, self.imgCols, self.channels))
        conv1 = Conv2D(self.dKernels, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs)
        pool1 = MaxPooling2D((4, 4))(conv1)
        conv1 = LeakyReLU(alpha = 0.2)(pool1)
        conv2 = Conv2D(self.dKernels*2, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(conv2)
        pool2 = MaxPooling2D((4, 4))(conv2)
        conv2 = LeakyReLU(alpha = 0.2)(pool2)
        conv3 = Conv2D(self.dKernels*4, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2)
        conv3 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(conv3)
        pool3 = MaxPooling2D((2, 2))(conv3)
        conv3 = LeakyReLU(alpha = 0.2)(pool3)
        flatten4 = Flatten()(conv3)
        dense4 = Dense(self.dKernels*16)(flatten4)
        dense4 = LeakyReLU(alpha = 0.2)(dense4)
        drop4 = Dropout(0.5)(dense4)
        dense5 = Dense(self.dKernels*16)(drop4)
        dense5 = LeakyReLU(alpha = 0.2)(dense5)
        drop5 = Dropout(0.5)(dense5)
        probability = Dense(1, activation = 'linear')(drop5)
        model = Model(input = inputs, output = probability, name='straight3')
        return model

    def vgg16(self):
        inputs = Input((self.imgRows, self.imgCols, self.channels*2))
        conv1 = Conv2D(self.dKernels, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(self.dKernels, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)
        conv2 = Conv2D(self.dKernels*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(self.dKernels*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)
        conv3 = Conv2D(self.dKernels*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(self.dKernels*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        conv3 = Conv2D(self.dKernels*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D((2,2))(conv3)
        conv4 = Conv2D(self.dKernels*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(self.dKernels*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        conv4 = Conv2D(self.dKernels*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        pool4 = MaxPooling2D((2,2))(conv4)
        conv5 = Conv2D(self.dKernels*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(self.dKernels*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        conv5 = Conv2D(self.dKernels*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        pool5 = MaxPooling2D((2,2))(conv5)
        flatten6 = Flatten()(pool5)
        dense6 = Dense(self.dKernels*64, activation = 'relu')(flatten6)
        drop6 = Dropout(0.5)(dense6)
        dense7 = Dense(self.dKernels*64, activation = 'relu')(drop6)
        drop7 = Dropout(0.5)(dense7)
        probability = Dense(1, activation = 'linear')(drop7)
        model = Model(input = inputs, output = probability, name='VGG16')
        return model

class GAN(object):
    def __init__(self, imgRows = 256, imgCols = 256, rawRows = 200, rawCols = 200, channels = 1, netDName = None, netGName = None, temporalDepth = None, uNetconnections = 1,
    activationG = 'sigmoid', lossFuncD = 'categorical_crossentropy', lossFuncG = 'mae', optimizerG = 'Adam', lossRatio = 100, learningRateG = 1e-4, learningRateD = 1e-4):
        self.imgRows = imgRows
        self.imgCols = imgCols
        self.rawRows = rawRows
        self.rawCols = rawCols
        self.channels = channels
        self.temporalDepth = temporalDepth
        self.netGName = netGName
        self.netDName = netDName
        self.lossFuncG = lossFuncG
        self.learningRateG = learningRateG
        self.learningRateD = learningRateD
        self.activationG = activationG
        self.network = networks(imgRows = self.imgRows, imgCols = self.imgCols, channels = self.channels, temporalDepth = self.temporalDepth, activationG = self.activationG)
        if self.netDName == 'straight3':
            self.netD = self.network.straight3()
            self.netDA = self.network.straight3()
        elif self.netDName == 'VGG16':
            self.netD = self.network.vgg16()
            self.netDA = self.network.vgg16()
        self.netDA.trainable = False
        if self.netGName == 'uNet':
            self.netG = self.network.uNet(connections = uNetconnections)
            inputsA = Input((self.imgRows, self.imgCols, self.channels))
            outputG = self.netG(inputsA)
            inputD = Concatenate(axis = -1)([inputsA, outputG])
        elif self.netGName == 'uNet3D':
            self.netG = self.network.uNet3D()
            inputsA = Input((self.temporalDepth, self.imgRows, self.imgCols, self.channels))
            outputG = self.netG(inputsA)
            middleLayerOfInputs = Lambda(slice, output_shape = (1, self.imgRows, self.imgCols, self.channels))(inputsA)
            middleLayerOfInputs = Lambda(squeeze, output_shape = (self.imgRows, self.imgCols, self.channels))(middleLayerOfInputs)
            inputD = Concatenate(axis = -1)([middleLayerOfInputs, outputG])
        outputD = self.netDA(inputD)
        self.netA = Model(input = inputsA, output =[outputG, outputD], name = 'netA')
        self.netA.summary()
        if optimizerG == 'Adam':
            self.netA.compile(optimizer = Adam(lr = self.learningRateG), loss = {self.netGName:lossFuncG, self.netDName:wassersteinDistance},
            loss_weights = [lossRatio, 1], metrics = {self.netGName:lossFuncG, self.netDName:lossFuncD})
        self.netD.compile(optimizer = Adam(lr = self.learningRateD), loss = wassersteinDistance, metrics = [lossFuncD])
        print(self.netA.metrics_names)
        self.netD.summary()
        self.netDA.summary()

    def trainGAN(self, extraPath, memPath, modelPath, epochsNum = 100, batchSize = 10, valSplit = 0.2, savingInterval = 50, netGOnlyEpochs = 25, continueTrain = False,
    preTrainedGPath = None):
        if self.activationG == 'tanh':
            dataRange = [-1., 1.]
        else:
            dataRange = [0., 1.]
        if self.netGName == 'uNet':
            extraSequence = dataProc.loadData(srcPath = extraPath, startNum = 0, resize = 1, normalization = 1, normalizationRange = dataRange)
            extraSequence = extraSequence.reshape((extraSequence.shape[0], self.imgRows, self.imgCols, self.channels))
            extraTrain = extraSequence
        elif self.netGName == 'uNet3D':
            extraSequence = dataProc.loadData(srcPath = extraPath, startNum = 0, resize = 1, normalization = 1, normalizationRange = dataRange)
            extraTrain = dataProc.create3DData(extraSequence, temporalDepth = self.temporalDepth)
            extraTrain = extraTrain.reshape((extraTrain.shape[0], self.temporalDepth, self.imgRows, self.imgCols, self.channels))
            extraSequence = extraSequence.reshape((extraSequence.shape[0], self.imgRows, self.imgCols, self.channels))
        memTrain = dataProc.loadData(srcPath = memPath, startNum = 0, resize = 1, normalization = 1, normalizationRange = dataRange)
        memTrain = memTrain.reshape((memTrain.shape[0], self.imgRows, self.imgCols, self.channels))
        print(extraTrain.shape)
        lossRecorder = np.ndarray((round(extraTrain.shape[0]/batchSize)*epochsNum, 2), dtype = np.float32)
        lossCounter = 0
        minLossG = 10000.0
        savingStamp =(0, 0)
        weightsNetGPath = modelPath + 'netG_GOnly.h5'
        weightsNetAPath = modelPath + 'netA_latest.h5'
        self.netG.compile(optimizer = Adam(lr = self.learningRateG), loss = self.lossFuncG, metrics = [self.lossFuncG])
        if continueTrain == False:
            checkpointer = ModelCheckpoint(weightsNetGPath, monitor = 'val_loss', verbose = 1, save_best_only = True, save_weights_only = True, mode = 'min')
            print('begin to train G')
            trainingHistoryG = self.netG.fit(x = extraTrain, y = memTrain, batch_size = batchSize*2, epochs = netGOnlyEpochs, verbose = 2,callbacks = [checkpointer],
            validation_split = valSplit)
        elif continueTrain == True:
            preTrainedGPath = preTrainedGPath + 'netG_GOnly.h5'
            self.netG.load_weights(preTrainedGPath)
        labelD = np.ones((batchSize*2), dtype = np.float64)
        labelD[0:batchSize] = -1
        labelA = np.ones((batchSize), dtype = np.float64)
        for currentEpoch in range(netGOnlyEpochs, epochsNum):
            for currentBatch in range(0, len(extraTrain), batchSize):
                if (currentEpoch == netGOnlyEpochs) and (currentBatch == 0):
                    print('begin to train GAN')
                extraLocal = extraTrain[currentBatch:currentBatch+batchSize, :]
                memLocal = memTrain[currentBatch:currentBatch+batchSize, :]
                memFake = self.netG.predict_on_batch(extraLocal)
                realAndFake = np.concatenate((memLocal,memFake), axis = 0)
                extraForD = np.concatenate((extraSequence[currentBatch:currentBatch+batchSize, :], extraSequence[currentBatch:currentBatch+batchSize, :]), axis = 0)
                extraAndMem = np.concatenate((extraForD, realAndFake), axis = -1)
                lossD = self.netD.train_on_batch(extraAndMem, labelD)
                self.netDA.set_weights(self.netD.get_weights())
                lossA = self.netA.train_on_batch(extraLocal, [memLocal, labelA])
                lossRecorder[lossCounter, 0] = lossD[0]
                lossRecorder[lossCounter, 1] = lossA[0]
                lossCounter += 1
                lossDStr = 'lossD is ' + lossD[0].astype(np.str) + ' '
                AccDStr = 'AccD is ' + lossD[1].astype(np.str) + ' '
                lossAStr = 'lossA is ' + lossA[0].astype(np.str) + ' '
                lossGStr = 'lossG is ' + lossA[1].astype(np.str) + ' '
                lossDAStr = 'lossDA is ' + lossA[2].astype(np.str) + ' '
                AccGStr = 'AccG is ' + lossA[3].astype(np.str) + ' '
                AccDAStr = 'AccDA is ' + lossA[4].astype(np.str)
                msg = 'epoch of ' + '%d '%(currentEpoch+1) + 'batch of ' + '%d '%(currentBatch/batchSize+1) + lossDStr + AccDStr + lossAStr + lossGStr \
                + lossDAStr + AccGStr + AccDAStr
                print(msg)
                if (minLossG > lossA[2]):
                    weightsNetGPath = modelPath + 'netG_latest.h5'
                    self.netG.save_weights(weightsNetGPath, overwrite = True)
                    #netA.save_weights(weightsNetAPath, overwrite = True)
                    minLossG = lossA[2]
                    savingStamp = (currentEpoch+1, round(currentBatch/batchSize+1))
            if (currentEpoch % savingInterval == (savingInterval-1)) and (currentEpoch != epochsNum-1):
                os.rename(modelPath+'netG_latest.h5', modelPath+'netG_%d_epochs.h5'%savingStamp[0])
        np.save(modelPath + 'loss', lossRecorder)
        print('training completed')

    def wassersteinDistance(self, src1, src2):
        dst = K.mean(src1*src2)
        return dst

    def gradientPenaltyLoss(self, ):
        return dst

def squeeze(src):
    dst = tf.squeeze(src, [1])
    return dst

def slice(src):
    srcShape = src.shape.as_list()
    middleLayer = math.floor(srcShape[1]/2.0)
    dst = tf.slice(src, [0, middleLayer, 0, 0, 0], [-1, 1, -1, -1, -1])
    print(srcShape)
    print(dst.shape)
    return dst

def randomlyWeightedAverage(src1, src2):
    length = src1.shape[0]
    weights = K.random_uniform((length, 1, 1, 1), minval = 0., maxval = 1., dtype = 'tf.float64')
    dst = (weights*src1) + ((1-weights)*src2)
    return dst
