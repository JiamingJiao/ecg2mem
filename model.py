#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
import os
import glob
import keras
from keras.models import *
from keras.layers import Input, merge, Conv2D, UpSampling2D, Dropout, BatchNormalization, Flatten, Dense, MaxPooling2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend
from keras.layers.advanced_activations import LeakyReLU
import dataProc

class networks(object):
    def __init__(self, imgRows = 256, imgCols = 256, rawRows = 200, rawCols = 200, channels = 1, gKernels = 64, dKernels = 64):
        self.imgRows = imgRows
        self.imgCols = imgCols
        self.rawRows = rawRows
        self.rawCols = rawCols
        self.channels = channels
        self.gKernels = gKernels
        self.dKernels = dKernels

    #Unet
    def uNet(self, connections = 1):
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
        decoder9 = Conv2D(1, 1, activation = 'sigmoid')(decoder8)
        model = Model(input = inputs, output = decoder9, name = 'uNet')
        return model

    def straight3(self):
        inputA = Input((self.imgRows, self.imgCols, self.channels))
        inputB = Input((self.imgRows, self.imgCols, self.channels))
        combinedImg = merge([inputA, inputB], mode = 'concat', concat_axis = -1)
        conv1 = Conv2D(self.dKernels, 3, padding = 'same', kernel_initializer = 'he_normal')(combinedImg)
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
        probability = Dense(2, activation = 'softmax')(drop5)
        model = Model(input = [inputA, inputB], output = probability, name='straight3')
        return model

    def vgg16(self):
        inputA = Input((self.imgRows, self.imgCols, self.channels))
        inputB = Input((self.imgRows, self.imgCols, self.channels))
        combinedImg = merge([inputA, inputB], mode = 'concat', concat_axis = -1)
        conv1 = Conv2D(self.dKernels, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(combinedImg)
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
        probability = Dense(2, activation = 'softmax')(drop7)
        model = Model(input = [inputA, inputB], output = probability, name='VGG16')
        return model

    def netA(self, uNetConnections = 1):
        inputA = Input((self.imgRows, self.imgCols, self.channels))
        inputB = Input((self.imgRows, self.imgCols, self.channels))
        netG = self.uNet(connections = uNetConnections)
        netD = self.straight3() #You should make change in trainGAN if you change this
        fakeB = netG(inputA)
        outputD = netD([inputA, fakeB])
        netA = Model(input = [inputA, inputB], output = [fakeB, outputD], name = 'netA')
        return netA

class GAN(object):
    def __init__(self, imgRows = 256, imgCols = 256, rawRows = 200, rawCols = 200, channels = 1):
        self.imgRows = imgRows
        self.imgCols = imgCols
        self.rawRows = rawRows
        self.rawCols = rawCols
        self.channels = channels

    def trainGAN(self, extraPath, memPath, modelPath, epochsNum = 100, batchSize = 10, valSplit = 0.2, lossRatio = 100, savingInterval = 50,
    uNetConnections = 1, lossFuncG = 'mae', lossFuncD = 'binary_crossentropy', learningRateG = 1e-4, learningRateD = 1e-6,
    lossFuncA1 = 'mae', lossFuncA2 = 'binary_crossentropy', netGOnlyEpochs = 25):
        network = networks()
        netG = network.uNet(connections = uNetConnections)
        netD = network.straight3()
        netA = network.netA(uNetConnections = uNetConnections)
        netD.trainble = False
        netA.compile(optimizer = Adam(lr = learningRateG), loss = [lossFuncA1, lossFuncA2], loss_weights = [lossRatio, 1], metrics = ['mae', 'accuracy'])
        netD.trainable = True
        netG.compile(optimizer = Adam(lr = learningRateG), loss = lossFuncG, metrics = [lossFuncG])
        netD.compile(optimizer = Adam(lr = learningRateD), loss = lossFuncD, metrics = ['accuracy'])
        extraTrain = dataProc.loadData(inputPath = extraPath, startNum = 0, resize = 1, normalization = 1)
        memTrain = dataProc.loadData(inputPath = memPath, startNum = 0, resize = 1, normalization = 1)
        extraTrain = extraTrain.reshape((extraTrain.shape[0], self.imgRows, self.imgCols, self.channels))
        memTrain = extraTrain.reshape((memTrain.shape[0], self.imgRows, self.imgCols, self.channels))
        lossRecorder = np.ndarray((round(extraTrain.shape[0]/batchSize)*epochsNum, 2), dtype = np.float32)
        lossCounter = 0
        minLossG = 10000.0
        savingStamp =(0, 0)
        weightsNetGPath = modelPath + 'netG_GOnly.h5'
        checkpointer = ModelCheckpoint(weightsNetGPath, monitor = 'val_loss', verbose = 1, save_best_only = True, save_weights_only = True, mode = 'min')
        print('begin to train G')
        netG.fit(x = extraTrain, y = memTrain, batch_size = 10, epochs = netGOnlyEpochs, verbose = 2, callbacks = [checkpointer], validation_split = 0.2)
        netG.load_weights(weightsNetGPath)
        for currentEpoch in range(netGOnlyEpochs, epochsNum):
            for currentBatch in range(0, len(extraTrain), batchSize):
                if (currentEpoch == netGOnlyEpochs) and (currentBatch == 0):
                    print('begin to train GAN')
                extraLocal = extraTrain[currentBatch:currentBatch+batchSize, :]
                memLocal = memTrain[currentBatch:currentBatch+batchSize, :]
                memFake = netG.predict_on_batch(extraLocal)
                extraForD = np.concatenate((extraLocal,extraLocal), axis = 0)
                realFake = np.concatenate((memLocal,memFake), axis = 0)
                labelD = np.zeros((batchSize*2, 2), dtype = np.float64)
                labelD[0:batchSize, 0] = 1
                labelD[batchSize:batchSize*2, 1] = 1
                lossD = netD.train_on_batch([extraForD, realFake], labelD)
                labelA = np.ones((batchSize, 2), dtype = np.float64) #to fool the netD
                labelA[:, 1] = 0
                lossA = netA.train_on_batch([extraLocal, memLocal], [memLocal, labelA])
                lossRecorder[lossCounter, 0] = lossD[0]
                lossRecorder[lossCounter, 1] = lossA[0]
                lossCounter += 1
                msg = 'epoch of ' + '%d '%(currentEpoch+1) + 'batch of ' + '%d '%(currentBatch/batchSize+1) + 'lossD1=%f '%lossD[0] + 'lossD2=%f'%lossD[1] \
                + 'lossA1=%f '%lossA[0] + 'lossA2=%f '%lossA[1] + 'lossA3=%f '%lossA[2] + 'lossA4=%f '%lossA[3] + 'lossA5=%f '%lossA[4]
                print(msg)
                if (minLossG > lossA[0]):
                    weightsNetGPath = modelPath + 'netG_latest.h5'
                    netG.save_weights(weightsNetGPath, overwrite = True)
                    minLossG = lossA[0]
                    savingStamp = (currentEpoch+1, round(currentBatch/batchSize+1))
            if (currentEpoch % savingInterval == (savingInterval-1)) and (currentEpoch != epochsNum-1):
                os.rename(modelPath+'netG_latest.h5', modelPath+'netG_%d_epochs.h5'%savingStamp[0])
        np.save(modelPath + 'loss', lossRecorder)
        print('training completed')
