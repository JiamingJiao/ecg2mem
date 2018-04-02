#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import glob
import keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import *
from keras.layers import Input, merge, Conv2D, UpSampling2D, Dropout, BatchNormalization, Flatten, Dense
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend
from keras.layers.advanced_activations import LeakyReLU

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
        decoder9 = Conv2D(1, 1, activation = 'sigmoid')(decoder8)

        model = Model(input = inputs, output = decoder9, name = 'uNet')
        #model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_absolute_percentage_error', metrics = ['accuracy'])
        return model

    def netD(self):
        inputs = Input((self.imgRows, self.imgCols,1))
        conv1 = Conv2D(self.dKernels, 1, padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = LeakyReLU(alpha = 0.2)(conv1)
        conv2 = Conv2D(self.dKernels*2, 1, padding = 'same', kernel_initializer = 'he_normal')(conv1)
        conv2 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.0001, center = False, scale = False)(conv2)
        conv2 = LeakyReLU(alpha = 0.2)(conv2)
        conv3 = Conv2D(1, 1, padding = 'same', kernel_initializer = 'he_normal')(conv2)
        conv4 = Conv2D(1, 1, activation = 'sigmoid')(conv3)
        flat = Flatten()(conv4)
        probability = Dense(1, activation = 'softmax')(flat)
        model = Model(input = inputs, output = probability, name='netD')
        #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        return model

    def netA(self):
        inputs = Input((self.imgRows, self.imgCols,1))
        fake = self.uNet()(inputs)
        #realFake = merge([inputs, fake], mode = 'concat', concat_axis = 1)
        outputD = self.netD()(fake)
        netA = Model(input = [inputs], output = [fake, outputD], name = 'netA')
        return netA

class GAN(object):
    def __init__(self, imgRows = 256, imgCols = 256, rawRows = 200, rawCols = 200, channels = 1):
        self.imgRows = imgRows
        self.imgCols = imgCols
        self.rawRows = rawRows
        self.rawCols = rawCols
        self.channels = channels

    def trainGAN(self, extraPath, memPath, extraForFakePath, memRealPath, modelPath, epochsNum = 100, batchSize = 10, valSplit = 0.2, checkPeriod = 10,
    lossFuncG = 'mae', lossFuncD = 'binary_crossentropy', lossFuncA1 = 'mae', lossFuncA2 = 'binary_crossentropy', lossRatio = 100):
        network = networks()
        netG = network.uNet()
        netD = network.netD()
        netA = network.netA()
        netG.compile(optimizer = Adam(lr = 1e-4), loss = lossFuncG, metrics = ['accuracy'])
        netD.trainable = True
        netD.compile(optimizer = Adam(lr = 1e-4), loss = lossFuncD, metrics = ['accuracy'])
        netA.compile(optimizer = Adam(lr = 1e-4), loss = [lossFuncA1, lossFuncA2], loss_weights = [lossRatio, 1], metrics = ['accuracy'])

        extraTrain = glob.glob(extraPath)
        memTrain = glob.glob(memPath)
        extraForFake = glob.glob(extraForFakePath)
        memReal = glob.glob(memRealPath)
        extraTrainMerge = np.ndarray((len(extraTrain), self.imgRows, self.imgCols, self.channels), dtype=np.uint8)
        memTrainMerge = np.ndarray((len(memTrain), self.imgRows, self.imgCols, self.channels), dtype=np.uint8)
        extraForFakeMerge = np.ndarray((len(extraForFake), self.imgRows, self.imgCols, self.channels), dtype=np.uint8)
        memRealMerge = np.ndarray((len(memReal), self.imgRows, self.imgCols, self.channels), dtype=np.uint8)
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
        j = 0
        for i in extraForFake:
            #rawImg = load_img(i)
            rawImg = cv2.imread(i,0)
            tempImg = cv2.resize(rawImg, (self.imgRows, self.imgCols))
            extraForFakeMerge[j] = img_to_array(tempImg)
            j += 1
        extraForFakeMerge = extraForFakeMerge.astype('float32')
        extraForFakeMerge /= 255
        j = 0
        for i in memReal:
            #rawImg = load_img(i)
            rawImg = cv2.imread(i,0)
            tempImg = cv2.resize(rawImg, (self.imgRows, self.imgCols))
            memRealMerge[j] = img_to_array(tempImg)
            j += 1
        memRealMerge = memRealMerge.astype('float32')
        memRealMerge /= 255


        for m in range(epochsNum):
            for n in range(0, len(memRealMerge), batchSize):
                extraLocal = extraTrainMerge[n:n+batchSize, :]
                memLocal = memTrainMerge[n:n+batchSize, :]
                extraForFakeLocal = extraForFakeMerge[n:n+batchSize, :]
                memRealLocal = memRealMerge[n:n+batchSize, :]
                lossG = netG.train_on_batch(extraLocal, memLocal)
                memFake = netG.predict_on_batch(extraForFakeLocal)
                realFake = np.concatenate((memRealLocal,memFake), axis = 0)
                labelD = np.zeros((batchSize*2, 1), dtype = np.uint8)
                labelD[0:batchSize, :] = 1
                lossD = netD.train_on_batch(realFake, labelD)
                netD.trainable = False                
                labelA = np.ones((batchSize,1), dtype = np.uint8) #to fool the netD
                #labelA[:, 0] = 0
                lossA = netA.train_on_batch(extraForFakeLocal, [memRealLocal, labelA])
                netD.trainable = True
                msg = 'epoch of ' + '%d'%m + ' batch of ' + '%d'%(n/batchSize)
                print(msg)
            weightsNetGPath = modelPath + 'netG_epoch_%d'%m + '.h5'
            netG.save_weights(weightsNetGPath, overwrite = True)
            weightsNetDPath = modelPath + 'netD_epoch_%d'%m + '.h5'
            netD.save_weights(weightsNetDPath, overwrite = True)
            weightsNetAPath = modelPath + 'netA_epoch_%d'%m + '.h5'
            netA.save_weights(weightsNetAPath, overwrite = True)
        print('training finished')

    def trainUNet(self, extraPath, memPath, modelPath):
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
        checkpoints = ModelCheckpoint(modelPath, monitor = 'loss', verbose = 2, save_best_only = True, save_weights_only = False, mode = 'auto', period = 20)
        model.fit(extraTrainMerge, memTrainMerge, batch_size = 10, epochs = 100, verbose = 2, validation_split = 0.4, shuffle = True,
        callbacks=[checkpoints])
