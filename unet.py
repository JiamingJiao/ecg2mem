#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import keras
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateSchedule
from keras import backend
from data import *

class netG(object):
    def __init__(self, imgRows = 200, imgCols = 200):
        self.imgRows = imgRows
        self.imgCols = imgCols

    #Unet
    def uNet(self):
        inputs = Input((self.imgRows, self.imgCols,1)) # single channel

        conv1_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1_2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

        conv2_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

        conv3_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3_1)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_2)

        conv4_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_1)
        drop4 = Dropout(0.5)(conv4_2)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5_1 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5_2 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5_1)
        drop5 = Dropout(0.5)(conv5_2)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
        conv6_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6_1)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6_2))
        merge7 = merge([conv3_2,up7], mode = 'concat', concat_axis = 3)
        conv7_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7_1)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7_2))
        merge8 = merge([conv2_2,up8], mode = 'concat', concat_axis = 3)
        conv8_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8_1)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8_2))
        merge9 = merge([conv1_2,up9], mode = 'concat', concat_axis = 3)
        conv9_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9_2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9_1)
        conv9_3 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9_2)

        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9_3)

        model = Model(input = inputs, output = conv10)
        model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        return model

    def train(self):
        # load data
        data = dataProcess(self.imgRows, self.imgCols)
        extraTrain = glob.glob('/mnt/recording/SimulationResults/mapping/2/train/extra/*.jpg')
        extraTrainMerge = np.ndarray((len(extraTrain), self.imgRows, self.imgCols), dtype=np.uint8)
        memTrainMerge = np.ndarray((len(extraTrain), self.imgRows, self.imgCols), dtype=np.uint8)
        for i in imgsTrain:
            img = load_img('/mnt/recording/SimulationResults/mapping/2/train/extra/%04d'%i + '.jpg')
            extraTrainMerge[i] = img_to_array(img)
            img = load_img('/mnt/recording/SimulationResults/mapping/2/train/mem/%04d'%i + '.jpg')
            memTrainMerge[i] = img_to_array(img)
        extraTrainMerge = extraTrainMerge.astype('float32')
        extraTrainMerge /= 255
        memTrainMerge = memTrainMerge.astype('float32')
        memTrainMerge /= 255
        # train model
        model = self.uNet()
        checkpoints = ModelCheckpoint('/mnt/recordins/SimulationResults/mapping/2/checkpoints', monitor = 'loss', verbose = 1,
        save_best_only = False, save_weights_only = False, mode = 'auto', period = 20)
        model.fit(extraTrainMerge, memTrainMerge, batch_size = 10, epoch = 200, verbose = 0, validation_split = 0.2, Shuffle = True,
        callbacks=[model_checkpoint])

network = netG()
network.train()
