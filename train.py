import numpy as np
import cv2 as cv
import os
import math
import datetime
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

from model import Networks, Gan
import dataProc

class Generator(Networks):
    def __init__(self, netgName, rawSize=(200, 200), batchSize=10, **networksArgs):
        super(Generator, self).__init__(**networksArgs)
        getModel = {'uNet': self.uNet, 'convLstm': self.convLstm, 'uNet3d': self.uNet3d}
        self.netg = getModel[netgName]()
        # Cannot make full use of GPU memroy if () is in the dictionary
        # instead of in the above line on Windows. Never know why.
        self.netg.summary()
        self.rawSize = rawSize
        self.batchSize = batchSize
    
    def train(self, ecgDirList, memDirList, continueTrain=False, pretrainedModelPath=None, modelDir=None,
    lossFuncG='mae', epochsNum=100, learningRateG=1e-4, earlyStoppingPatience=10, valSplit=0.2):
        self.netg.compile(optimizer=Adam(lr=learningRateG), loss=lossFuncG, metrics=[lossFuncG])
        if continueTrain == True:
            self.netg.load_weights(pretrainedModelPath)
        if not os.path.exists(modelDir):
            os.makedirs(modelDir)
        checkpointer = ModelCheckpoint(modelDir+'netg.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
        earlyStopping = EarlyStopping(patience=earlyStoppingPatience, verbose=1)
        learningRate = ReduceLROnPlateau('val_loss', 0.1, earlyStoppingPatience, 1, 'auto', 1e-4, min_lr=learningRateG*1e-4)
        ecg, mem = dataProc.mergeSequence(ecgDirList, memDirList, self.temporalDepth, self.netg.name, self.rawSize, self.imgSize, dataProc.NORM_RANGE)
        historyG = self.netg.fit(x=ecg, y=mem, batch_size=self.batchSize, epochs=epochsNum, verbose=2, shuffle=True, validation_split=valSplit,
        callbacks=[checkpointer, learningRate, earlyStopping])
        return historyG

    def predict(self, ecgDirList, dstDirList, modelDir, priorEcgRange, priorMemRange, batchSize=10):
        self.netg.load_weights(modelDir)
        length = len(ecgDirList)
        for i in range(0, length):
            rawEcg = dataProc.loadData(srcDir=ecgDirList[i], resize=True, dstSize=self.imgSize)
            if self.netg.name == 'uNet':
                ecg = rawEcg
            else:
                ecg = dataProc.create3dEcg(rawEcg, self.temporalDepth, self.netg.name)
            ecg = dataProc.scale(ecg, priorEcgRange, dataProc.NORM_RANGE)
            print(ecg.shape)
            mem = self.netg.predict(ecg, batch_size=self.batchSize, verbose=1)
            pngMem = mem*255
            mem = dataProc.scale(mem, dataProc.NORM_RANGE, priorMemRange)
            npyDir = dstDirList[i] + 'npy/'
            if not os.path.exists(npyDir):
                os.makedirs(npyDir)
            pngDir = dstDirList[i] + 'png/'
            if not os.path.exists(pngDir):
                os.makedirs(pngDir)
            for i in range (0, mem.shape[0]):
                resizedMem = cv.resize(mem[i], self.rawSize)
                np.save(npyDir+'%06d.npy'%i, resizedMem)
                resizedPng = cv.resize(pngMem[i], self.rawSize)
                cv.imwrite(pngDir+'%06d.png'%i, resizedPng)
        return mem

# ((number of training samples*validation split ratio)/batch size) should be an integer
def trainGan(pecgDirList, memDirList, modelDir, rawSize=(200, 200), imgSize=(256, 256), channels=1, netgName='uNet', netDName='VGG16', activationG='relu', lossFuncG='mae',
temporalDepth=None, gKernels=64, gKernelSize=None, dKernels=64, dKernelSize=None, gradientPenaltyWeight=10, lossDWeight=0.01, learningRateG=1e-4, learningRateD=1e-4,
trainingRatio=5, epochsNum=100, earlyStoppingPatience=10, batchSize=10, valSplit=0.2, continueTrain=False, pretrainedGPath=None, pretrainedDPath=None, beta1=0.5, beta2=0.9):

    gan = Gan(imgSize[0], imgSize[1], channels, netDName, netgName, temporalDepth, gKernels, dKernels, gKernelSize, activationG, lossFuncG, gradientPenaltyWeight,
    lossDWeight, learningRateG, learningRateD, beta1, beta2, batchSize)
    pecg, mem = dataProc.mergeSequence(pecgDirList, memDirList, temporalDepth, netgName, rawSize, imgSize, dataProc.NORM_RANGE)
    #delete some data here to match the batch size
    print('traing data loaded')

    trainingDataLength = math.floor((1-valSplit)*pecg.shape[0]+0.1)
    lossRecorder = np.ndarray((math.floor(trainingDataLength/gan.batchSize + 0.1)*epochsNum, 2), dtype=np.float32)
    lossCounter = 0
    minLossG = np.inf
    weightsDPath = modelDir + 'netD_latest.h5'
    weightsGPath = modelDir + 'netG_latest.h5'
    if continueTrain == True:
        gan.netG.load_weights(pretrainedGPath)
        gan.netD.load_weights(pretrainedDPath)
    labelReal = np.ones((gan.batchSize), dtype=np.float32)
    labelFake = -np.ones((gan.batchSize), dtype=np.float32)
    dummyMem = np.zeros((gan.batchSize), dtype=np.float32)
    earlyStoppingCounter = 0
    print('begin to train GAN')

    for currentEpoch in range(0, epochsNum):
        beginingTime = datetime.datetime.now()
        [pEcgTrain, pEcgVal, memTrain, memVal] = dataProc.splitTrainAndVal(pecg, mem, valSplit)
        for currentBatch in range(0, trainingDataLength, gan.batchSize):
            pEcgLocal = pEcgTrain[currentBatch:currentBatch+gan.batchSize, :]
            memLocal = memTrain[currentBatch:currentBatch+gan.batchSize, :]
            randomIndexes = np.random.randint(low=0, high=trainingDataLength-gan.batchSize-1, size=trainingRatio, dtype=np.int32)
            for i in range(0, trainingRatio):
                pEcgForD = pEcgTrain[randomIndexes[i]:randomIndexes[i]+gan.batchSize]
                memForD = memTrain[randomIndexes[i]:randomIndexes[i]+gan.batchSize]
                lossD = gan.penalizedNetD.train_on_batch([pEcgForD, memForD], [labelReal, labelFake, dummyMem])
            lossA = gan.netA.train_on_batch(pEcgLocal, [memLocal, labelReal])
        #validate the model
        lossVal = gan.netG.evaluate(x=pEcgVal, y=memVal, batch_size=gan.batchSize, verbose=0)
        lossRecorder[lossCounter, 0] = lossA[0]
        lossRecorder[lossCounter, 1] = lossVal[0]
        lossCounter += 1
        if (minLossG > lossVal[0]):                
            gan.netG.save_weights(weightsGPath, overwrite=True)
            gan.netD.save_weights(weightsDPath, overwrite=True)
            minLossG = lossVal[0]
            earlyStoppingCounter = 0
        earlyStoppingCounter += 1
        displayLoss(lossD, lossA, lossVal, beginingTime, currentEpoch+1)
        if earlyStoppingCounter == earlyStoppingPatience:
            print('early stopping')
            break
    np.save(modelDir + 'loss', lossRecorder)
    print('training completed')


def prediction(pecgDirList, dstDirList, modelDir, priorEcgRange, priorMemRange, rawSize=(200, 200), imgSize=(256, 256), channels=1, netgName='uNet',
activationG='relu', lossFuncG='mae', temporalDepth=None, gKernels=64, gKernelSize=3, batchSize=10):
    network = Networks(imgSize=imgSize, channels=channels, gKernels=gKernels, gKernelSize=gKernelSize, temporalDepth=temporalDepth)
    if netgName == 'uNet':
        netG = network.uNet()
    elif netgName == 'convLstm':
        netG = network.convLstm()
    elif netgName == 'uNet3d':
        netG = network.uNet3d()
    netG.load_weights(modelDir)
    length = len(pecgDirList)
    for i in range(0, length):
        rawEcg = dataProc.loadData(srcDir=pecgDirList[i], resize=True, dstSize=imgSize)
        if netgName == 'uNet':
            ecg = rawEcg
        else:
            ecg = dataProc.create3dEcg(rawEcg, temporalDepth, netgName)
        ecg = dataProc.scale(ecg, priorEcgRange, dataProc.NORM_RANGE)
        mem = netG.predict(ecg, batch_size=batchSize, verbose=1)
        pngMem = mem*255
        mem = dataProc.scale(mem, dataProc.NORM_RANGE, priorMemRange)
        dstDir = dstDirList[i]
        if not os.path.exists(dstDir):
            os.makedirs(dstDir)
        npyDir = dstDir + 'npy/'
        if not os.path.exists(npyDir):
            os.mkdir(npyDir)
        pngDir = dstDir + 'png/'
        if not os.path.exists(pngDir):
            os.mkdir(pngDir)
        for i in range (0, mem.shape[0]):
            resizedMem = cv.resize(mem[i], rawSize)
            np.save(npyDir+'%06d.npy'%i, resizedMem)
            resizedPng = cv.resize(pngMem[i], rawSize)
            cv.imwrite(pngDir+'%06d.png'%i, resizedPng)


def displayLoss(lossD, lossA, lossVal, beginingTime, epoch):
    lossValStr = ' - lossVal: ' + '%.6f'%lossVal[0]
    lossDStr = ' - lossD: ' + lossD[0].astype(np.str) + ' '
    lossDOnRealStr = ' - lossD_real: ' + lossD[1].astype(np.str) + ' '
    lossDOnFakeStr = ' - lossD_fake: ' + lossD[2].astype(np.str) + ' '
    lossDOnPenalty = ' - penalty: ' + lossD[3].astype(np.str) + ' '
    lossAStr = ' - lossA: ' + lossA[0].astype(np.str) + ' '
    lossGStr = ' - lossG: ' + lossA[1].astype(np.str) + ' '
    lossDInA = ' - lossD: ' + lossA[2].astype(np.str) + ' '
    endTime = datetime.datetime.now()
    trainingTime = endTime - beginingTime
    msg = ' - %d'%trainingTime.total_seconds() + 's' + ' - epoch: ' + '%d '%(epoch) + lossDStr + lossDOnRealStr + lossDOnFakeStr \
    + lossDOnPenalty + lossAStr + lossGStr + lossDInA + lossValStr
    print(msg)
