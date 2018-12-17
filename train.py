import numpy as np
import cv2 as cv
import os
import math
import datetime
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

import dataProc
from model import Networks, Gan
from analysis import calculateMae

class Generator(Networks):
    def __init__(self, netgName, rawSize=(200, 200), batchSize=10, **networksArgs):
        super(Generator, self).__init__(**networksArgs)
        getModel = {'uNet': self.uNet, 'convLstm': self.convLstm, 'uNet3d': self.uNet3d, 'convLstm5': self.convLstm5}
        self.netg = getModel[netgName]()
        self.netg.summary()
        self.rawSize = rawSize
        self.batchSize = batchSize
        self.dataRange = [0.0]*4 # minPecg, maxPecg, minVmem, maxVmem
    
    def trainConvLstm(self, pecgDirList, vmemDirList, continueTrain=False, pretrainedModelPath=None, modelDir=None,
    lossFuncG='mae', epochsNum=100, learningRateG=1e-4, earlyStoppingPatience=10, valSplit=0.2, length=200):
        self.netg.compile(optimizer=Adam(lr=learningRateG), loss=lossFuncG, metrics=[lossFuncG])
        if continueTrain == True:
            self.netg.load_weights(pretrainedModelPath)
        if not os.path.exists(modelDir):
            os.makedirs(modelDir)
        checkpointer = ModelCheckpoint(modelDir+'netg.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
        earlyStopping = EarlyStopping(patience=earlyStoppingPatience, verbose=1)
        learningRate = ReduceLROnPlateau('val_loss', 0.1, earlyStoppingPatience, 1, 'auto', 1e-4, min_lr=learningRateG*1e-4)

        #ecg, mem, dataRange = dataProc.mergeSequence(ecgDirList, memDirList, self.temporalDepth, self.netg.name, self.rawSize, self.imgSize, dataProc.NORM_RANGE)
        pecg = dataProc.Pecg(groups=len(pecgDirList), length=length, height=self.rawSize[0], width=self.rawSize[1], channels=self.channels)
        vmem = dataProc.Vmem(groups=len(vmemDirList), length=length, height=self.rawSize[0], width=self.rawSize[1], channels=self.channels)
        pecg.set2dData(pecgDirList)
        vmem.set2dData(vmemDirList)
        pecg.normalize()
        vmem.normalize()
        dataProc.shuffleData(pecg.twoD, vmem.twoD)
        pecg.setRnnData(self.temporalDepth)
        vmem.setRnnData(self.temporalDepth)
        pecg.splitTrainAndVal(valSplit)
        vmem.splitTrainAndVal(valSplit)
        trainGenerator = dataProc.generatorRnn(pecg.train, vmem.train, self.batchSize)
        valGenerator = dataProc.generatorRnn(pecg.val, vmem.val, self.batchSize)
        print(pecg.range)
        print(vmem.range)
        historyG = self.netg.fit_generator(trainGenerator, epochs=epochsNum, verbose=2, callbacks=[checkpointer, learningRate],
        validation_data=valGenerator, use_multiprocessing=True)
        self.dataRange[:2] = pecg.range
        self.dataRange[2:] = vmem.range
        return [self.dataRange, historyG]

    def predict(self, pecgDirList, dstDir, trueVmemDirList, modelDir, length=200, batchSize=10):
        self.netg.load_weights(modelDir)
        pecg = dataProc.Pecg(groups=len(pecgDirList), length=length, height=self.rawSize[0], width=self.rawSize[1], channels=self.channels)
        pecg.set2dData(pecgDirList)
        pecg.twoD = dataProc.scale(pecg.twoD, self.dataRange[:2])
        pecg.setRnnData(self.temporalDepth)
        vmem = dataProc.Vmem(groups=len(pecgDirList), length=length, height=self.rawSize[0], width=self.rawSize[1], channels=self.channels)
        vmem.setRnnData(self.temporalDepth)
        scaledPng = np.zeros((self.rawSize + (self.channels,)), dtype=dataProc.DATA_TYPE)
        for i in range(0, vmem.groups):
            # predict, then save .npy and .png files
            vmem.rnn[i] = self.netg.predict(pecg.rnn[i], batch_size=self.batchSize, verbose=1)
            npyDir = ''.join([dstDir, pecgDirList[i][-23:-6], '/npy/']) # create a folder for every sequence
            pngDir = ''.join([dstDir, pecgDirList[i][-23:-6], '/png/'])
            if not os.path.exists(npyDir):
                os.makedirs(npyDir)
            if not os.path.exists(pngDir):
                os.makedirs(pngDir)
            for j in range (0, vmem.rnn.shape[1]):
                #resizedPng = cv.resize(vmem.rnn[i, j], self.rawSize)
                scaledPng = 255*vmem.rnn[i,j]
                cv.imwrite(pngDir+'%06d.png'%j, scaledPng)
            vmem.rnn[i] = dataProc.scale(vmem.rnn[i], dataProc.NORM_RANGE, self.dataRange[2:])
            for j in range (0, vmem.rnn.shape[1]):
                #resizedMem = cv.resize(vmem.rnn[i, j], self.rawSize)
                np.save(npyDir+'%06d.npy'%j, vmem.rnn[i, j])
        # calculate MAE
        trueVmem = dataProc.Vmem(groups=len(pecgDirList), length=length, height=self.rawSize[0], width=self.rawSize[1], channels=self.channels)
        trueVmem.set2dData(trueVmemDirList)
        trueVmem.setRnnData(self.temporalDepth)
        mae,stdError = calculateMae(vmem.rnn, trueVmem.rnn)
        print('mae is %f, std_error is %f'%(mae, stdError))


# ((number of training samples*validation split ratio)/batch size) should be an integer
'''
def trainGan(pecgDirList, memDirList, modelDir, rawSize=(200, 200), imgSize=(256, 256), channels=1, netgName='uNet', netDName='VGG16', activationG='relu', lossFuncG='mae',
temporalDepth=None, gKernels=64, gKernelSize=None, dKernels=64, dKernelSize=None, gradientPenaltyWeight=10, lossDWeight=0.01, learningRateG=1e-4, learningRateD=1e-4,
trainingRatio=5, epochsNum=100, earlyStoppingPatience=10, batchSize=10, valSplit=0.2, continueTrain=False, pretrainedGPath=None, pretrainedDPath=None, beta1=0.5, beta2=0.9):

    gan = Gan(imgSize[0], imgSize[1], channels, netDName, netgName, temporalDepth, gKernels, dKernels, gKernelSize, activationG, lossFuncG, gradientPenaltyWeight,
    lossDWeight, learningRateG, learningRateD, beta1, beta2, batchSize)
    pecg, mem, dataRange = dataProc.mergeSequence(pecgDirList, memDirList, temporalDepth, netgName, rawSize, imgSize, dataProc.NORM_RANGE)
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
    [pEcgTrain, pEcgVal, memTrain, memVal] = dataProc.splitTrainAndVal(pecg, mem, valSplit)
    print('begin to train GAN')

    for currentEpoch in range(0, epochsNum):
        beginingTime = datetime.datetime.now()
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
'''

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
