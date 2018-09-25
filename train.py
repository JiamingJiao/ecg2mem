import numpy as np
import cv2 as cv
import os
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler

from model import Networks
import dataProc

def trainG(dataDirList, modelDir, electrodesNum, rawSize=(200, 200), imgSize=(256, 256), channels=1, netGName='uNet', activationG='relu', lossFuncG='mae', temporalDepth=None,
gKernels=64, gKernelSize=4, epochsNum=100, batchSize=10, learningRateG=1e-4, earlyStoppingPatience=10, valSplit=0.2, continueTrain=False, pretrainedModelPath=None):
    network = Networks(imgRows=imgSize[0], imgCols=imgSize[1], channels=channels, gKernels=gKernels, gKernelSize=gKernelSize, temporalDepth=temporalDepth)
    if netGName == 'uNet':
        netG = network.uNet()
    elif netGName == 'convLstm':
        netG = network.convLstm()
    elif netGName == 'uNet3d':
        netG = network.uNet3d()
    ecg, mem = dataProc.mergeSequence(dataDirList, electrodesNum, temporalDepth, netGName, rawSize, imgSize, dataProc.NORM_RANGE)
    netG.compile(optimizer=Adam(lr=learningRateG), loss=lossFuncG, metrics=[lossFuncG])
    netG.summary()
    if continueTrain == True:
        netG.load_weights(pretrainedModelPath)
    checkpointer = ModelCheckpoint(modelDir+'netG_latest.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
    earlyStopping = EarlyStopping(patience=earlyStoppingPatience, verbose=1)
    print('begin to train netG')
    historyG = netG.fit(x=ecg, y=mem, batch_size=batchSize, epochs=epochsNum, verbose=2, shuffle=True, validation_split=valSplit, callbacks=[checkpointer, earlyStopping])

def prediction(dataDir, dstDir, nameList, modelDir, priorEcgRange, priorMemRange, electrodesNum, rawSize=(200, 200), imgSize=(256, 256), channels=1, netGName='uNet',
activationG='relu', lossFuncG='mae', temporalDepth=None, gKernels=64, gKernelSize=3, batchSize=10):
    network = Networks(imgRows=imgSize[0], imgCols=imgSize[1], channels=channels, gKernels=gKernels, gKernelSize=gKernelSize, temporalDepth=temporalDepth)
    if netGName == 'uNet':
        netG = network.uNet()
    elif netGName == 'convLstm':
        netG = network.convLstm()
    elif netGName == 'uNet3d':
        netG = network.uNet3d()
    netG.load_weights(modelDir)
    for name in nameList:
        srcDir = dataDir+name+'/'+dataProc.ECG_FOLDER_NAME+'_%d/'%electrodesNum
        rawEcg = dataProc.loadData(srcDir=srcDir, resize=True, srcSize=rawSize, dstSize=imgSize, normalization=False)
        ecg = dataProc.create3dEcg(rawEcg, temporalDepth, netGName)
        ecg = dataProc.scale(ecg, priorEcgRange, dataProc.NORM_RANGE)
        mem = netG.predict(ecg, batch_size=batchSize, verbose=1)
        pngMem = mem*255
        mem = dataProc.scale(mem, dataProc.NORM_RANGE, priorMemRange)
        memDir = dstDir+name+'/'
        if not os.path.exists(memDir):
            os.mkdir(memDir)
        npyDir = memDir+'npy/'
        if not os.path.exists(npyDir):
            os.mkdir(npyDir)
        pngDir = memDir+'png/'
        if not os.path.exists(pngDir):
            os.mkdir(pngDir)
        for i in range (0, mem.shape[0]):
            resizedMem = cv.resize(mem[i], rawSize)
            np.save(npyDir+'%06d.npy'%i, resizedMem)
            cv.imwrite(pngDir+'%06d.png'%i, pngMem[i])
