import numpy as np
import dataProc
from model import networks
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler

def trainG(dataDirList, modelDir, electrodesNum, rawSize=(200, 200), imgSize=(256, 256), channels=1, netGName='uNet', activationG='relu', lossFuncG='mae', temporalDepth=None,
gKernels=64, gKernelSize=4, epochsNum=100, batchSize=10, learningRateG=1e-4, earlyStoppingPatience=10, valSplit=0.2, continueTrain=False, pretrainedModelPath=None):
    network = networks(imgRows=imgSize[0], imgCols=imgSize[1], channels=channels, gKernels=gKernels, gKernelSize=gKernelSize, temporalDepth=temporalDepth)
    if activationG == 'tanh':
        normalizationRange = [-1., 1.]
    else:
        normalizationRange = [0., 1.]
    if netGName == 'convLstm':
        netG = network.convLstm()
        pEcg, mem = dataProc.mergeSequence(dataDirList, electrodesNum, temporalDepth, netGName, rawSize, imgSize, normalizationRange)
    netG.compile(optimizer=Adam(lr=learningRateG), loss=lossFuncG, metrics=[lossFuncG])
    netG.summary()
    if continueTrain == True:
        netG.load_weights(pretrainedModelPath)
    checkpointer = ModelCheckpoint(modelDir+'netG_latest.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
    earlyStopping = EarlyStopping(patience=earlyStoppingPatience, verbose=1)
    print('begin to train netG')
    historyG = netG.fit(x=pEcg, y=mem, batch_size=batchSize, epochs=epochsNum, verbose=2, shuffle=True, validation_split=valSplit, callbacks=[checkpointer, earlyStopping])

def prediction(srcDirList, dstDirList, modelDir, electrodesNum, rawSize=(200, 200), imgSize=(256, 256), channels=1, netGName='uNet', activationG='relu', lossFuncG='mae',
temporalDepth=None, gKernels=64, gKernelSize=4):
    network = networks(imgRows=imgSize[0], imgCols=imgSize[1], channels=channels, gKernels=gKernels, gKernelSize=gKernelSize, temporalDepth=temporalDepth)
    for srcDir in srcDirList:
        if activationG == 'tanh':
            normalizationRange = [-1, 1]
        else:
            normalizationRange = [0, 1]
    if netGName == 'convLstm':
        netG = network.convLstm()
    netG.load_weights(modelDir)
    