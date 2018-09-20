import numpy as np
import model
import dataProc

def trainG(dataDirList, modelDir, imgRows=256, imgCols=256, channels=1, netGName='uNet', activationG='relu', lossFuncG='mae', temporalDepth=None, gKernels=64, gKernelSize=4,
epochsNum=100, batchSize=10, learningRateG=1e-4, earlyStoppingPatience=10, valSplit=0.2, continueTrain=False, pretrainedModelPath=None):
    if activationG == 'tanh':
        dataRange = [-1., 1.]
    else:
        dataRange = [0., 1.]
    pEcg, vMem = dataProc.merge