import os
import shutil
import glob
import json
import sys
import numpy as np

import dataProc
import train

def iterate(argsFilename='./iterationArgs.json'):
    with open(argsFilename) as argsFile:
        args = json.load(argsFile)
    simulationDir = args["simulationDir"]
    trainingDataList = args["train"]["nameList"]
    tempDir = args["tempDir"]
    # Copy data to a temp folder in working directory
    srcPhieDirList = dataProc.createDirList(simulationDir, trainingDataList, 'phie_')
    srcVmemDirList = dataProc.createDirList(simulationDir, trainingDataList, 'vmem_')
    phieDirList = dataProc.createDirList(tempDir+'phie/', trainingDataList)
    vmemDirList = dataProc.createDirList(tempDir+'true_vmem/', trainingDataList)
    for i in range(0, len(args["train"]["nameList"])):
        phieDir = phieDirList[i]
        os.makedirs(phieDir)
        srcPhieFiles = sorted(glob.glob(srcPhieDirList[i]+'*.npy'))
        vmemDir = vmemDirList[i]
        os.makedirs(vmemDir)
        srcVmemFiles = sorted(glob.glob(srcVmemDirList[i]+'*.npy'))
        for j in range(0, len(srcPhieFiles)):
            shutil.copy(srcPhieFiles[j], phieDir)
            shutil.copy(srcVmemFiles[j], vmemDir)
    print('phie and vmem copied')
    # Loop of reducing electrodes
    electrodesDir = args["electrodes"]["dir"]
    parentElectrodes = np.load(electrodesDir+args["electrodes"]["initial"])
    i = 0
    reducedNum = 0
    mapSize = args["mapSize"]
    fullSize = args["electrodes"]["fullSize"]
    pecgDirList = dataProc.createDirList(tempDir+'pecg/', trainingDataList)
    inputSize = args["net"]["inputSize"]
    trainArgs = (pecgDirList, vmemDirList, mapSize, inputSize, args["net"]["channels"], args["net"]["name"], args["net"]["activationG"], args["net"]["lossFuncG"],
    args["net"]["temporalDepth"], args["net"]["gKernels"], args["net"]["gKernelSize"], args["net"]["epochsNum"], args["net"]["batchSize"], args["net"]["learningRateG"],
    args["net"]["earlyStoppingPatience"], args["net"]["valSplit"])
    isFirstIteration = True
    bestParentModel = None
    while True:
        electrodesNum = parentElectrodes.shape[0]-reducedNum
        electrodesPath = args["electrodesDir"]["dir"] + args["experimentName"] + '_%d_%d.npy'%(i, electrodesNum)
        ecg = dataProc.AccurateSparsePecg(mapSize, reducedNum, fullSize, parentElectrodes, electrodesPath)
        disablePrint()
        print('Messages from method: AccurateSparsePecg.resizeAndCalc are muted!')
        ecg.resizeAndCalc(phieDirList, pecgDirList)
        enablePrint()
        modelPath = args["train"]["modelDir"] + args["experimentName"] + '/' + '%d_%d_'%(i, electrodesNum)
        train.trainG(*trainArgs, continueTrain=isFirstIteration, pretrainedModelPath=bestParentModel, modelDir=modelPath)
    return 0

def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

if __name__ == '__main__':
    iterate()
