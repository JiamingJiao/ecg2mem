import os
import shutil
import glob
import json
import sys
import numpy as np

import dataProc
import train
import analysis

ZERO = 1e-4

def iterate(argsFilename='./iterationArgs.json'):
    with open(argsFilename) as argsFile:
        args = json.load(argsFile)
    simulationDir = args["simulationDir"]
    trainingDataList = args["training"]["nameList"]
    predictionDataList = args["prediction"]["nameList"]    
    dataList = trainingDataList + predictionDataList
    trainingDataNum = len(trainingDataList)
    predictionDataNum = len(predictionDataList)
    tempDir = args["tempDir"]
    # Copy data to a temp folder in working directory
    srcPhieDirList = dataProc.createDirList(simulationDir, dataList, 'phie_')
    srcTrueVmemDirList = dataProc.createDirList(simulationDir, dataList, 'vmem_')
    phieDirList = dataProc.createDirList(tempDir+'phie/', dataList)
    trueVmemDirList = dataProc.createDirList(tempDir+'true_vmem/', dataList)
    vmemDirList = dataProc.createDirList(tempDir+'vmem/', predictionDataList)
    for i in range(0, len(dataList)):
        phieDir = phieDirList[i]
        os.makedirs(phieDir)
        srcPhieFiles = sorted(glob.glob(srcPhieDirList[i]+'*.npy'))
        vmemDir = trueVmemDirList[i]
        os.makedirs(vmemDir)
        srcVmemFiles = sorted(glob.glob(srcTrueVmemDirList[i]+'*.npy'))
        for j in range(0, len(srcPhieFiles)):
            shutil.copy(srcPhieFiles[j], phieDir)
            shutil.copy(srcVmemFiles[j], vmemDir)
    print('phie and vmem copied')
    # Loop of reducing electrodes
    electrodesDir = args["electrodes"]["dir"]
    parentElectrodes = np.load(electrodesDir+args["electrodes"]["initial"])
    i = 0
    stepSize = 0
    mapSize = args["mapSize"]
    fullSize = args["electrodes"]["fullSize"]
    pecgDirList = dataProc.createDirList(tempDir+'pecg/', trainingDataList)
    inputSize = args["net"]["inputSize"]
    generatorArgs = (args["net"]["name"], mapSize, args["net"]["batchSize"], inputSize, args["net"]["channels"], args["net"]["gKernels"], args["net"]["gKernelSize"],
    args["net"]["temporalDepth"], args["net"]["activationG"], args["net"]["lossFuncG"])
    generator = train.Generator(generatorArgs)
    trainArgs = (args["net"]["lossFuncG"], args["net"]["epochsNum"], args["net"]["earlyStoppingPatience"], args["net"]["valSplit"])
    notFirstIteration = False
    bestParentModelPath = None
    # Load true vmem for evaluation
    trueVmem = np.empty((predictionDataNum), object)
    for i in range(0, predictionDataNum):
        trueVmem[i] = dataProc.loadData(dataList[trainingDataNum+i])
    trueVmem = np.concatenate(trueVmem)
    threshold = args["iteration"]["maeThreshold"]
    condition = True
    while condition:
        electrodesNum = parentElectrodes.shape[0]-stepSize
        electrodesPath = args["electrodesDir"]["dir"] + args["experimentName"] + '_%d_%d.npy'%(i, electrodesNum)
        ecg = dataProc.AccurateSparsePecg(mapSize, stepSize, fullSize, parentElectrodes, electrodesPath)
        disablePrint()
        print('Messages from method: AccurateSparsePecg.resizeAndCalc are muted!')
        ecg.resizeAndCalc(phieDirList, pecgDirList)
        enablePrint()
        del ecg
        modelPath = args["training"]["modelDir"] + args["experimentName"] + '/' + '%d_%d_'%(i, electrodesNum)
        dataRange, history = generator.train(pecgDirList[:trainingDataNum], trueVmemDirList[:trainingDataNum], notFirstIteration, bestParentModelPath, modelPath, *trainArgs)
        notFirstIteration = True
        #prediction and evaluation
        vmem = generator.predict(pecgDirList[trainingDataNum:], vmemDirList, modelPath, dataRange[0:2], dataRange[2:4], args["net"]["batchSize"])
        vmem = np.concatenate(vmem)
        mae = analysis.calculateMae(vmem, trueVmem)
        # Find next electrodes number
        parentElectrodes = np.load(electrodesPath)
        if abs(mae-threshold) >= ZERO:
            if notFirstIteration:
                k = (parentMae-mae) / (parentElectrodes.shape[0]-electrodesNum)
                stepSize = (mae-threshold) / k
            else:
                if mae >= threshold:
                    print('MAE of initial electrodes is larger than threshold. ')
                    condition = False
                stepSize = args["iteration"]["initialStep"]
            #nextElectrodesNum = parentElectrodes.shape[0]-stepSize
            parentMae = mae
        else:
            condition = False
        if condition == False:
            print('Iteration stopped. ')
    return 0

def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

if __name__ == '__main__':
    iterate()
