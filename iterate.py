import os
import shutil
import distutils.dir_util
import json
import numpy as np
import csv
import copy

import dataProc
import train
import analysis

def searchElectrodes(argsPath='./iterationArgs.json'):
    with open(argsPath) as argsFile:
        args = json.load(argsFile)
    argsFile.close()
    if not os.path.exists(args["dir"]):
        os.makedirs(args["dir"])
    with open(os.path.join(args["dir"], 'parameters.json'), 'w') as outFile:
        json.dump(args, outFile, sort_keys=True, indent=4)
    # select data for training and selecting electrodes
    phieList = readDataList(args["phieList"])
    trainingDataSize = (int(len(phieList)*args["trainingSplit"]),) + tuple(args["sequenceSize"])
    selectionDataSize = (len(phieList) - trainingDataSize[0],) + tuple(args["sequenceSize"]) # size of data for selecting electrodes
    # set phie data
    initialElectrodes = np.load(args["electrodes"]["initial"])
    phieTraining = dataProc.Phie(initialElectrodes, *trainingDataSize)
    phieTraining.setData2(phieList[0:trainingDataSize[0]])
    phieSlection = dataProc.Phie(initialElectrodes, *selectionDataSize)
    phieSlection.setData2(phieList[trainingDataSize[0]:])
    # set vmem data
    vmemList = readDataList(args["vmemList"])
    vmemTraining = dataProc.Vmem(*trainingDataSize)
    vmemTraining.setData2(vmemList[0:trainingDataSize[0]])
    vmemSelection = dataProc.Vmem(*selectionDataSize)
    vmemSelection.setData2(vmemList[trainingDataSize[0]:])

    # compile model
    generator = train.Generator(**args["netg"])

    # train and predict on data of initial electrodes
    print('\n initial iteration \n')
    tempPhie = preprocessPhie(phieTraining, initialElectrodes)
    tempVmem = copy.deepcopy(vmemTraining)
    subFolder = os.path.join(args["tempDir"], 'initial')
    generator.train(tempPhie, tempVmem, **args["train"], constantLearningRate=False, modelDir=subFolder)

    tempPhie = preprocessPhie(phieSlection, initialElectrodes)
    tempVmem = copy.deepcopy(vmemSelection)
    mae, stdErr = generator.predict(tempPhie, None, tempVmem, os.path.join(subFolder, 'netg.h5'))
    np.save(os.path.join(subFolder, 'test_loss'), np.array([mae, stdErr]))

    parentElectrodes = np.copy(initialElectrodes)
    parentModelPath = os.path.join(subFolder, 'netg.h5')

    # start iteration
    schedule = np.array(args["schedule"], np.uint8)
    for [electrodesNum, iterationsNum] in schedule:
        bestMae = float('inf')
        for i in range(iterationsNum):
            print('\n %delectrodes %diter'%(electrodesNum, i))
            subFolder = os.path.join(args["tempDir"], '%delectrodes_%diter'%(electrodesNum, i))
            if not os.path.exists(subFolder):
                os.makedirs(subFolder)

            np.random.shuffle(parentElectrodes)
            currentElectrodes = parentElectrodes[0: electrodesNum]
            np.save(os.path.join(subFolder, 'coordinates'), currentElectrodes)
            _ = analysis.drawElectrodes(currentElectrodes, dstPath=os.path.join(subFolder, 'coordinates.png'))

            tempPhie = preprocessPhie(phieTraining, currentElectrodes)
            tempVmem = copy.deepcopy(vmemTraining)
            generator.train(tempPhie, tempVmem, **args["train"], constantLearningRate=True, modelDir=subFolder, continueTrain=True, pretrainedModelPath=parentModelPath)

            tempPhie = preprocessPhie(phieSlection, currentElectrodes)
            tempVmem = copy.deepcopy(vmemSelection)
            mae, stdErr = generator.predict(tempPhie, None, tempVmem, os.path.join(subFolder, 'netg.h5'))
            np.save(os.path.join(subFolder, 'test_loss'), np.array([mae, stdErr]))

            if mae < bestMae:
                print('\n MAE reduced from %f to %f'%(bestMae, mae))
                bestMae = mae
                bestElectrodes = currentElectrodes
                bestModelPath = os.path.join(subFolder, 'netg.h5')
            elif not args["saveModels"] == 'all':
                os.remove(os.path.join(subFolder, 'netg.h5'))
        
        if args["saveModels"] == 'no':
            os.remove(parentModelPath)

        parentElectrodes = bestElectrodes
        parentModelPath = bestModelPath

    if args["saveModels"] == 'no':
        os.remove(parentModelPath)
    distutils.dir_util.copy_tree(args["tempDir"], args["dir"])
    shutil.rmtree(args["tempDir"])

def readDataList(path):
    dst = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            for col in row:
                dst.append(col)
    return dst

def preprocessPhie(src, coordinates):
    dst = copy.deepcopy(src)
    dst.coordinates = np.copy(coordinates)
    dst.downSample()
    dst.setGround(0)
    return dst
