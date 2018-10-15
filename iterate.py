import os
import shutil

import dataProc
import train

def iterate(simulationDir, dataDir, trainingDataList, predictionDataList, **trainArgs):
    # Store training data temporally in GPUWS, delete it after training.
    if os.path.exists('./temp/phie/'):
        os.makedirs('./temp/phie')
    if os.path.exists('./temp/true_mem/'):
        os.mkdir('./temp/true_mem/')
    srcDirList = dataProc.createDirList()
    return 0

if __name__ == '__main__':
    iterate()
