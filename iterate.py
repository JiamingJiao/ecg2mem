import os
import shutil
import glob

import dataProc
import train

def iterate(simulationDir, trainingDataList, initialElectrodesDir, predictionDataList, **trainArgs):
    # Store training data temporally in GPUWS, delete it after training.
    srcPhieDirList = dataProc.createDirList(simulationDir, trainingDataList, 'phie_')
    srcVmemDirList = dataProc.createDirList(simulationDir, trainingDataList, 'vmem_')
    phieDirList = dataProc.createDirList('./temp/phie/', trainingDataList)
    vmemDirList = dataProc.createDirList('./temp/true_vmem/', trainingDataList)
    for i in range(0, len(trainingDataList)):
        phieDir = phieDirList[i]
        os.makedirs(phieDir)
        srcPhieFiles = sorted(glob.glob(srcPhieDirList[i]+'*.npy'))
        vmemDir = vmemDirList[i]
        os.makedirs(vmemDir)
        srcVmemFiles = sorted(glob.glob(srcVmemDirList[i]+'*.npy'))
        for j in range(0, len(files)):
            shutil.copy(srcPhieFiles[j], phieDir)
            shutil.copy(srcVmemFiles[j], vmemDir)
    print('phie and vmem copied')
    initialElectrodes = np.load(initialElectrodesDir)
    return 0

if __name__ == '__main__':
    iterate()
