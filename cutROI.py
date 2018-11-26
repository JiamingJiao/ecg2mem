import numpy as np
import dataProc

roiSize = 40
coordinates = np.array([[0, 0], [0, 19], [0, 39]], dtype=np.uint16) # relative position of electrodes in a sub area
dataList = ('', '')
phieDir = ''
for i in range(0, 5): # A 200x200 map will be divided into 25 40x40 maps
    for j in range(0, 5):
        absCoordinates = np.copy(coordinates)
        absCoordinates[:, 0] = np.add(absCoordinates[:, 0], roiSize*i)
        absCoordinates[:, 1] = np.add(absCoordinates[:, 1], roiSize*j)
        roiStartY = i*roiSize
        roiStartX = j*roiSize
        roiEndY = (i+1)*roiSize
        roiEndX = (j+1)*roiSize
        ecg = dataProc.SparsePecg((roiSize, roiSize), absCoordinates)
        for dataName in dataList:
            srcDir = phieDir + dataName + '/phie_'
            dstDir = phieDir + dataName + '_%d/'%(i*5 + j)
            phie = dataProc.loadData(srcDir)
            vmemDir = phieDir + dataName + '/vmem_'
            vmem = dataProc.loadData(vmemDir)
            for k in range(0, phie.shape[0]):
                ecg.calcPecg(phie[k])
                np.save(dstDir+'%06d'%k, ecg.dst[roiStartY:roiEndY, roiStartX:roiEndX])
