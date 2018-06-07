import cv2 as cv
import dataProc
import numpy as np

def accuracy(srcPath1, srcPath2, method = 'mae'):
    src1 = dataProc.loadImage(inputPath = srcPath1, normalization = 1)
    src2 = dataProc.loadImage(inputPath = srcPath2, normalization = 1)
    imgRows = float(src1.shape[1])
    imgCols = float(src1.shape[2])
    localAverage = 0
    if (method == 'mae'):
        for i in range(0, src1.shape[0]):
            absDiff = cv.absdiff(src1[i, :, :], src2[i, :, :])
            #percentageArray = cv.divide(absDiff+0.001, src2[i, :, :]+0.001)
            sum = cv.sumElems(absDiff)
            localAverage = sum[0]/(imgRows*imgCols)
        averageAcc = localAverage/(float(src1.shape[0]))
        return localAverage

acc1 = accuracy('C:/data/makeVideo/dst/20180604_1/png_2/', 'C:/data/makeVideo/simulation_data/20180308-1/mem/')
print(acc1)
acc2 = accuracy('C:/data/makeVideo/dst/20180606_1/png_1/', 'C:/data/makeVideo/simulation_data/20180308-1/mem/')
print(acc2)
acc3 = accuracy('C:/data/makeVideo/dst/20180606_2/png_2/', 'C:/data/makeVideo/simulation_data/20180308-1/mem/')
print(acc3)
