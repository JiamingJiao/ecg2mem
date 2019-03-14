import keras.utils
import math
import numpy as np

class generatorRnn(keras.utils.Sequence):
    # batch generator, generate data from memory
    # generate and read from memory, or generate and save to disk then read data from disk?
    def __init__(self, pecg, vmem, batchSize):
        self.pecg = pecg # 3D data, a view of the 2D data array
        self.vmem = vmem
        self.groups = self.pecg.shape[0]
        self.length = self.pecg.shape[1]
        self.batchSize = batchSize
        
    def __len__(self):
        return int(self.groups*self.length/self.batchSize)

    def __getitem__(self, index):
        groupIndexBegin = int(math.floor(index/self.length))
        groupIndexEnd = int(math.floor((index+self.batchSize)/self.length))
        timeIndexBegin = int(index % self.length)
        timeIndexEnd = int((index+self.batchSize) % self.length)
        if groupIndexBegin == groupIndexEnd:
            pecgBatch = self.pecg[groupIndexBegin, timeIndexBegin: timeIndexEnd]
            vmemBatch = self.vmem[groupIndexBegin, timeIndexBegin: timeIndexEnd]
        else:
            pecgBatch = np.concatenate((self.pecg[groupIndexBegin, timeIndexBegin:], self.pecg[groupIndexEnd, :timeIndexEnd]))
            vmemBatch = np.concatenate((self.vmem[groupIndexBegin, timeIndexBegin:], self.vmem[groupIndexEnd, :timeIndexEnd]))
        return (pecgBatch, vmemBatch)
