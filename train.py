#!/usr/bin/env python
# -*- coding: utf-8 -*-

from model import *

network = GAN()
network.trainGAN(extraPath = '/mnt/recordings/SimulationResults/mapping/2D/train/sparse/20180416/extra/',
memPath = '/mnt/recordings/SimulationResults/mapping/2D/train/mem/',
modelPath = '/mnt/recordings/SimulationResults/mapping/2D/checkpoints/20180416_1/',
epochsNum = 200, batchSize = 5)
