#!/usr/bin/env python
# -*- coding: utf-8 -*-

from model import *

network = GAN()
network.trainGAN(extraPath = '/mnt/recordings/SimulationResults/mapping/2D/train/sparse/25/extra/',
memPath = '/mnt/recordings/SimulationResults/mapping/2D/train/mem/',
extraForFakePath = '/mnt/recordings/SimulationResults/mapping/2D/train/sparse/25/extra_for_fake/',
memRealPath = '/mnt/recordings/SimulationResults/mapping/2D/train/mem_real/',
modelPath = '/mnt/recordings/SimulationResults/mapping/2D/checkpoints/20180402_1/', epochsNum = 100, lossRatio = 100)
