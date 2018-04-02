#!/usr/bin/env python
# -*- coding: utf-8 -*-

from model import *

network = GAN()
network.trainGAN(extraPath = '/mnt/recordings/SimulationResults/mapping/2D/train/sparse/25/extra/*.png',
memPath = '/mnt/recordings/SimulationResults/mapping/2D/train/mem/*.jpg',
extraForFakePath = '/mnt/recordings/SimulationResults/mapping/2D/train/sparse/25/extra_for_fake/*.png',
memRealPath = '/mnt/recordings/SimulationResults/mapping/2D/train/mem_real/*.jpg',
modelPath = '/mnt/recordings/SimulationResults/mapping/2D/checkpoints/20180402_1/', epochsNum = 100, lossRatio = 100)
