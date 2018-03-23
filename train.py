#!/usr/bin/env python
# -*- coding: utf-8 -*-

from model import *

network = GAN()
network.trainGAN(extraPath = '/mnt/recordings/SimulationResults/mapping/2/train/extra/*.jpg',
memPath = '/mnt/recordings/SimulationResults/mapping/2/train/mem/*.jpg',
extraForFakePath = '/mnt/recordings/SimulationResults/mapping/2/train/extra_for_fake/*.jpg',
memRealPath = '/mnt/recordings/SimulationResults/mapping/2/train/mem_real/*.jpg',
modelPath = '/mnt/recordings/SimulationResults/mapping/2/checkpoints/20180322/')
