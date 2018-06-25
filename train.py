#!/usr/bin/env python
# -*- coding: utf-8 -*-

from model import *

network = GAN(netDName = 'VGG16', netGName = 'uNet', temporalDepth =3)
network.trainGAN(extraPath = '/mnt/recordings/SimulationResults/mapping/simulation_data/20180228-1/pseudoECG/',
                memPath = '/mnt/recordings/SimulationResults/mapping/simulation_data/20180228-1/mem/',
                modelPath ='/mnt/recordings/SimulationResults/mapping/training/checkpoints/20180622_4',
                epochsNum = 25, batchSize = 5, valSplit = 0.2)
