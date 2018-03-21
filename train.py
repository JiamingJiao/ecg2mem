#!/usr/bin/env python
# -*- coding: utf-8 -*-

from model_pix2pix import *

network = netG()
network.train(extraPath = '/mnt/recordings/SimulationResults/mapping/2/train/extra/*.jpg',
memPath = '/mnt/recordings/SimulationResults/mapping/2/train/mem/*.jpg',
savePath = '/mnt/recordings/SimulationResults/mapping/2/checkpoints/20180314.hdf5')
