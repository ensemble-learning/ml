#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yufeng Huang
"""

import numpy as np


def loadNN(paramFile):
    nnParams = {}
    params = np.load(paramFile)
    nnParams['nL1'] = int(params['nL1'])
    nnParams['nL2'] = int(params['nL2'])
    nnParams['learningRate'] = float(params['learningRate'])
    nnParams['nEpoch'] = int(params['nEpoch'])
    return nnParams


def loadFeat(paramFile):
    featParams = {}
    params = np.load(paramFile)
    featParams['featA'] = params['featA']
    featParams['featB'] = params['featB']
    featParams['n2b'] = int(params['n2b'])
    featParams['n3b'] = int(params['n3b'])
    featParams['nFeat'] = int(params['nFeat'])
    return featParams
