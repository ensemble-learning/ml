import py_util as pyu
import py_func as pyf
import numpy as np

nnParams = pyu.loadNN("log/nnParams.npz")
featParams = pyu.loadFeat("log/featParams.npz")

featSets = np.load("features.npz")
engySets = np.load("energies.npz")
featSets = [featSets[key] for key in featSets.files][0]
engySets = [engySets[key] for key in engySets.files][0]

pyf.trainEL_getError(featSets, engySets, featParams, nnParams)