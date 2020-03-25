# -*- coding: utf-8 -*-
#
# Yufeng Huang
# Department of Chemical Engineering
# California Institute of Technology
# 11/07/2018
#

# Generate features from VASP POSCAR

import numpy as np
import argparse

def getCoord(poscar):
    with open(poscar,'r') as p:
        nAtoms=0
        lattice = np.zeros((3,3))
        p.readline()
        p.readline()
        lattice[0,:] = np.array(p.readline().split(),dtype=float)
        lattice[1,:] = np.array(p.readline().split(),dtype=float)
        lattice[2,:] = np.array(p.readline().split(),dtype=float)
        p.readline()
        nAtoms = int(p.readline())
        if (p.readline().strip(" "))[0].upper() == "S":
            p.readline()
        R = np.zeros((nAtoms,3))
        for i in range(nAtoms):
            R[i] = np.array(p.readline().split()[:3],dtype=float)
        R = R - [0.5,0.5,0.5]
    return nAtoms, R.dot(lattice)


def getCos(x, numBasis):
    nodes = np.linspace(-1,1,numBasis)
    y = x[:,np.newaxis] - nodes
    h = 2/(numBasis-1)
    zeroMask = (y ==0)
    y[np.abs(y)>h] = 0
    y[y!=0] = np.cos(y[y!=0]/h*np.pi)/2+0.5
    y[zeroMask] = 1
    y[np.abs(x)>1] = 0
    return y


def getFeat(nAtoms, coord,n2b,n3b):
    Rl = np.sqrt(np.sum(coord**2,axis=1))[1:]
    Dc = coord[1:,np.newaxis] - coord[1:]
    Dc = np.sqrt(np.sum(Dc**2,axis=2))
    yR = np.sum(getCos(Rl/4-1,n2b),axis=0)
    yD = np.zeros((nAtoms-1,nAtoms-1,n3b))
    yD[Dc!=0] = getCos(Dc[Dc!=0]/4-1,n3b)
    yD = np.sum(getCos(Rl/4-1,n3b)[:,np.newaxis,:,np.newaxis] * yD[:,:,np.newaxis,:],axis=0)
    yD = np.sum(getCos(Rl/4-1,n3b)[:,np.newaxis,:,np.newaxis] * yD[:,:,np.newaxis,:],axis=0)
    return np.concatenate([yR, yD.reshape(-1)])


parser = argparse.ArgumentParser()
parser.add_argument("POSCAR", type=str, help="POSCAR file. \
                    The coordinates must be fractional (or direct). \
                    The center atom must be in the center, (0.5, 0.5, 0.5)")

args = parser.parse_args()
poscar = getattr(args, "POSCAR")
numA, coord = getCoord(poscar)
features = getFeat(numA, coord, 12, 3)
print(features)