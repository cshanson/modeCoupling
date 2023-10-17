#!/opt/local/anaconda/anaconda-2.2.0/bin/python

pathToRoutines = '../..'
import sys
sys.path.insert(0,pathToRoutines)

import numpy     as NP
from pyCompHelio import *
import h5py

func   = loadObject(sys.argv[1])
args   = loadObject(sys.argv[2])
myNode = int(sys.argv[3])
nNodes = int(sys.argv[4])
N      = int(sys.argv[5])
rType  = sys.argv[6]
resDir = sys.argv[7]
saveH5py   = int(sys.argv[8])

# Compute partition and get range to compute

starts  = partition(N,nNodes)
myStart = starts[myNode]
myEnd   = starts[myNode+1]

# Build arguments table
if (rType is not None and rType[:3] == 'AVE'):
  weights = rType[3:]
  firstArg = 2
else:
  weights = None
  firstArg = 1

argsTable = buildArgTable(args,N,nNodes,False,weights)[myNode]
resTmp = func(*(argsTable[0][firstArg:]))

if rType == 'None':
  res = [NP.asarray(resTmp)]
elif rType[:3] == 'AVE':
  res = argsTable[0][1]*NP.asarray(resTmp)
else:
  res = NP.asarray(resTmp)

for i in range(myStart+1,myEnd):

  resTmp = NP.asarray(func(*(argsTable[i-myStart][firstArg:])))

  if rType == 'None':
    res.append(resTmp)
  elif rType == 'SUM':
    res += resTmp
  elif rType[:3] == 'AVE':
    res += argsTable[i-myStart][1]*resTmp
  elif rType == 'PROD':
    res *= resTmp
  elif rType == 'MAX':
    res  = NP.maximum(res,resTmp)
  elif rType == 'MIN':
    res  = NP.minimum(res,resTmp)

if rType == 'None':
  res = NP.array(res)

if saveH5py:
  h5f = h5py.File('%s/res_%d_%d.h5'%(resDir,myStart,myEnd), 'w')
  h5f.create_dataset('res', data=res)
  h5f.close()
else:
  NP.save('%s/res_%d_%d.npy'%(resDir,myStart,myEnd),res)

