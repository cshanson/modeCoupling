#!/scratch/ch3246/project/PythonEnv/miniconda3/envs/dalma-python3-CPU/bin/python

# Routine to bin all the sizes into population bins

import sys
sys.path.insert(0,pathToRoutines)
print("Importing pythonRoutines")
import numpy as NP
from pyCompHelio import *
from matplotlib.pyplot import *
from astropy.io import fits
from scipy.sparse import coo_matrix,csr_matrix,hstack,vstack
from astropy.convolution import Gaussian2DKernel as astropyGaussian2D
from astropy.convolution import Box2DKernel      as astropyBox2D
from astropy.convolution import convolve as astropyConv
plt.ion()

# plt.close('all')

LCT = True
OUTFLOWS = False


# Load in the data. In our paper, this is on a for loop for all cubes
with NP.load(DATADIR + '/avgSupergranuleData_SurfaceFlows.npz') as npyDICT:
	avg_Flow_maps = npyDICT['avg_Flow_maps']
	Roll_inds = npyDICT['Roll_inds']
	Amps = npyDICT['amplitudes']
	Sizes = npyDICT['sizes']

NP.savetxt('data/SG_Amplitudes.dat',Amps)
NP.savetxt('data/SG_size.dat' ,Sizes)
