#!/scratch/ch3246/project/PythonEnv/miniconda3/envs/dalma-python3-CPU/bin/python


# Routine to build the LCT maps from intensity images
pathToRoutines = '/home/ch3246/mps_montjoie/'
import sys
sys.path.insert(0,pathToRoutines)
print("Importing pythonRoutines")
import numpy as NP
from pyCompHelio import *
from matplotlib.pyplot import *
from astropy.io import fits
import pyflct

plt.ion()
plt.close('all')

# Set to true if you want to recompute LCT maps
OVERWRITE = False

# initialize a timer
tini = time.time()


# Check if OVERWRITE or continuum data is missing
if os.path.isfile(DATADIR + '/LCT_maps.npz') and not OVERWRITE:
	raise(Exception('LCT file exists, skipping calculation'))
if not os.path.isfile(DATADIR + '/continuum.fits'):
	raise(Exception('Continuum file missing'))


# Initialize the class for continuum data
# Note for LCT to work, you have to use full resolution continuum data
cart_MC = cartesian_modeCoupling(DATADIR + '/continuum.fits',\
									dxFactor=1,\
									dyFactor=1,\
									kMax=5000/RSUN,\
									OmegaMax=0.006*2*NP.pi,\
									timeSubSample=1)

# Get real space grid for LCT calc
xgrid,ygrid = cart_MC.computeRealSpace()[:2]
dx = NP.diff(xgrid)[0]


# Load in the continuum data into variable
phi_xyt,header = cart_MC.readFits()


# Define function to compute LCT. Can then be used in parallel
def computeLCT_mca(ID,dt,dx,gaussianWidth,quiet=True,kr=None,skip=None):
	# See pyFLCT for options in LCT calc
	if kr is not None:
		tmp =  NP.array(pyflct.flct(phi_xyt[ID],phi_xyt[ID+1],dt,dx,gaussianWidth,quiet=quiet,thresh=0,kr=kr,skip=skip)[:2])
	else:
		tmp =  NP.array(pyflct.flct(phi_xyt[ID],phi_xyt[ID+1],dt,dx,gaussianWidth,quiet=quiet,skip=skip)[:2])

	# Subsample the flow maps
	if skip is not None:
		tmp = tmp[:,::skip,::skip]
	return tmp


# Run the LCT routine in parallel. See parallelTools.py
V = reduce(computeLCT_mca,(NP.arange(len(phi_xyt)-1),cart_MC.dt_,dx,4.,True,None,4),len(phi_xyt)-1,26,progressBar=True)
VX,VY = NP.nanmean(V,axis=-1);
xgrid = xgrid[::4];ygrid = ygrid[::4]

del phi_xyt


# Save data
NP.savez_compressed(DATADIR + '/LCT_maps.npz',\
					VX = VX,\
					VY = VY,\
					xgrid = xgrid - (xgrid[-1])/2,\
					ygrid = ygrid - (ygrid[-1])/2)

# Report calculation time
print('Time elapsed: ',time.time() - tini,' seconds')
