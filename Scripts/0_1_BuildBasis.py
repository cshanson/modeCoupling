# Routine to build the depth basis functions

import numpy as NP
from pyCompHelio import *
from matplotlib.pyplot import *

#---------------------------------------------------------------------

# Load in the z grid, sound speed and density. Compute the acoustic depth
with NP.load('../../eigenfunctions_combined/eigs00.npz') as DICT:
	z      = DICT['z'] # zgrid
	cs     = DICT['cs'] # sound speed
	rho    = DICT['rho'] # density
	ad     = NP.zeros(len(z)) # acoustic depth vector. Here we initialize
	for ii in range(len(ad)):
		# Compute the acoustic depth
		ad[ii] = simps(1/cs[ii:],x=z[ii:])  

# Compute uniform grid in acoustic depth
tt = NP.linspace(ad.max(),ad.min(),20) 

# Assign the locations of the knots
knots  = NP.interp(tt,NP.flip(ad) ,NP.flip(z) )


# Create basis function
Basis1D = BsplineBasis1D(z,knots,3)
BasisStr = '_Bspline'

# Save basis function
NP.save('../../OUTPUTDIR/Basis%s.npy' % (BasisStr),[Basis1D])