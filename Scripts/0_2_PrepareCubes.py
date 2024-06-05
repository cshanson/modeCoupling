# Routine to read in the doppler cube and compute the fourier transform

import sys
sys.path.insert(0,pathToRoutines)
print("Importing pythonRoutines")
import numpy as NP
from pyCompHelio import *
from matplotlib.pyplot import *
from astropy.io import fits

# Enable interactive plots
plt.ion()

# Initialize the instance
cart_MC = cartesian_modeCoupling(DATADIR + '/V.fits',\ # location of doppler cube
									dxFactor=2,\ # subsample factor in x from original HMI res
									dyFactor=2,\ # subsample factor in y from original HMI res
									kMax=2000/RSUN,\ # What is the max k to save to
									OmegaMax=0.006*2*NP.pi,\ # what is the max frequency to store
									timeSubSample=1) # Subsample rate in time. Usually leave as 1 to avoid alias	

# Read in the doppler cubes
phi_xyt = cart_MC.readFits(storeInInstance=True)[0]

# Compute and save the fourier coefficients
phi_kw = cart_MC.computeFFT(storeInInstance=True,\
							fitsPathSave=DATADIR +'/V_kw.fits')

# Simple plots to ensure your choices in dx and dy Factor are correct for the original cube
# Ridges should line up with the observed and reported frequencies
cart_MC.test_plots(radial_order = 0,nu = 0.003,vmaxScale=0.1,lineWidthFactor=2)

