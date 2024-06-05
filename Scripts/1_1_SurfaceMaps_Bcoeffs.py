#!/scratch/ch3246/project/PythonEnv/miniconda3/envs/dalma-python3-CPU/bin/python

# Routine to compute the B coefficients at the surface. Not key to inversion, but useful

import sys
sys.path.insert(0,pathToRoutines)
print("Importing pythonRoutines")
import numpy as NP
from pyCompHelio import *
from matplotlib.pyplot import *
from astropy.io import fits
plt.ion()

plt.close('all')


# min and max k to compute the coefficients from
kRmax = 1000
kRmin = 900

# number of linewidths to integrate the mode coupling. See epsilon in Hanson et al. 2021
WINDFACTOR = 2
modeSpacingFactor = None


# Padding so that we can reconstruct in real space
nPad = 70


# initialize mode coupling and load in the fourier coefficients
cart_MC = cartesian_modeCoupling(DATADIR + '/V.fits',\
					dxFactor=2,dyFactor=2,kMax=2000/RSUN,\
					OmegaMax=0.006*2*NP.pi,\
					timeSubSample=1)
phi_kw = cart_MC.computeFFT(storeInInstance=True,\
							fitsPathLoad = DATADIR + '//V_kw.fits')


# Compute the kx,ky omega grid
kx,ky,omega = cart_MC.computeFreq()
dkx = NP.diff(kx)[0];dky = NP.diff(ky)[0];dw = NP.diff(omega)[0]
kxg,kyg = NP.meshgrid(kx,ky,indexing='ij')
abskg = NP.sqrt(kxg**2+kyg**2)
absk = NP.sqrt(kx[kx>=0]**2+ky[ky>=0]**2)


# The radial mode to compute the couplings
nn = 0

# QX, QY and SIGMA to compute couplings (hanson et al. 2021). SIGMA = 0: Mean flows
QX = NP.arange(-30,31) 
QY = NP.arange(-30,31)
SIGMA = NP.array([0])


# Create grids for computations
qxgrid,qygrid = NP.meshgrid(QX*dkx,QY*dky,indexing='ij')
xgridI = NP.arange(QX.min()-nPad,QX.max()+nPad+1)*0.16/180*RSUN*NP.pi /1e6
xgrid,ygrid = NP.meshgrid(xgridI,xgridI,indexing='ij')


# Generate mask, for which modes not used in coupling are set to zero.
omegaM,gamma,amp = cart_MC.ref_fit_params(absk,nn)
indl = NP.argmin(abs(NP.nan_to_num(omegaM) / (2*NP.pi)-0.002)) # mask low freq
indr = NP.argmin(abs(NP.nan_to_num(omegaM) / (2*NP.pi)-0.005)) # mask high freq
mask = NP.where((abskg*RSUN < min(absk[indr]*RSUN,kRmax))*(abskg*RSUN > max(absk[indl]*RSUN,kRmin)),1,0)


# get the indices of the fourier cube that are allowed by mask
inds_kx,inds_ky = NP.where(mask)

# Compute the azimuthally averaged amplitudes of the modes (N_nk of Hanson et al. 2021)
amps_avg = cart_MC.compute_N_nk(mask,nn,25,num_linewidths=2,PLOT = False)


# $ Compute the bcoeffs
kxm,kym,Bcoeff_grid = cart_MC.compute_bcoeffs_parallel(nn,mask,QX,QY,SIGMA,amps_avg,nu_min=0.0015,nu_max=0.0055,absq_range=NP.array([0,250])/RSUN,\
												VERBOSE=True,nbProc=4,reorg_k=False,returnNoise=False,\
												windfactor=WINDFACTOR,modeSpacingFactor=modeSpacingFactor)

# Compute the associated Noise model
print('\tComputing Noise Model')
kxmN,kymN,Noise_grid = cart_MC.compute_bcoeffs_parallel(nn,mask,QX,QY,SIGMA,amps_avg,nu_min=0.0015,nu_max=0.0055,absq_range=NP.array([0,250])/RSUN,\
													VERBOSE=True,nbProc=4,reorg_k=False,returnNoise='theory',\
													windfactor=WINDFACTOR,modeSpacingFactor=modeSpacingFactor,rtype=None)

# Transform the B coefficents to real space
polMap = NP.fft.ifftn(NP.fft.ifftshift(NP.pad(NP.nansum(Bcoeff_grid,axis=-1).squeeze(),((nPad,nPad),(nPad,nPad)),constant_values=0)),norm='ortho')
Flow_Maps = polMap.real




# SAVE the data
Bcoeff_grid = NP.array(Bcoeff_grid)
mkdir_p(DATADIR + '/Surface_Flow_Maps/' )


with h5py.File(DATADIR + '/Surface_Flow_Maps/SurfaceFlows_Bcoeffs.h5','w') as h5f:
	h5f.create_dataset('QX',data = QX)
	h5f.create_dataset('QY',data = QY)
	h5f.create_dataset('SIGMA',data = SIGMA)

	h5f.create_dataset('xgrid',data = xgrid)
	h5f.create_dataset('ygrid',data = ygrid)

	h5f.create_dataset('kRmax',data = kRmax)
	h5f.create_dataset('kRmin',data = kRmin)

	h5f.create_dataset('kxm',data = kxm)
	h5f.create_dataset('kym',data = kym)
	h5f.create_dataset('mask',data = mask)

	h5f.create_dataset('dkx',data = dkx)
	h5f.create_dataset('dky',data = dky)
	h5f.create_dataset('dw',data = dw)

	
	h5f.create_dataset('Flow_Maps',data = Flow_Maps)

	h5f.create_dataset('nPad',data = nPad)
	h5f.create_dataset('WINDFACTOR',data = WINDFACTOR)

	Bgrp = h5f.create_group('Bcoeffs')
	Ngrp = h5f.create_group('NoiseModel')
	for ii in range(len(QX)):
		Bsubgrp    = Bgrp.create_group('QX%i' % QX[ii])
		Nsubgrp    = Ngrp.create_group('QX%i' % QX[ii])
		for jj in range(len(QY)):
			Bssubgrp    = Bsubgrp.create_group('QY%i' % QY[jj])
			Nssubgrp    = Nsubgrp.create_group('QY%i' % QY[jj])
			for kk in range(len(SIGMA)):
				Bsssubgrp    = Bssubgrp.create_group('SIGMA%i' % SIGMA[kk])
				Nsssubgrp    = Nssubgrp.create_group('SIGMA%i' % SIGMA[kk])
				
				Bsssubgrp.create_dataset('data',data = Bcoeff_grid[ii,jj,kk])
				Nsssubgrp.create_dataset('data',data = Noise_grid[ii,jj,kk])