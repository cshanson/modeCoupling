#!/scratch/ch3246/project/PythonEnv/miniconda3/envs/dalma-python3-CPU/bin/python

# Routine to build the average ur flow maps

import sys
sys.path.insert(0,pathToRoutines)
print("Importing pythonRoutines")
import numpy as NP
from pyCompHelio import *
from matplotlib.pyplot import *
from astropy.io import fits


plt.ion()

plt.close('all')

OUTFLOWS = False

#-------------------------------------------------
# Of all the meaured amps and sizes, build population groups
percentLimits     = NP.arange(0,101,10)#[0,20,40,60,80,100]

Sizes_all         = NP.genfromtxt('data/SG_size.dat')
Amplitudes_all    = NP.genfromtxt('data/SG_Amplitudes.dat')
percentiles_amps  = NP.percentile(Amplitudes_all,percentLimits)
percentiles       = NP.percentile(Sizes_all,percentLimits)

# Initialize a counter for how many SG are in each population
Ncount      = NP.zeros(len(percentiles))
Ncount_amps = NP.zeros(len(percentiles))


# Load in the location data, will need to be looped for each doppler
FILENAME = DATADIR + '/avgSupergranuleData_SurfaceFlows.npz'
with NP.load(FILENAME) as nypdict:
	Roll_inds  = nypdict['Roll_inds']
	Roll_inds_size  = nypdict['Roll_inds_size']
	xgrid_tmp  = nypdict['xgrid']
	ygrid_tmp  = nypdict['ygrid']
	Ncount_tmp = nypdict['Ncount']
	amplitudes = nypdict['amplitudes']
	sizes      = nypdict['sizes']
	# nPad = nPad
Ncount[-1]      += len(Roll_inds_size)
Ncount_amps[-1] += len(Roll_inds)

# initialize the average Doppler maps.
DoppAvg      = NP.zeros((len(percentiles),len(xgrid_tmp),len(ygrid_tmp)))
DoppAvg_amps = NP.zeros((len(percentiles),len(xgrid_tmp),len(ygrid_tmp)))
Dopp_var     = []


# Load in the Doppler cube, put in for loop if needed
cart_MC = cartesian_modeCoupling(DATADIR + '/V.fits',\
									apodization_fitsPath = os.getcwd() + '/Apodize.fits',\
									dxFactor=2,dyFactor=2,kMax=5000/RSUN,OmegaMax=0.006*2*NP.pi,\
									timeSubSample=1)
if NP.round(cart_MC.Nx_/len(xgrid_tmp)) != 2:
	raise Exception('Error: we are assuming Nx = 2*len(xgrid_tmp)')
else:
	subSample = 2

phi_xyt,header = cart_MC.readFits()
phi_xyt = phi_xyt - NP.mean(phi_xyt[640:1280]) # subtract mean signals (rotation etc)
phi_xyt = NP.nanmean(phi_xyt[640:1280],axis=0).T[::subSample,::subSample] # average over timne

# subtract further mean signals
phi_xyt = phi_xyt - NP.nanmean(phi_xyt,axis=0)[None,:] - NP.nanmean(phi_xyt,axis=1)[:,None]


# Build the average Doppler map for each population group
# Using amplitudes
for ii in range(len(Roll_inds)):
	xinds,yinds = Roll_inds[ii]
	DoppAvg_amps[-1] += NP.roll(NP.roll(phi_xyt,int(xinds),axis=0),int(yinds),axis=1)
	# Dopp_var.append(NP.roll(NP.roll(phi_xyt,int(xinds),axis=0),int(yinds),axis=1))
	for jj in range(len(percentiles) -1 ):
		if amplitudes[ii] >= percentiles_amps[jj] and amplitudes[ii] < percentiles_amps[jj+1]:
			DoppAvg_amps[jj] += NP.roll(NP.roll(phi_xyt,int(xinds),axis=0),int(yinds),axis=1)
			Ncount_amps[jj] += 1

# using sizes
for ii in range(len(Roll_inds_size)):
	xinds,yinds = Roll_inds_size[ii]
	DoppAvg[-1]      += NP.roll(NP.roll(phi_xyt,int(xinds),axis=0),int(yinds),axis=1)
	Dopp_var.append(NP.roll(NP.roll(phi_xyt,int(xinds),axis=0),int(yinds),axis=1))

	for jj in range(len(percentiles) -1 ):
		if sizes[ii] >= percentiles[jj] and sizes[ii] < percentiles[jj+1]:
			DoppAvg[jj] += NP.roll(NP.roll(phi_xyt,int(xinds),axis=0),int(yinds),axis=1)
			Ncount[jj] += 1


# don't forget the denominator in the mean
DoppAvg = DoppAvg/Ncount[:,None,None]
DoppAvg_amps = DoppAvg_amps/Ncount_amps[:,None,None]

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------


# Various plotting for checks
cart_MC.computeRealSpace()
xgrid   = cart_MC.xgrid_[::subSample]*1e-6
ygrid   = cart_MC.ygrid_[::subSample]*1e-6
xgrid = xgrid - (xgrid[-1])/2
ygrid = ygrid - (ygrid[-1])/2
xgrid,ygrid = NP.meshgrid(xgrid,ygrid,indexing='ij')

fig,ax = plt.subplots(2,len(percentiles)//2+1)
ax = ax.ravel()
for ii in range(len(percentiles)):
	ax[ii].pcolormesh(xgrid,ygrid,DoppAvg[ii])
	ax[ii].set_xlim(-50,50)
	ax[ii].set_ylim(-50,50)


x_bins = NP.histogram(xgrid[:,0][xgrid[:,0]>=0],bins=75)[1]
absX = NP.sqrt(xgrid**2+ygrid**2)
ur_azi_mean = NP.zeros((len(DoppAvg),len(x_bins)-1));
ur_azi_std  = copy.copy(ur_azi_mean);

for binInd in range(len(x_bins)-1):
	inds                    = (absX > x_bins[binInd])*(absX < x_bins[binInd+1])
	for jj in range(len(DoppAvg)):
		ur_azi_mean   [jj,binInd]  = NP.nanmean(DoppAvg[jj]      [inds])
		ur_azi_std     [jj,binInd] = NP.nanstd (DoppAvg[jj]      [inds])/NP.sqrt(NP.sum(inds))


fig,ax = plt.subplots(2,len(percentiles)//2+1)
ax = ax.ravel()
for jj in range(len(ax)-1):
	ax[jj].axhline(y=0,color='k')
	ax[jj].errorbar(x_bins[:-1] + NP.diff(x_bins)[0]/2,ur_azi_mean[jj],yerr=ur_azi_std[jj])
	ax[jj].set_xlim([0,50])

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
# Save the data
NP.savez_compressed(DATADIR + '/avgSupergranuleData_Doppler.npz',\
						DopplerAverage = DoppAvg,\
						DopplerAverage_amps = DoppAvg_amps,\
						DopplerStd = NP.std(Dopp_var,axis=0)/NP.sqrt(len(Dopp_var)),\
						Ncount = Ncount,\
						Ncount_amps = Ncount_amps,\
						percentiles = percentLimits,\
						percentiles_amps = percentiles_amps,\
						xgrid = xgrid,\
						ygrid = ygrid,\
						absX_bins = x_bins,\
						ur_azi_mean = ur_azi_mean,\
						ur_azi_std = ur_azi_std)