#!/scratch/ch3246/project/PythonEnv/miniconda3/envs/dalma-python3-CPU/bin/python


# Build the average surface flow maps from the LCT data
import sys
sys.path.insert(0,pathToRoutines)
print("Importing pythonRoutines")
import numpy as NP
from pyCompHelio import *
from matplotlib.pyplot import *
from astropy.io import fits

plt.ion()

plt.close('all')

SORTsize = True

LCT = True
OUTFLOWS = True
#-------------------------------------------------
# Build the population groups

percentLimits     = NP.arange(0,101,10)#[0,20,40,60,80,100]
if SORTsize:
	Sizes_all    = NP.genfromtxt('data/SG_size.dat')
	percentiles       = NP.percentile(Sizes_all,percentLimits)
else:	
	Amplitudes_all    = NP.genfromtxt('data/SG_Amplitudes.dat')
	percentiles       = NP.percentile(Amplitudes_all,percentLimits)

# Load in the location data, should be in loop of more then one
with NP.load(DATADIR + '/avgSupergranuleData_SurfaceFlows.npz') as nypdict:		
	xgrid_tmp  = nypdict['xgrid']
	ygrid_tmp  = nypdict['ygrid']
	Ncount_tmp = nypdict['Ncount']
	if SORTsize:
		sortQuant = nypdict['sizes']
		Roll_inds  = nypdict['Roll_inds_size']
	else:
		sortQuant = nypdict['amplitudes']
		Roll_inds  = nypdict['Roll_inds']

# Initialize the array for average maps
uhAvg = NP.zeros((len(percentiles),2,len(xgrid_tmp),len(ygrid_tmp)))
uh_var = []

# initialize the class for various grid calculations
cart_MC = cartesian_modeCoupling(DATADIR + '/V.fits',\
									apodization_fitsPath = os.getcwd() + '/Apodize.fits',\
									dxFactor=2,dyFactor=2,kMax=5000/RSUN,OmegaMax=0.006*2*NP.pi,\
									timeSubSample=1)

if NP.round(cart_MC.Nx_/len(xgrid_tmp)) != 2:
	raise Exception('Error: we are assuming Nx = 2*len(xgrid_tmp)')
else:
	subSample = 2

# Load in the LCT maps
with NP.load(DATADIR + '/LCT_maps.npz') as npyDICT:
	VX = npyDICT['VX'].T
	VY = npyDICT['VY'].T


# Build the average flows [UX, UY], for each population group
for ii in range(len(Roll_inds)):
	xinds,yinds = Roll_inds[ii]
	uhAvg[-1,0] += NP.roll(NP.roll(VX,int(xinds),axis=0),int(yinds),axis=1)
	uhAvg[-1,1] += NP.roll(NP.roll(VY,int(xinds),axis=0),int(yinds),axis=1)
	uh_var.append([NP.roll(NP.roll(VX,int(xinds),axis=0),int(yinds),axis=1),NP.roll(NP.roll(VY,int(xinds),axis=0),int(yinds),axis=1)])
	Ncount[-1] += 1
	for jj in range(len(percentiles) -1 ):
		if sortQuant[ii] >= percentiles[jj] and sortQuant[ii] < percentiles[jj+1]:
			uhAvg[jj,0] += NP.roll(NP.roll(VX,int(xinds),axis=0),int(yinds),axis=1)
			uhAvg[jj,1] += NP.roll(NP.roll(VY,int(xinds),axis=0),int(yinds),axis=1)				
			Ncount[jj] += 1
uh_var = NP.array(uh_var)

# Build some grids
cart_MC.computeRealSpace()
xgrid   = cart_MC.xgrid_[::subSample]*1e-6
ygrid   = cart_MC.ygrid_[::subSample]*1e-6
xgrid = xgrid - (xgrid[-1])/2
ygrid = ygrid - (ygrid[-1])/2

# Initialize the FDM matrices
dx = FDM_Compact(xgrid*1e6)
dy = FDM_Compact(ygrid*1e6)




# Taking care of denominator in mean and removing some systematics
uhAvg = uhAvg/Ncount[:,None,None,None]
uhAvg = uhAvg - NP.mean(uhAvg[:,:,150:,:],axis=2)[:,:,None,:] - NP.mean(uhAvg[:,:,:,150:],axis=3)[:,:,:,None]

# Compute the horizontal divergence
divAvg = dx.Compute_derivative(uhAvg[:,0],axis=1) + dy.Compute_derivative(uhAvg[:,1],axis=2)

# build some grids
xgrid,ygrid = NP.meshgrid(xgrid,ygrid,indexing='ij')

# Plot average horizontal flows
for comp in range(2):
	fig,ax = plt.subplots(2,len(percentiles)//2 +1)
	ax = ax.ravel()
	for ii in range(len(percentiles)):
		ax[ii].pcolormesh(xgrid,ygrid,uhAvg[ii,comp])
		ax[ii].set_xlim(-50,50)
		ax[ii].set_ylim(-50,50)


# COmpute horizontal flows
unit_h = (NP.array([xgrid,ygrid])/NP.sqrt(NP.sum(NP.abs([xgrid,ygrid])**2,axis=0)))
unit_h = NP.where(NP.sqrt(NP.sum(NP.abs([xgrid,ygrid])**2,axis=0))==0,0,unit_h)
uh = NP.sum(unit_h[None,...] * uhAvg,axis=1)


#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
# Plot the horizontal flow
fig,ax = plt.subplots(2,len(percentiles)//2+1)
ax = ax.ravel()
for ii in range(len(percentiles)):
	im = ax[ii].pcolormesh(xgrid,ygrid,uh[ii],vmax = 1.2*NP.amax(abs(uh[-1])),vmin = -1.2*NP.amax(abs(uh[-1])),cmap='jet')
	ax[ii].set_xlim(-50,50)
	ax[ii].set_ylim(-50,50)
	if ii not in [0,3]:
		ax[ii].set_yticks([])

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

fig.suptitle(r'horizontal flow')


x_bins = NP.histogram(xgrid[:,0][xgrid[:,0]>=0],bins=75)[1]
absX = NP.sqrt(xgrid**2+ygrid**2)

ur_azi_mean = NP.zeros((len(divAvg),len(x_bins)-1));
ur_azi_std  = copy.copy(ur_azi_mean);

for binInd in range(len(x_bins)-1):
	inds                    = (absX > x_bins[binInd])*(absX < x_bins[binInd+1])
	for jj in range(len(divAvg)):
		ur_azi_mean   [jj,binInd]  = NP.nanmean(divAvg[jj]      [inds])
		ur_azi_std     [jj,binInd] = NP.nanstd (divAvg[jj]      [inds])/NP.sqrt(NP.sum(inds))


fig,ax = plt.subplots(2,len(percentiles)//2+1)
ax = ax.ravel()
for jj in range(len(percentiles)):
	ax[jj].axhline(y=0,color='k')
	ax[jj].errorbar(x_bins[:-1] + NP.diff(x_bins)[0]/2,ur_azi_mean[jj],yerr=ur_azi_std[jj])
	ax[jj].set_xlim([0,50])
	ax[jj].axvline(x=11,color='k')
	ax[jj].axvline(x=18,color='k')

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# Save the data
NP.savez_compressed(DATADIR + '/avgSupergranuleData_LCT.npz',\
						LCTAverage = uhAvg,\
						divAverage = divAvg,\
						LCT_var = NP.std(uh_var,axis=0)/NP.sqrt(len(uh_var)),\
						uhAverage = uh,\
						Ncount = Ncount,\
						percentiles = percentLimits,\
						xgrid = xgrid,\
						ygrid = ygrid,\
						absX_bins = x_bins)
