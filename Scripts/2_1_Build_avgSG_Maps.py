#!/scratch/ch3246/project/PythonEnv/miniconda3/envs/dalma-python3-CPU/bin/python


# Routine to construct the average supergranule from the LCT data

import sys
sys.path.insert(0,pathToRoutines)
print("Importing pythonRoutines")
import numpy as NP
from pyCompHelio import *
from matplotlib.pyplot import *
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel as astropyGaussian2D
from astropy.convolution import Box2DKernel      as astropyBox2D
from astropy.convolution import convolve as astropyConv
from skimage.feature import blob_dog, blob_log, blob_doh

plt.ion()

plt.close('all')

# for plotting and verbose output
TEST = True

# want to build outflow or inflow
OUTFLOWS = True
#-------------------------------------------------

# padding for real space construct
nPad = 70


Ncount = [];Ncount_size = [];sizes_total = []

POW = []

with NP.load(DATADIR + '/LCT_maps.npz') as npyDict:

	# Load in the LCT data
	Flow_Maps_x = NP.array(npyDict['VX']).T
	Flow_Maps_y = NP.array(npyDict['VY']).T
	xgridLCT     = NP.array(npyDict['xgrid'])
	ygridLCT     = NP.array(npyDict['ygrid'])

	# initialize FDMs for derivatives 
	dx = FDM_Compact(xgridLCT)
	dy = FDM_Compact(ygridLCT)

	# initialize grids
	xgridLCT,ygridLCT = NP.meshgrid(xgridLCT,ygridLCT,indexing='ij')


	# compute the wavenumber grid
	dkt = NP.fft.fftshift(NP.fft.fftfreq(len(xgridLCT),NP.diff(xgridLCT[:,0])[0])*2*NP.pi)
	abskt = NP.sqrt(dkt[:,None]**2 + dkt[None,:]**2)

	# filter to remove high ell information. Helps identify features
	FILT = PsGaussian(abskt*RSUN,100,25)

	# Filter the flow maps
	Flow_Maps_x = fft.ifftn(fft.ifftshift(fft.fftshift(fft.fftn(Flow_Maps_x))*FILT)).real
	Flow_Maps_y = fft.ifftn(fft.ifftshift(fft.fftshift(fft.fftn(Flow_Maps_y))*FILT)).real

	# Compute the horizontal divergence for feature ID
	Flow_MapsLCT = dx.Compute_derivative(Flow_Maps_x,axis=0) + dy.Compute_derivative(Flow_Maps_y,axis=1)
	
	# Apply the same apodizer that is used in Doppler FFT
	Flow_Maps = Flow_MapsLCT * fits.open('Apodize.fits')[0].data[::4,::4]

	# Return grid to Mm
	xgrid = xgridLCT*1e-6 
	ygrid = ygridLCT*1e-6

		
# Noramilze maps for feature ID
Flow_Maps_tmp = (Flow_Maps  - NP.mean(Flow_Maps))/ NP.std(Flow_Maps)
peak_map = detect_peaks(Flow_Maps_tmp,lower_limit = 2.)
peak_plot = NP.where(peak_map,1.,NP.nan)

# Find the features using blob_log
if OUTFLOWS:
	blobs_log = blob_log(Flow_Maps_tmp, min_sigma=3, max_sigma=15, num_sigma=1000, threshold=1)
else:
	blobs_log = blob_log(-Flow_Maps_tmp, min_sigma=3, max_sigma=15, num_sigma=1000, threshold=1)

blobs_log[:, 2] = blobs_log[:, 2] * NP.sqrt(2) ### Compute radii in the 3rd column. (first two columns contain the x,y coordinates)


# Test plots
if TEST:
	plt.figure()
	plt.pcolormesh(xgrid,ygrid,Flow_Maps*1e5,cmap='coolwarm',vmin=-4,vmax=4,rasterized=True)
	cbar = plt.colorbar()
	cbar.set_label(r'$10^5\times\nabla_h\cdot\mathbf{u}$ [1/s]',fontsize=15)
	plt.xlabel(r'$x$ [Mm]',fontsize=15)
	plt.ylabel(r'$y$ [Mm]',fontsize=15)
	# plt.plot((xgrid*peak_plot).ravel(),(ygrid*peak_plot).ravel(),'xw',mew=3)


# Determine the x and y values [in Mm] of the ID'd features

#ID using peak amps
peak_inds = peak_map.ravel()
xinds,yinds = NP.where(peak_map)
xvals = xgrid.ravel()[peak_inds]
yvals = ygrid.ravel()[peak_inds]

# ID using sizes
xinds_size,yinds_size = blobs_log[:,:2].astype(int).T
xvals_size = xgrid[:,0][blobs_log[:,0].astype(int)]
yvals_size = ygrid[0,:][blobs_log[:,1].astype(int)]
sizes = blobs_log[:, 2]*NP.diff(xgrid[:,0])[0]

#---------------------------------------------------------------------

# Filter for appropriate features
dellist = [];amplitudes = []
for jj in range(len(xinds)):
	suitability = True
	current_amp = Flow_Maps[xinds[jj],yinds[jj]]
	other_xinds = NP.delete(xinds,jj)
	other_yinds = NP.delete(yinds,jj)
	dists_other = NP.sqrt((NP.delete(xvals,jj) - xvals[jj])**2 + (NP.delete(yvals,jj)-yvals[jj])**2)

	# Ignore features too close to edges
	if NP.sqrt(xvals[jj]**2 + yvals[jj]**2) > 170:
			suitability = False

	# Filter two close objects for strongest
	for kk in range(len(other_xinds)):
		if (Flow_Maps[other_xinds[kk],other_yinds[kk]] > current_amp) and (dists_other[kk] < 15):		
			suitability = False

	if not suitability:
		dellist.append(jj)
	else:
		amplitudes.append(current_amp)

xinds = NP.delete(xinds,dellist);yinds = NP.delete(yinds,dellist)
xvals = NP.delete(xvals,dellist);yvals = NP.delete(yvals,dellist)

#---------------------------------------------------------------------

# Filter using size population
dellist = [];
for jj in range(len(xinds_size)):
	suitability = True
	current_amp = Flow_Maps[xinds_size[jj],yinds_size[jj]]
	other_xinds = NP.delete(xinds_size,jj)
	other_yinds = NP.delete(yinds_size,jj)
	dists_other = NP.sqrt((NP.delete(xvals_size,jj) - xvals_size[jj])**2 + (NP.delete(yvals_size,jj)-yvals_size[jj])**2)

	# Remove if too close to edges
	if NP.sqrt(xvals_size[jj]**2 + yvals_size[jj]**2) > 170:
		suitability = False

	# for kk in range(len(other_xinds)):
	# 	if (Flow_Maps[other_xinds[kk],other_yinds[kk]] > current_amp) and (dists_other[kk] < 15):		
	# 		suitability = False

	if not suitability:
		dellist.append(jj)


xinds_size = NP.delete(xinds_size,dellist);yinds_size = NP.delete(yinds_size,dellist)
xvals_size = NP.delete(xvals_size,dellist);yvals_size = NP.delete(yvals_size,dellist)
sizes = NP.delete(sizes,dellist)

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# Plto locations on flow maps
if TEST:
	xinds_size_lct = copy.copy(xinds_size)
	yinds_size_lct = copy.copy(yinds_size)

	for ii in range(len(xinds_size)):
		if ii == 0:
			plt.plot(xgrid[xinds_size[ii],0],ygrid[0,yinds_size[ii]],'.w',mew=2,label='MCA')
		plt.plot(xgrid[xinds_size[ii],0],ygrid[0,yinds_size[ii]],'.w',mew=2)

	plt.tick_params(labelsize=12)
	plt.tight_layout()

# Build the average divergence maps

# using amplitudes
DIV_Cen_Maps = NP.zeros((len(xinds),Flow_Maps.shape[0],Flow_Maps.shape[1]));
Roll_inds_tmp = NP.zeros((len(xinds),2))
for jj in range(len(xinds)):
	DIV_Cen_Maps[jj] = NP.roll(NP.roll(Flow_Maps,len(xgrid)//2-xinds[jj],axis=0),len(ygrid)//2-yinds[jj],axis=1)
	Roll_inds_tmp[jj] = NP.array([len(xgrid)//2-xinds[jj],len(ygrid)//2-yinds[jj]])
Ncount.append(len(xinds))

# using sizes
DIV_Cen_Maps_size = NP.zeros((len(xinds_size),Flow_Maps.shape[0],Flow_Maps.shape[1]));
Roll_inds_tmp_size = NP.zeros((len(xinds_size),2))
for jj in range(len(xinds_size)):
	DIV_Cen_Maps_size[jj] = NP.roll(NP.roll(Flow_Maps,len(xgrid)//2-xinds_size[jj],axis=0),len(ygrid)//2-yinds_size[jj],axis=1)
	Roll_inds_tmp_size[jj] = NP.array([len(xgrid)//2-xinds_size[jj],len(ygrid)//2-yinds_size[jj]])
Ncount_size.append(len(xinds_size))

avg_maps = NP.sum(DIV_Cen_Maps_size,axis=0)


if TEST:
	fig,ax = plt.subplots()
	ax.pcolormesh(xgrid,ygrid,avg_maps,cmap='coolwarm')
	ax.set_aspect(1)
	plt.tight_layout()


# Save
NP.savez_compressed(DATADIR + '/avgSupergranuleData_SurfaceFlows.npz',\
						avg_Flow_maps = DIV_Cen_Maps,\
						Roll_inds = Roll_inds_tmp,\
						avg_Flow_maps_size = DIV_Cen_Maps_size,\
						Roll_inds_size = Roll_inds_tmp_size,\
						xgrid = xgrid,ygrid = ygrid,\
						Ncount = len(xinds),\
						Ncount_size = len(xinds_size),\
						nPad = nPad,\
						FILT = FILT,\
						amplitudes = amplitudes,\
						sizes = sizes)

