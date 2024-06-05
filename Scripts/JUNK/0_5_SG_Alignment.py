#!/scratch/ch3246/project/PythonEnv/miniconda3/envs/dalma-python3-CPU/bin/python

pathToRoutines = '/home/ch3246/mps_montjoie/'
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

plt.close('all')

multiplyH=True
kRmax = 1000
kRmin = 200
WINDFACTOR = 2
modeSpacingFactor = 2

nu0 = 0.004362
# DATADIR = [];CRs = []
# for CR in NP.arange(2197,2204):
# 	DATADIR.append('/scratch/ch3246/OBSDATA/modeCouple/Cartesian/mTrack_modeCoupling_11d[%i][090][090.0][+00.0][+00.0]' % CR)
# 	CRs.append('CR%i_90deg' % CR)
# 	DATADIR.append('/scratch/ch3246/OBSDATA/modeCouple/Cartesian/mTrack_modeCoupling_11d[%i][270][270.0][+00.0][+00.0]' % CR)
# 	CRs.append('CR%i_270deg' % CR)

DATADIR = []
for CR in NP.arange(2197,2201):
	DATADIR.append('/scratch/ch3246/OBSDATA/modeCouple/Cartesian/mTrack_modeCoupling_1d[%i][090][090.0][+00.0][+00.0]/' % CR)
#-----------------FIG 2--------------------------


Bcoeff_grid_summed = []
for iFile in [0]:#range(len(DATADIR)):
#-----------------FIG 2--------------------------
	cart_MC = cartesian_modeCoupling(DATADIR[iFile] + 'V.fits',\
										dxFactor=4,dyFactor=4,kMax=1500/RSUN,OmegaMax=0.0055*2*NP.pi,\
										timeSubSample=1)
	phi_kw = cart_MC.computeFFT(storeInInstance=True,fitsPathLoad = DATADIR[iFile] + 'V_kw.fits')

	kx,ky,omega = cart_MC.computeFreq()
	kxg,kyg = NP.meshgrid(kx,ky,indexing='ij')
	abskg = NP.sqrt(kxg**2+kyg**2)
	absk = NP.sqrt(kx[kx>=0]**2+ky[ky>=0]**2)


	Bcoeff_grid = []


	QX = NP.arange(-15,16)
	QY = NP.arange(-15,16)
	SIGMA = NP.array([0])

	# QX = NP.arange(-20,21)
	# QY = NP.array([0])
	# SIGMA = NP.array([3])


	Ngrid = NP.array([3])
	for nn in Ngrid:
		omegaM,gamma,amp = cart_MC.ref_fit_params(absk,nn)
		indl = NP.argmin(abs(NP.nan_to_num(omegaM) / (2*NP.pi)-0.0015))
		indr = NP.argmin(abs(NP.nan_to_num(omegaM) / (2*NP.pi)-0.005))

		mask = NP.where((abskg*RSUN < min(absk[indr]*RSUN,kRmax))*(abskg*RSUN > max(absk[indl]*RSUN,kRmin)),1,0)
		# mask[::2] = 0;mask[:,::2] = 0
		# mask,omega_mask = cart_MC.mask_cube(nn,'all',0,kRmin = 200,kRmax=1000,nu_min = 0.002,nu_max = 0.004)
		# cart_MC.test_plots(radial_order = nn,mask=mask,nu = 0.003,vmaxScale=0.1,lineWidthFactor=2)
		# abort

		
		inds_kx,inds_ky = NP.where(mask)

		amps_avg = cart_MC.compute_N_nk(mask,nn,25,num_linewidths=2,PLOT = False)

		# inds_kx = NP.array([-50,50])
		# inds_ky = NP.array([100,100])
		# mask = NP.zeros(mask.shape)
		# mask[:,66] = 1
		# inds_kx,inds_ky = NP.where(mask)
		# mask_og = copy.copy(mask)
		# mask = (NP.sum(mask,axis=-1)!=0).astype(int)
		# mask[::2] = 0
		# mask[:,::2] = 0

		# omega_inds = cart_MC.compute_bcoeff_serial(nn,inds_kx[0],inds_ky[0],QX,QY,SIGMA,nu_min=0.0025,nu_max=0.0045,\
									# absq_range=NP.array([0,1e9])/RSUN,VERBOSE=False,returnNoise=False,\
									# windfactor = WINDFACTOR,TEST=1)
		# plt.plot(omega/(2*NP.pi)*1e6,abs(phi_kw[:,inds_ky[0],inds_kx[0]])**2)
		# plt.plot(omega[omega_inds]/(2*NP.pi)*1e6,abs(phi_kw[omega_inds,inds_ky[0],inds_kx[0]])**2)
		# plt.xlim([2000,5000])
		# omega_inds = cart_MC.compute_bcoeff_serial(nn,inds_kx[0],inds_ky[0],NP.array([5]),NP.array([5]),SIGMA,amps_avg,nu_min=0.002,nu_max=0.0055,\
									# absq_range=NP.array([0,300])/RSUN,VERBOSE=True,returnNoise=False,\
									# windfactor = WINDFACTOR,TEST=1)
		# abort
		# absk = NP.sqrt(kx[inds_kx[0]]**2 + ky[inds_ky[0]]**2)
		kxm,kym,tmp = cart_MC.compute_bcoeffs_parallel(nn,mask,QX,QY,SIGMA,amps_avg,nu_min=0.002,nu_max=0.0055,absq_range=NP.array([0,600])/RSUN,\
														VERBOSE=True,nbProc=27,reorg_k=False,returnNoise=False,\
														windfactor=WINDFACTOR,rtype=None)
		# abort
		# tmp2 = cart_MC.compute_bcoeffs_parallel(nn,mask,NP.array([0]),NP.array([0]),NP.array([0]),nu_min=0.002,nu_max=0.005,absq_range=NP.array([0,1e9])/RSUN,VERBOSE=True,nbProc=8)
		Bcoeff_grid.append(tmp)
	if len(Ngrid) > 1:
		Bcoeff_grid = NP.concatenate(Bcoeff_grid,axis=-1)
	else:
		Bcoeff_grid = NP.array(Bcoeff_grid)

	# NP.savez_compressed(DATADIR[iFile] + '/Bcoeff_Prasad.npz',\
	# 					QX = QX,QY = QY,SIGMA = SIGMA,\
	# 					dkx = NP.diff(kx)[0],dky = NP.diff(ky)[0],domega = NP.diff(omega)[0],\
	# 					Ngrid = Ngrid,\
	# 					Bcoeffs = Bcoeff_grid)
	# abort
	# Bcoeff_grid_summed = []
	# for ii in range(len(Bcoeff_grid)):
	# 	Bcoeff_grid_summed.append(NP.sum(Bcoeff_grid[ii],axis=-1))
	Bcoeff_grid_summed.append(NP.nansum(Bcoeff_grid,axis=-1))

Bcoeff_grid_summed = NP.array(Bcoeff_grid_summed)
dkx = NP.diff(kx)[0];dky = NP.diff(ky)[0];domega = NP.diff(omega)[0]

plt.figure()
plt.pcolormesh((QX-0.5)*NP.diff(kx)[0]*RSUN,(QY-0.5)*NP.diff(kx)[0]*RSUN,NP.nanmean(abs(Bcoeff_grid_summed)**2,axis=0).squeeze().T)
plt.axvline(x=0,color='r')
plt.axhline(y=0,color='r')
plt.xlabel(r'$k_xR_\odot$')
plt.ylabel(r'$k_yR_\odot$')
plt.colorbar()


tmp = Bcoeff_grid_summed.squeeze()

xgridI = NP.arange(-phi_kw.shape[-1]/2,phi_kw.shape[-1]/2)*0.04/180*RSUN*NP.pi /1e6


qxgrid,qygrid = NP.meshgrid(QX*dkx,QY*dky,indexing='ij')
xgrid,ygrid = NP.meshgrid(xgridI,xgridI,indexing='ij')

polMap = NP.sum(tmp[:,:,None,None] * NP.exp(1.j* (qxgrid[:,:,None,None]*xgrid[None,None,:,:]*1e6+qygrid[:,:,None,None]*ygrid[None,None,:,:]*1e6)),axis=(0,1))
polMap = fft.fftshift(polMap).T