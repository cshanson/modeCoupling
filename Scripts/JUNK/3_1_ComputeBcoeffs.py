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

nn = int(sys.argv[1])
iFileMin = int(sys.argv[2])
iFileMax = int(sys.argv[3])

FILTER = False
SORTsize = True

# nn = 0;
# iFileMax = 6
# iFileMin = 5

nnp = nn
subDir = '1Day'
InversionData = NP.genfromtxt('data/InversionDataDir.txt',dtype=str)
percentLimits     = [0,20,40,60,80,100]

if SORTsize:
	Sizes_all    = NP.genfromtxt('data/SG_size.dat')
	percentiles       = NP.percentile(Sizes_all,percentLimits)
else:	
	Amplitudes_all    = NP.genfromtxt('data/SG_Amplitudes.dat')
	percentiles       = NP.percentile(Amplitudes_all,percentLimits)
TEST = False
OVERWRITE = True


calcDetails = NP.genfromtxt('data/Bcoeff_params.dat')
calcDetails = calcDetails[(calcDetails[:,0] == nn)*(calcDetails[:,1]==nnp)].squeeze()
kRmax = calcDetails[3]
kRmin = calcDetails[2]
WINDFACTOR = 2
modeSpacingFactor = None

QX = NP.arange(-30,31)
QY = NP.arange(-30,31)
SIGMA = NP.array([0])


tini = time.time()

FirstRun = False
for iFile in range(iFileMin,iFileMax):
	print('Computing for %s' % (InversionData[iFile][0].split('/')[-1].split('g')[-1]))
	
	


	DATADIR = InversionData[iFile][0]
	tindl   = int(InversionData[iFile][1])
	tindr   = int(InversionData[iFile][2])
	
	print(iFile,DATADIR)
	with NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/InversionData/avgSupergranuleData_SurfaceFlows%s%s.npz' % (DATADIR.split('/')[-1].split('g')[-1],['','_FILT%i' % int(FILTER)][int(FILTER is not False)]),allow_pickle=True) as DICT:
		if SORTsize:
			avg_Flow_maps,Roll_inds_iFile,Ncount,nPad,xgrid,ygrid,FILT,sizes = [DICT[x] for x in ['avg_Flow_maps_size','Roll_inds_size','Ncount_size','nPad','xgrid','ygrid','FILTQ','sizes']]
		else:
			avg_Flow_maps,Roll_inds_iFile,Ncount,nPad,xgrid,ygrid,FILT,amplitudes = [DICT[x] for x in ['avg_Flow_maps','Roll_inds','Ncount','nPad','xgrid','ygrid','FILTQ','amplitudes']]

	if os.path.isfile(DATADIR + '/%s/Bcoeffs/Bcoeffs_n%i_np%i_%i_%i%s%s.h5' % (subDir,nn,nnp,tindl,tindr,DATADIR.split('/')[-1].split('g')[-1],['','_FILT%i' % int(FILTER)][int(FILTER is not False)])) and not OVERWRITE:
		print('Bcoeffs already computed, will not OVERWRITE')
		continue		

	Bcoeff_grid_summed = []; Flow_Maps = []
	for tind in [1]:#range(len(tinds)-1):

		Roll_inds_iFile_iTime = Roll_inds_iFile.astype(int)
		

		cart_MC = cartesian_modeCoupling(DATADIR + '/V.fits',\
											apodization_fitsPath = os.getcwd() + '/Apodize.fits',\
											dxFactor=2,dyFactor=2,kMax=2000/RSUN,OmegaMax=0.006*2*NP.pi,\
											timeSubSample=1,timeInds = [tindl,tindr])
		phi_kw = cart_MC.computeFFT(storeInInstance=True,fitsPathLoad = DATADIR + '/%s/V_kw_%i_%i.fits' % (subDir,cart_MC.timeInds_[0],cart_MC.timeInds_[1]))

		kx,ky,omega = cart_MC.computeFreq()
		dkx = NP.diff(kx)[0];dky = NP.diff(ky)[0];dw = NP.diff(omega)[0]
		kxg,kyg = NP.meshgrid(kx,ky,indexing='ij')
		abskg = NP.sqrt(kxg**2+kyg**2)
		absk = NP.sqrt(kx[kx>=0]**2+ky[ky>=0]**2)

		omegaM,gamma,amp = cart_MC.ref_fit_params(absk,nn)
		indl = NP.argmin(abs(NP.nan_to_num(omegaM) / (2*NP.pi)-0.002))
		indr = NP.argmin(abs(NP.nan_to_num(omegaM) / (2*NP.pi)-0.005))

		mask = NP.where((abskg*RSUN < min(absk[indr]*RSUN,kRmax))*(abskg*RSUN > max(absk[indl]*RSUN,kRmin)),1,0)
		if TEST:
			mask[::2,::2] = 0
		
		inds_kx,inds_ky = NP.where(mask)

		amps_avg  = cart_MC.compute_N_nk(mask,nn,30,num_linewidths=2,PLOT = False)
		amps_avgp = cart_MC.compute_N_nk(mask,nnp,30,num_linewidths=2,PLOT = False)

		print('\tComputing B coefficients')
		kxm,kym,Bcoeff_grid = cart_MC.compute_bcoeffs_parallel([nn,nnp],mask,QX,QY,SIGMA,[amps_avg,amps_avgp],nu_min=0.0015,nu_max=0.0055,absq_range=NP.array([0,300])/RSUN,\
														VERBOSE=True,nbProc=27,reorg_k=False,returnNoise=False,\
														windfactor=WINDFACTOR,modeSpacingFactor = modeSpacingFactor,rtype=None)

		
		if not FirstRun:
			print('\tComputing Noise Model')
			kxmN,kymN,Noise_grid = cart_MC.compute_bcoeffs_parallel([nn,nnp],mask,QX,QY,SIGMA,[amps_avg,amps_avgp],nu_min=0.0015,nu_max=0.0055,absq_range=NP.array([0,300])/RSUN,\
																VERBOSE=True,nbProc=27,reorg_k=False,returnNoise='theory',\
																windfactor=WINDFACTOR,modeSpacingFactor=modeSpacingFactor,rtype=None)
		

			good_inds = NP.sum(Bcoeff_grid,axis=(0,1,2)) != 0
			Noise_grid  = Noise_grid[...,good_inds]
			FirstRun = True

		inds_kx = inds_kx[good_inds];inds_ky = inds_ky[good_inds]
		kxm = kxm[good_inds];kym = kym[good_inds]
		Bcoeff_grid = Bcoeff_grid[...,good_inds]

		Flow_Maps_avgSG = 0
		Btmp = NP.sum(Bcoeff_grid,axis=-1).squeeze()
		Flow_Map = fft.ifftn(fft.ifftshift(NP.pad(Btmp,((nPad,nPad),(nPad,nPad)),constant_values=0)),norm='ortho').real
		for jj in range(len(Roll_inds_iFile_iTime)):
			Flow_Maps_avgSG += NP.roll(NP.roll(Flow_Map,Roll_inds_iFile_iTime[jj,0],axis=0),Roll_inds_iFile_iTime[jj,1],axis=1)
		# Flow_Maps_avgSG = Flow_Maps_avgSG/len(Roll_inds_iFile_iTime)
		


		if TEST:
			Flow_Maps_avgSGTEST = []
			for jj in range(len(Roll_inds_iFile_iTime)):
				Flow_Maps_avgSGTEST .append(NP.roll(NP.roll(Flow_Map,Roll_inds_iFile_iTime[jj,0],axis=0),Roll_inds_iFile_iTime[jj,1],axis=1))
			# Flow_Maps_avgSGTEST = Flow_Maps_avgSGTEST/len(Roll_inds_iFile_iTime)

			POW  = NP.abs(NP.sum(Bcoeff_grid,axis=-1)).squeeze()**2
			POWn = NP.abs(NP.sum(Noise_grid,axis=-1)).squeeze()
			POWavg = abs(NP.fft.fftshift(NP.fft.fftn(NP.nanmean(Flow_Maps_avgSGTEST,axis=0),axes = (0,1),norm='ortho'),axes= (0,1))[nPad:len(xgrid)-nPad,nPad:len(ygrid)-nPad]).squeeze()**2
			# POWavg2 = abs(NP.fft.fftshift(NP.fft.fftn(Flow_Maps_avgSG,axes = (0,1),norm='ortho'),axes= (0,1))[nPad:len(xgrid)-nPad,nPad:len(ygrid)-nPad]).squeeze()**2

			ABSQ = NP.sqrt((QX[:,None]*dkx*RSUN)**2+(QY[None,:]*dky*RSUN)**2)
			ABSQ = ABSQ.ravel()

			absk_bins = NP.histogram(ABSQ,bins=25)[1]
			absk_bins = absk_bins[NP.argmin(abs(absk_bins-15)):NP.argmin(abs(absk_bins-415))]

			power_nn = NP.zeros((len(absk_bins)-1));
			power_noise = NP.zeros((len(absk_bins)-1))
			power_avg = NP.zeros((len(absk_bins)-1));
			# power_avg2 = NP.zeros((len(absk_bins)-1));

			DAT       = POW.ravel();DATavg = POWavg.ravel()
			DATN      = POWn.ravel();#DATavg2 = POWavg2.ravel()
			for binInd in range(len(absk_bins)-1):
				inds = (ABSQ > absk_bins[binInd])*(ABSQ < absk_bins[binInd+1])
				power_nn    [binInd] = NP.nanmean(DAT       [inds])
				power_noise [binInd] = NP.nanmean(DATN      [inds])
				power_avg   [binInd] = NP.nanmean(DATavg    [inds])
				# power_avg2  [binInd] = NP.nanmean(DATavg2   [inds])

			plt.figure()
			plt.plot(absk_bins[:-1] + NP.diff(absk_bins)/2,power_nn)
			plt.plot(absk_bins[:-1] + NP.diff(absk_bins)/2,power_noise,'--',color='b')
			plt.plot(absk_bins[:-1] + NP.diff(absk_bins)/2,power_avg,color='r')
			# plt.plot(absk_bins[:-1] + NP.diff(absk_bins)/2,power_avg2,'.',color='k')

			plt.xlabel(r'$qR_\odot$')

		

			plt.figure()
			vmax = NP.amax(Flow_Maps_avgSG)
			plt.pcolormesh(xgrid,ygrid,Flow_Maps_avgSG.squeeze(),cmap='jet',vmax=vmax,vmin=-vmax)
			circle2 = plt.Circle((0, 0), 11, color='k', fill=False)
			plt.gca().add_patch(circle2)
			circle2 = plt.Circle((0, 0), 18, color='k', fill=False)
			plt.gca().add_patch(circle2)
			plt.xlim(-45,45)
			plt.ylim(-45,45)
			plt.xlabel(r'$x$ [Mm]')
			plt.ylabel(r'$y$ [Mm]')

		if FILTER is not False:
			Bcoeff_grid = Bcoeff_grid * FILT[:,:,None,None]
			if FirstRun:
				Noise_grid = Noise_grid * FILT[:,:,None,None]

			
		

		Bcoeff_avgSG = NP.zeros(Bcoeff_grid.shape + (len(percentiles),),complex)
		NSGs = NP.zeros(len(percentiles))
		qxgrid,qygrid = NP.meshgrid(QX*dkx,QY*dky,indexing='ij')
		dx = NP.diff(xgrid[:,0])[0]*1e6;dy = NP.diff(ygrid[0,:])[0]*1e6
		for jj in range(len(Roll_inds_iFile_iTime)):
			tmp = Bcoeff_grid * NP.exp(-1.j*(qxgrid * dx*Roll_inds_iFile_iTime[jj,0] + qygrid * dy*Roll_inds_iFile_iTime[jj,1]))[:,:,None,None]
			Bcoeff_avgSG[...,-1] += tmp
			NSGs[-1] += 1
			for kk in range(len(percentiles) -1 ):
				if SORTsize:
					if sizes[jj] > percentiles[kk] and sizes[jj] < percentiles[kk+1]:
						Bcoeff_avgSG[...,kk] += tmp
						NSGs[kk] += 1
				else:
					if amplitudes[jj] > percentiles[kk] and amplitudes[jj] < percentiles[kk+1]:
						Bcoeff_avgSG[...,kk] += tmp
						NSGs[kk] += 1


		if TEST:
			fig,ax = plt.subplots(2,len(percentiles)//2)
			ax = ax.ravel()
			for kk in range(len(percentiles)):
				Btmp2 = NP.sum(Bcoeff_avgSG,axis=-2).squeeze()[...,kk]
				Flow_Maps_avgSG2 = fft.ifftn(fft.ifftshift(NP.pad(Btmp2,((nPad,nPad),(nPad,nPad)),constant_values=0)),norm='ortho').real

				ax[kk].pcolormesh(xgrid,ygrid,Flow_Maps_avgSG2.T)
				ax[kk].set_ylim([-50,50])
				ax[kk].set_xlim([-50,50])
				# plt.figure()
				# plt.plot(ygrid[0,:],Flow_Maps_avgSG[100],'b',lw=2,label = 'Avg SG from NP.roll')
				# plt.plot(ygrid[0,:],Flow_Maps_avgSG2[100],'r',lw=1,label = 'Avg SG from fft translation')
				# plt.legend()
			abort


			


		# Bcoeff_avgSG = 0.
		# BmatPad = fft.ifftshift(NP.pad(Bcoeff_grid,((nPad,nPad),(nPad,nPad),(0,0),(0,0)),constant_values=0),axes=(0,1))

		# if 1:
		# 	aa1 = pyfftw.empty_aligned(BmatPad.shape, dtype='complex128')
		# 	aa2 = pyfftw.empty_aligned(BmatPad.shape, dtype='complex128')
		# 	fft_object_a = pyfftw.FFTW(aa1, aa2,axes=(0,1), ortho=True,normalise_idft=False,direction='FFTW_BACKWARD')
		# 	bb1 = pyfftw.empty_aligned(BmatPad.shape, dtype='complex128')
		# 	bb2 = pyfftw.empty_aligned(BmatPad.shape, dtype='complex128')
		# 	fft_object_b = pyfftw.FFTW(bb1, bb2,axes=(0,1), ortho=True,normalise_idft=False,direction='FFTW_FORWARD')

		# tini = time.time()
		# polMap = fft_object_a(BmatPad)
		# del aa1,aa2,fft_object_a
		# PG = progressBar(len(Roll_inds_iFile_iTime),'serial')
		# for jj in range(len(Roll_inds_iFile_iTime)):
		# 	polMap_c = NP.roll(NP.roll(polMap,Roll_inds_iFile_iTime[jj,0],axis=0),Roll_inds_iFile_iTime[jj,1],axis=1)
		# 	Bcoeff_avgSG += NP.fft.fftshift(fft_object_b(polMap_c))[nPad:len(xgrid)-nPad,nPad:len(ygrid)-nPad]
		# 	PG.update()
		# del PG
		# print(time.time() - tini)
		# # abort
		# del bb1,bb2,fft_object_b
		# Bcoeff_avgSG2 = 0.
		# print('\tCentering the maps')
		# tini=time.time()
		# polMap = fft.ifftn(fft.ifftshift(NP.pad(Bcoeff_grid,((nPad,nPad),(nPad,nPad),(0,0),(0,0)),constant_values=0),axes=(0,1)),axes=(0,1),norm='ortho')
		# PG = progressBar(len(Roll_inds_iFile_iTime),'serial')
		# for jj in range(3):#len(Roll_inds_iFile_iTime)):
		# 	polMap_c = NP.roll(NP.roll(polMap,Roll_inds_iFile_iTime[jj,0],axis=0),Roll_inds_iFile_iTime[jj,1],axis=1)

		# 	Bcoeff_avgSG2 += NP.fft.fftshift(NP.fft.fftn(polMap_c,axes = (0,1),norm='ortho'),axes= (0,1))[nPad:len(xgrid)-nPad,nPad:len(ygrid)-nPad]
		# 	PG.update()
		# del PG
		# print(time.time() - tini)


		mkdir_p(DATADIR + '/%s/Bcoeffs/' % subDir)

		# NP.savez_compressed(DATADIR[iFile] + '/%s/Bcoeffs/Bcoeffs_n%i_np%i_%i_%i%s.npz' % (subDir,nn,nnp,tinds[tind],tinds[tind+1],DATADIR[iFile].split('/')[-2].split('g')[-1]),\
		# 					Bcoeffs_field = Bcoeff_grid,\
		# 					Bcoeffs_avgSG = Bcoeff_avgSG,\
		# 					Noise_grid = Noise_grid,\
		# 					Flow_Maps_avgSG = Flow_Maps_avgSG,nPad = nPad,\
		# 					xgrid = xgrid,ygrid = ygrid,\
		# 					QX = QX,QY = QY,SIGMA = SIGMA,\
		# 					dkx = dkx,dky = dky,dw = dw,\
		# 					radial_order = nn,kRmax = kRmax, kRmin = kRmin,\
		# 					inds_kx = inds_kx,inds_ky = inds_ky,\
		# 					kxm = kxm,kym = kym,mask = mask,\
		# 					WINDFACTOR = WINDFACTOR,modeSpacingFactor=modeSpacingFactor,\
		# 					NumberSGs = Ncount[iFile])


		with h5py.File(DATADIR + '/%s/Bcoeffs/Bcoeffs_n%i_np%i_%i_%i%s%s.h5' % (subDir,nn,nnp,tindl,tindr,DATADIR.split('/')[-1].split('g')[-1],['','_FILT%i' % int(FILTER)][int(FILTER is not False)]),'w') as h5f:
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

			h5f.create_dataset('inds_kx',data = inds_kx)
			h5f.create_dataset('inds_ky',data = inds_ky)

			h5f.create_dataset('nPad',data = nPad)
			h5f.create_dataset('WINDFACTOR',data = WINDFACTOR)
			# h5f.create_dataset('modeSpacingFactor',data = modeSpacingFactor)
			h5f.create_dataset('NumberSGs',data=NSGs)
			h5f.create_dataset('Percentiles',data=percentLimits)

			h5f.create_dataset('Flow_Maps_avgSG',data = Flow_Maps_avgSG)

			Bgrp = h5f.create_group('Bcoeffs_field')
			BAVGgrp = h5f.create_group('Bcoeffs_avgSG')
			Ngrp = h5f.create_group('NoiseModel')
			for ii in range(len(QX)):
				Bsubgrp    = Bgrp.create_group('QX%i' % QX[ii])
				BAVGsubgrp = BAVGgrp.create_group('QX%i' % QX[ii])
				Nsubgrp    = Ngrp.create_group('QX%i' % QX[ii])
				for jj in range(len(QY)):
					Bssubgrp    = Bsubgrp.create_group('QY%i' % QY[jj])
					BAVGssubgrp = BAVGsubgrp.create_group('QY%i' % QY[jj])
					Nssubgrp    = Nsubgrp.create_group('QY%i' % QY[jj])
					for kk in range(len(SIGMA)):
						Bsssubgrp    = Bssubgrp.create_group('SIGMA%i' % SIGMA[kk])
						BAVGsssubgrp = BAVGssubgrp.create_group('SIGMA%i' % SIGMA[kk])
						Nsssubgrp    = Nssubgrp.create_group('SIGMA%i' % SIGMA[kk])
						
						Bsssubgrp.create_dataset('data',data = Bcoeff_grid[ii,jj,kk])
						BAVGsssubgrp.create_dataset('data',data = Bcoeff_avgSG[ii,jj,kk])
						Nsssubgrp.create_dataset('data',data = Noise_grid[ii,jj,kk])


print(time.time() - tini,' seconds')