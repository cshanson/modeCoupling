#!/scratch/ch3246/project/PythonEnv/miniconda3/envs/dalma-python3-CPU/bin/python


# Routine to compute the Bcoefficients. Built with loop in mind to average the Bcoeffs as computed
import sys
sys.path.insert(0,pathToRoutines)
print("Importing pythonRoutines")
import numpy as NP
from pyCompHelio import *
from matplotlib.pyplot import *
from astropy.io import fits
plt.ion()

plt.close('all')


# Some inital information
nn = 0; # radial order n
nnshift = 0 # difference between n and the radial order to couple to 
iFileMax = 1 # max file number
iFileMin = 0 # min file number

# use size for location and population
SORTsize = True


# Do you want Bcoeffs for flow or soundspeed
Measurement = 'FLOW'
# Measurement = 'SOUNDSPEED'

# compute radial order prime
nnp = nn + nnshift


# define some suffixes for output file
if Measurement.upper() == 'SOUNDSPEED':
	fileNameSuffix = '_cs'
else:
	fileNameSuffix = ''


# Build populations
percentLimits     = NP.arange(0,101,10)
if SORTsize:
	Sizes_all    = NP.genfromtxt('data/SG_size.dat')
	percentiles       = NP.percentile(Sizes_all,percentLimits)
else:	
	Amplitudes_all    = NP.genfromtxt('data/SG_Amplitudes.dat')
	percentiles       = NP.percentile(Amplitudes_all,percentLimits)


# Load in the ideal parameters (Table 1 of Hanson et al. 2024)
calcDetails = NP.genfromtxt('data/Bcoeff_params.dat')
calcDetails = calcDetails[(calcDetails[:,0] == nn)*(calcDetails[:,1]==nnp)].squeeze()
kRmax = calcDetails[3]
kRmin = calcDetails[2]
WINDFACTOR = 2
modeSpacingFactor = None

# qx, qy and sigma grid to compute Bcoeefs
QX = NP.arange(-30,31)
QY = NP.arange(-30,31)
SIGMA = NP.array([0])

# Start timer
tini = time.time()

# If first run, will compute and oiverwrite file, otherwise add add to averaging of existing file
FirstRun = False
for iFile in range(iFileMin,iFileMax):

	print(iFile,DATADIR)

	# Load in the location data
	with NP.load(DATADIR + 'avgSupergranuleData_SurfaceFlows.npz',allow_pickle=True) as DICT:
		if SORTsize:
			avg_Flow_maps,Roll_inds_iFile,Ncount,nPad,xgrid,ygrid,sizes = [DICT[x] for x in ['avg_Flow_maps_size','Roll_inds_size','Ncount_size','nPad','xgrid','ygrid','sizes']]
		else:
			avg_Flow_maps,Roll_inds_iFile,Ncount,nPad,xgrid,ygrid,amplitudes = [DICT[x] for x in ['avg_Flow_maps','Roll_inds','Ncount','nPad','xgrid','ygrid','amplitudes']]		

	# define so empty lists to fill
	Bcoeff_grid_summed = []; Flow_Maps = []

	# This for loop is useless, but can be used in sub dividing individual cubes
	for tind in [1]:#range(len(tinds)-1):
		Roll_inds_iFile_iTime = Roll_inds_iFile.astype(int)
		

		# Initialize and load in the fourier cubes
		cart_MC = cartesian_modeCoupling(DATADIR + '/V.fits',\
											apodization_fitsPath = os.getcwd() + '/Apodize.fits',\
											dxFactor=2,dyFactor=2,kMax=2000/RSUN,OmegaMax=0.006*2*NP.pi,\
											timeSubSample=1)
		phi_kw = cart_MC.computeFFT(storeInInstance=True,fitsPathLoad = DATADIR + '/V_kw.fits' )


		# Build the appropriate grids
		kx,ky,omega = cart_MC.computeFreq()
		dkx = NP.diff(kx)[0];dky = NP.diff(ky)[0];dw = NP.diff(omega)[0]
		kxg,kyg = NP.meshgrid(kx,ky,indexing='ij')
		abskg = NP.sqrt(kxg**2+kyg**2)
		absk = NP.sqrt(kx[kx>=0]**2+ky[ky>=0]**2)

		# Build mask in order to avoid couplings at high and low freq, wave number
		omegaM,gamma,amp = cart_MC.ref_fit_params(absk,nn)
		indl = NP.argmin(abs(NP.nan_to_num(omegaM) / (2*NP.pi)-0.002))
		indr = NP.argmin(abs(NP.nan_to_num(omegaM) / (2*NP.pi)-0.005))
		mask = NP.where((abskg*RSUN < min(absk[indr]*RSUN,kRmax))*(abskg*RSUN > max(absk[indl]*RSUN,kRmin)),1,0)
		
		# We found that for the low n, there are so many couplings, you only need every second
		if nn < 5:
			mask[::2] = 0
			mask[:,::2] = 0
		
		# indices of fourier cube that can be used in couplking
		inds_kx,inds_ky = NP.where(mask)

		# Compute the azimuthally averaged amplitudes (N_nk)
		amps_avg  = cart_MC.compute_N_nk(mask,nn,30,num_linewidths=2,PLOT = False)
		amps_avgp = cart_MC.compute_N_nk(mask,nnp,30,num_linewidths=2,PLOT = False)


		# Compute the Bcoefficients
		print('\tComputing B coefficients')
		kxm,kym,Bcoeff_grid = cart_MC.compute_bcoeffs_parallel([nn,nnp],mask,QX,QY,SIGMA,[amps_avg,amps_avgp],nu_min=0.0015,nu_max=0.0055,absq_range=NP.array([0,300])/RSUN,\
														VERBOSE=True,nbProc=27,reorg_k=False,returnNoise=False,\
														windfactor=WINDFACTOR,modeSpacingFactor = modeSpacingFactor,rtype=None,Measurement=Measurement)

		# If in loop for many cubes, compute the noise in first run
		if not FirstRun:
			print('\tComputing Noise Model')
			kxmN,kymN,Noise_grid = cart_MC.compute_bcoeffs_parallel([nn,nnp],mask,QX,QY,SIGMA,[amps_avg,amps_avgp],nu_min=0.0015,nu_max=0.0055,absq_range=NP.array([0,300])/RSUN,\
																VERBOSE=True,nbProc=27,reorg_k=False,returnNoise='theory',\
																windfactor=WINDFACTOR,modeSpacingFactor=modeSpacingFactor,rtype=None,Measurement=Measurement)
		

			FirstRun = True

		# convert to real space for checks
		Flow_Maps_avgSG = 0
		Btmp = NP.sum(Bcoeff_grid,axis=-1).squeeze()
		Flow_Map = fft.ifftn(fft.ifftshift(NP.pad(Btmp,((nPad,nPad),(nPad,nPad)),constant_values=0)),norm='ortho').real
		for jj in range(len(Roll_inds_iFile_iTime)):
			Flow_Maps_avgSG += NP.roll(NP.roll(Flow_Map,Roll_inds_iFile_iTime[jj,0],axis=0),Roll_inds_iFile_iTime[jj,1],axis=1)
		# Flow_Maps_avgSG = Flow_Maps_avgSG/len(Roll_inds_iFile_iTime)
		

		
		# Perform fourier translation to average the Bcoefficients
		Bcoeff_avgSG = NP.zeros(Bcoeff_grid.shape + (len(percentiles),),'complex64')
		NSGs = NP.zeros(len(percentiles))
		qxgrid,qygrid = NP.meshgrid(QX*dkx,QY*dky,indexing='ij')
		dx = NP.diff(xgrid[:,0])[0]*1e6;dy = NP.diff(ygrid[0,:])[0]*1e6
		
		for jj in range(len(Roll_inds_iFile_iTime)):
			tmp = Bcoeff_grid * NP.exp(-1.j*(qxgrid * dx*Roll_inds_iFile_iTime[jj,0] + qygrid * dy*Roll_inds_iFile_iTime[jj,1]))[:,:,None,None]
			Bcoeff_avgSG[...,-1] += tmp
			for kk in range(len(percentiles) -1 ):
				if SORTsize:
					if sizes[jj] >= percentiles[kk] and sizes[jj] < percentiles[kk+1]:
						Bcoeff_avgSG[...,kk] += tmp
						NSGs[kk] += 1
				else:
					if amplitudes[jj] >= percentiles[kk] and amplitudes[jj] < percentiles[kk+1]:
						Bcoeff_avgSG[...,kk] += tmp
						NSGs[kk] += 1 
			NSGs[-1] += 1




			
		#---------------------------------------------------------------------
		#---------------------------------------------------------------------
		#---------------------------------------------------------------------
		# Save the data


		OUTDATADIR = DATADIR
		OUTFILE = OUTDATADIR + '/Bcoeffs_n%i_np%i%s.h5' % (nn,nnp,fileNameSuffix)

		# If not first calculation, load in previous calculation and update the appropriate data
		if iFile > 0:

			# os.system('mv %s %s' % (OUTFILE,OUTFILE_tmp))
			with h5py.File(OUTFILE,'r+') as h5fprev:
				NSGs_tmp = h5fprev['NumberSGs']
				NSGs_tmp[...] = NP.array(NSGs_tmp) + NSGs
				
				if iFile == len(InversionData) -1:
					Nscale = NP.array(NSGs_tmp) + NSGs
				else:
					Nscale = NP.ones(len(NSGs))



				Flow_Maps_avgSG_tmp = h5fprev['Flow_Maps_avgSG']
				Flow_Maps_avgSG_tmp[...] = (NP.array(Flow_Maps_avgSG_tmp) + 	Flow_Maps_avgSG	)/Nscale[-1]

				for ii in range(len(QX)):
					for jj in range(len(QY)):
						for kk in range(len(SIGMA)):
							if not VARIANCE_calc:
								Bcoeff_avgSG_tmp = h5fprev['Bcoeffs_avgSG']['QX%i' % QX[ii]]['QY%i' % QY[jj]]['SIGMA%i' % SIGMA[kk]]['data']
								Bcoeff_avgSG_tmp[...] = (Bcoeff_avgSG[ii,jj,kk] + NP.array(Bcoeff_avgSG_tmp))/Nscale
							else:
								Bcoeff_avgSG_tmp = h5fprev['Bcoeff_avgSG_variance']['QX%i' % QX[ii]]['QY%i' % QY[jj]]['SIGMA%i' % SIGMA[kk]]['data']
								Bcoeff_avgSG_tmp[...] = (Bcoeff_avgSG[ii,jj,kk] + NP.array(Bcoeff_avgSG_tmp))/Nscale[-1]

		else:
			with h5py.File(OUTFILE,'w') as h5f:
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

				# Bgrp = h5f.create_group('Bcoeffs_field')
				BAVGgrp = h5f.create_group('Bcoeffs_avgSG')
				Ngrp = h5f.create_group('NoiseModel')
				for ii in range(len(QX)):
					BAVGsubgrp = BAVGgrp.create_group('QX%i' % QX[ii])
					Nsubgrp    = Ngrp.create_group('QX%i' % QX[ii])
					for jj in range(len(QY)):
						BAVGssubgrp = BAVGsubgrp.create_group('QY%i' % QY[jj])
						Nssubgrp    = Nsubgrp.create_group('QY%i' % QY[jj])
						for kk in range(len(SIGMA)):
							BAVGsssubgrp = BAVGssubgrp.create_group('SIGMA%i' % SIGMA[kk])
							Nsssubgrp    = Nssubgrp.create_group('SIGMA%i' % SIGMA[kk])
							BAVGsssubgrp.create_dataset('data',data = Bcoeff_avgSG[ii,jj,kk])
							Nsssubgrp.create_dataset('data',data = Noise_grid[ii,jj,kk])

	

print(time.time() - tini,' seconds')
