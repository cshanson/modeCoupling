
# Routine to recompute the matrix for the inversion using slepians
# VERY VERY MEMORY INTENSIVE. USE HPC!

pathToRoutines = '/home/ch3246/mps_montjoie/'
import sys
sys.path.insert(0,pathToRoutines)
print("Importing pythonRoutines")
import numpy as NP
from pyCompHelio import *
from matplotlib.pyplot import *
from astropy.io import fits
from scipy.sparse import coo_matrix,csr_matrix,hstack,vstack
plt.ion()

plt.close('all')

# Padding for real space building
nPad = 70

# What is the real space contour radius to use
SlepContour = 45 #Mm
SlepianDict = scipy.io.loadmat(DATADIR + '/Slepian_%iMm.mat' % SlepContour)

# Define the basis and the order of the L matrix in the inversion
LM    = 2 #Order of the Lmatrix
BASIS = 'Bspline'
BasisStr = '_Bspline'

# n and associated n' coupling to use in the matrix
Ngrid = NP.concatenate([NP.arange(10),NP.arange(1,9),NP.arange(2,8)])
Npgrid = NP.concatenate([NP.arange(10),NP.arange(1,9)+1,NP.arange(2,8)+2])

# We now compute each kernel in a loop
Nruns = 0
for Measurement in ['POLOIDAL','UX','UY','UZ']:

	print('Computing for %s Kernels' % Measurement)

	# Load in thge basis
	Basis1D    = NP.load('../../OUTPUTDIR/Basis%s.npy' % (BasisStr),allow_pickle=True)[0]
	print(text_special('Computing for %s' %['CHEBYSHEV','BSPLINE'][int(BASIS=='Bspline')],'g',True,True))

	# Load in some Bcoeff information
	with h5py.File(DATADIR + '/Bcoeffs_n%i_np%i%s.h5' % (0,0,['','_cs'][int(Measurement.upper() in ['SOUNDSPEED','ALFVEN','DENSITY'])]),'r') as h5f:
		xgrid,ygrid,Ncount,dkx,dky,QX,QY = [NP.array(h5f[x]) for x in ['xgrid','ygrid','NumberSGs','dkx','dky','QX','QY']]
	
	#---------------------------------------------------------------------
	#---------------------------------------------------------------------
	
	# Load in the average ur maps for constraints
	with NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/InversionData/avgSupergranuleData_Doppler.npz') as npyDICT:
		ur_Doppler = -NP.moveaxis(npyDICT['DopplerAverage'],0,-1)[...,-1] # negative because z is outward, but doppler inward
		xgridD     = npyDICT['xgrid']
		ygridD     = npyDICT['ygrid']

	# Load in the horizontal maps for constraints
	with NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/InversionData/LCT/avgSupergranuleData_LCT.npz') as npyDICT:
		uxy_LCT = NP.moveaxis(npyDICT['LCTAverage'],0,-1)[...,10]
		div_LCT = NP.moveaxis(npyDICT['divAverage'],0,-1)[...,10]

	with NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/InversionData/avgSupergranuleData_Magnetogram_LCTcentered.npz') as npyDICT:
		Magdata = NP.moveaxis(npyDICT['MagnetogramAverage'],0,-1)[...,10] # negative because z is outward, but doppler inward


	#Load in the Slepians and compute the fft
	Slepians = SlepianDict['G']
	Lambda_alpha = SlepianDict['V'].squeeze()
	Qmax = SlepianDict['MaxEll'][0][0]
	CoeffsKeep = int(NP.ceil(NP.sum(Lambda_alpha)))
	Slepians_fft = fft.fftshift(fft.fftn(Slepians,axes=(0,1)),axes=(0,1))[nPad:-nPad,nPad:-nPad,:CoeffsKeep]* Lambda_alpha[:CoeffsKeep]

	print('Max Q in kernel: ',Qmax)

	# We build the constraint value
	constraint_values = simps(simps(div_LCT[...,None]*Slepians,x=xgridD[:,0]/amax(xgridD),axis=0),x=ygridD[0,:]/amax(ygridD),axis=0)
	MassMatrix = NP.zeros((Slepians.shape[-1],Slepians.shape[-1]))
	for ii in range(Slepians.shape[-1]):
		for jj in range(Slepians.shape[-1]):
			MassMatrix[ii,jj] = simps(simps(Slepians[...,ii]*Slepians[...,jj],x=xgridD[:,0]/amax(xgridD),axis=0),x=ygridD[0,:]/amax(ygridD),axis=0)
	constraint_values = NP.dot(constraint_values,NP.linalg.inv(MassMatrix))
	recon = NP.sum(Slepians * constraint_values,axis=-1)

	Slepians_abs = abs(Slepians_fft)


	#-----------------Load Data --------------------------
	Qmin  = 0 # use if you want minimum Q (not necessary)
	SIGMAind = 0
	print(text_special('Loading in Kernels and Bcoeffs','y'))

	# initialize some empty lists
	KK_total = [];BB_total = [];NN_total = [];BBv_total = [];
	QXinds_total = [];QYinds_total = []
	radial_order_total = [];radial_order_p_total = []
	qxprev = NP.nan
	maskq = NP.zeros((len(QX),len(QY)))
	QXfull = NP.arange(-30,31)
	QYfull = NP.arange(-30,31)

	# For each n n' coupling load in kernals and Bcoefficients
	for ii in range(len(Ngrid)):
		nn = Ngrid[ii]; nnp = Npgrid[ii]


		KK = [];BB = [];NN = [];BB_var = []
		KK_concat = [];BB_concat = [];BBv_concat = [];NN_concat = []
		QXinds = [];QYinds = [];radial_order = [];radial_order_p = []
		for qx in QX:
			for qy in QY:

				if NP.sqrt((qx*dkx*RSUN)**2 + (qy*dky*RSUN)**2) > Qmax:
					continue
				if NP.sqrt((qx*dkx*RSUN)**2 + (qy*dky*RSUN)**2) <= Qmin:
					continue

				# Note that the order you saved the kernels in WILL impact this part
				with h5py.File(DATADIR + '/Kernels%s_n%i_np%i%s.h5' % (BasisStr,nn,nnp,['_flow','_structure'][int(Measurement.upper() in ['SOUNDSPEED','ALFVEN','DENSITY'])]),'r') as h5f:
					KKt = NP.array(h5f['Kernels']['QX%i' % qx]['QY%i' % qy]['data'])
					mask_kk = NP.array(h5f['mask'])

					if Measurement.upper() == 'POLOIDAL':
						KKt = KKt[0].T
					elif Measurement.upper() == 'UX':
						KKt = KKt[1].T
					elif Measurement.upper() == 'UY':
						KKt = KKt[2].T
					elif Measurement.upper() == 'UZ':
						KKt = KKt[3].T
					elif Measurement.upper() == 'SOUNDSPEED':
						KKt = KKt[0].T
					elif Measurement.upper() == 'ALFVEN':
						KKt = KKt[1].T
					elif Measurement.upper() == 'DENSITY':
						KKt = KKt[2].T

				# load in the Bcoefficnets
				with h5py.File(DATADIR + '/Bcoeffs_n%i_np%i%s.h5' % (nn,nnp,['','_cs'][int(Measurement.upper() in ['SOUNDSPEED','ALFVEN','DENSITY'])]),'r') as h5f:
					xgrid,ygrid,Ncount,dkx,dky,kxm,kym,inds_kx,inds_ky,mask = [NP.array(h5f[x]) for x in ['xgrid','ygrid','NumberSGs','dkx','dky','kxm','kym','inds_kx','inds_ky','mask']]
					BBt = NP.array(h5f['Bcoeffs_avgSG']['QX%i' % qx]['QY%i' % qy]['SIGMA%i' % SIGMAind]['data'])
					NNt = NP.array(h5f['NoiseModel']['QX%i' % qx]['QY%i' % qy]['SIGMA%i' % SIGMAind]['data'])
				# Load in the inflow coeffs
				with h5py.File(DATADIR + '/Bcoeffs_n%i_np%i%s_INFLOW.h5' % (nn,nnp,['','_cs'][int(Measurement.upper() in ['SOUNDSPEED','ALFVEN','DENSITY'])]),'r') as h5f:
					BBt_inflow = NP.array(h5f['Bcoeffs_avgSG']['QX%i' % qx]['QY%i' % qy]['SIGMA%i' % SIGMAind]['data'])
				BBt = NP.concatenate([BBt,BBt_inflow],axis=-1)

				# Load in the variance
				with h5py.File(DATADIR + '/Bcoeffs_n%i_np%i_var%s.h5' % (nn,nnp,['','_cs'][int(Measurement.upper() in ['SOUNDSPEED','ALFVEN','DENSITY'])]),'r') as h5f:
					BBv = NP.array(h5f['Bcoeff_avgSG_variance']['QX%i' % qx]['QY%i' % qy]['SIGMA%i' % SIGMAind]['data'])[:,-1]

				# Ensure all Bcoeffients of the population 10 group are non-zero
				good_inds = (BBt[:,10] != 0)
				KKt = KKt[good_inds]
				BBt = BBt[good_inds]
				NNt = NNt[good_inds]
				BBv = BBv[good_inds]


				# Store the QX if you want
				QXinds.append(NP.ones(len(KKt))*NP.argmin(abs(QXfull - qx)))
				QYinds.append(NP.ones(len(KKt))*NP.argmin(abs(QYfull - qy)))
				maskq[NP.argmin(abs(QX - qx)),NP.argmin(abs(QY - qy))] = 1

				# Store the radial order of each Bcoeff
				radial_order.append(NP.ones(len(KKt))*nn)
				radial_order_p.append(NP.ones(len(KKt))*nnp)

				
				KK.append(KKt)
				BB.append(BBt)
				NN.append(NNt)
				BB_var.append(BBv)

			if len(KK) > 0:
				KK_concat.append(NP.concatenate(KK,axis=0))
				BB_concat.append(NP.concatenate(BB,axis=0))
				NN_concat.append(NP.concatenate(NN,axis=0))
				BBv_concat.append(NP.concatenate(BB_var,axis=0))
				KK = [];BB = [];NN = [];BB_var = [];

		KK_total.append(NP.concatenate(KK_concat,axis=0))
		BB_total.append(NP.concatenate(BB_concat,axis=0))
		NN_total.append(NP.concatenate(NN_concat,axis=0))
		BBv_total.append(NP.concatenate(BBv_concat,axis=0))
		QXinds_total.append(NP.concatenate(QXinds,axis=0))
		QYinds_total.append(NP.concatenate(QYinds,axis=0))
		radial_order_total.append(NP.concatenate(radial_order,axis=0))
		radial_order_p_total.append(NP.concatenate(radial_order_p,axis=0))


	KK_total     = NP.concatenate(KK_total,axis=0)
	BB_total     = NP.concatenate(BB_total,axis=0)
	NN_total     = NP.concatenate(NN_total,axis=0)
	BBv_total    = NP.concatenate(BBv_total,axis=0)
	QXinds_total = NP.concatenate(QXinds_total,axis=0).astype(int)
	QYinds_total = NP.concatenate(QYinds_total,axis=0).astype(int)
	radial_order_total = NP.concatenate(radial_order_total,axis=0).astype(int)
	radial_order_p_total = NP.concatenate(radial_order_p_total,axis=0).astype(int)

	Slepians_fft_KK = []
	for ii in range(len(QXinds_total)):
		Slepians_fft_KK.append(Slepians_fft[QXinds_total[ii],QYinds_total[ii]])
	Slepians_fft_KK = NP.array(Slepians_fft_KK)


	with h5py.File(DATADIR + '/SlepianKernels_%s_%s_%iMm.h5' % (BASIS,Measurement.upper(),SlepContour),'w') as h5f:
		h5f.create_dataset('Kernels',data = KK_total)
		h5f.create_dataset('Bcoeffs',data = BB_total)
		h5f.create_dataset('Bcoeffs_var',data = BBv_total)
		h5f.create_dataset('NoiseModel',data = NN_total)
		h5f.create_dataset('SlepianCoeffs',data = Slepians_fft_KK)
		h5f.create_dataset('QXinds',data = QXinds_total)
		h5f.create_dataset('QYinds',data = QYinds_total)
		h5f.create_dataset('radial_order',data = radial_order_total)
		h5f.create_dataset('radial_order_p',data = radial_order_p_total)
		h5f.create_dataset('Nx',data = len(Slepians))

	Nruns +=1
