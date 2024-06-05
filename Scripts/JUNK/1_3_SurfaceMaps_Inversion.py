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

TOROIDAL = False
CMD = 'ALPHA'
TargetDepths = NP.linspace(-5,0.4,75)*1e6



subDir = '1Day'
InversionData = NP.genfromtxt('data/InversionDataDir.txt',dtype=str)[:1]
DATADIR = InversionData[0][0]
tindl   = int(InversionData[0][1])
tindr   = int(InversionData[0][2])

Ngrid = NP.arange(7)

BASIS = 'Bspline'
LM    = 2 #Order of the Lmatrix


if BASIS == 'Bspline':
	BasisStr = '_Bspline'
else:
	BasisStr = ''
Basis1D    = NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/InversionData/Basis%s.npy' % (BasisStr),allow_pickle=True)[0]

print(text_special('Computing for %s' %['CHEBYSHEV','BSPLINE'][int(BASIS=='Bspline')],'g',True,True))

QX = NP.arange(0,31,2)
QY = NP.arange(-30,31,2)
SIGMA = NP.array([0])



if CMD == 'TEST':

	#-----------------Load Data --------------------------
	if 'KK' not in locals():
		QX = 11;QY = 0;SIGMA = 0
		print(text_special('Loading in Kernels and Bcoeffs','y'))
		with h5py.File('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/Kernels/Kernels_%s_SOLA_SurfaceMaps.h5' % (BASIS),'r') as h5f:
			# QX,QY = [NP.array(h5f[x]) for x in ['QX','QY']]
			KKt = NP.array(h5f['Kernels']['QX%i' % QX]['QY%i' % QY]['data'])
			if TOROIDAL:
				KK = KKt.reshape(-1,KKt.shape[-1]).T
			else:
				KK = KKt[0].T

		DATADIR = InversionData[0][0]
		tindl   = int(InversionData[0][1])
		tindr   = int(InversionData[0][2])
		BcoeffFile = DATADIR + '/%s/Surface_Flow_Maps/SurfaceFlows_Bcoeffs.h5' % (subDir)
		with h5py.File(BcoeffFile,'r') as h5f:
			xgrid,ygrid,dkx,dky = [NP.array(h5f[x]) for x in ['xgrid','ygrid','dkx','dky']]
			BB = NP.array(h5f['Bcoeffs']['QX%i' % QX]['QY%i' % QY]['SIGMA%i' % SIGMA]['data'])
			NN = NP.array(h5f['NoiseModel']['QX%i' % QX]['QY%i' % QY]['SIGMA%i' % SIGMA]['data'])



		print(text_special('Inverting for |q|Rsun = %1.2f' % (NP.sqrt((QX*dkx)**2+(QY*dky)**2)*RSUN),'g',True,True))



	good_inds = (KK[:,0] != 0)*(BB != 0)*(NN != 0)*(~NP.isnan(NN))
	KKt = KK[good_inds]
	NNt = NN[good_inds]
	BBt = BB[good_inds]



	NNi = 1/NP.sqrt(NNt)
	KKi = NNi[:,None]*KKt;BBi = NNi*BBt

	#-------------------------------------------------------------------
	# 				SOLA INVERSION
	#-------------------------------------------------------------------
	print(text_special('Computing SOLA inversion','y',True,True))

	# select f mode vertical wavelength at ell 750 (2.5mHz) 2/lambda_r = omega^2/c^2 - l(l+1)/RSUN^2
	TargetWidth = 2*NP.pi/NP.sqrt((2*NP.pi*0.0025)**2/7892**2 - 750**2/RSUN**2) /2
	TargetWidth = 0.5e6
	TargetDepths = NP.linspace(-5,0,10)*1e6

	coeffs,KKz,Target = SOLA_coeffCalc_MCA(KKi,1e-7,\
						Basis1D,TargetDepths,TargetWidth*NP.ones(len(TargetDepths)),True)

	plt.figure()
	plt.plot(Basis1D.x_*1e-6,Target[-1],'--k')
	plt.plot(Basis1D.x_*1e-6,NP.dot(coeffs[-1,:-1],KKz),'r')
	plt.xlim(-10,1)

	# X0 = -5; SIG = 2;AMP = 0.5
	X0 = 0; SIG = 2;AMP = 5

	synModelP = AMP*NP.exp(-(Basis1D.x_*1e-6 - X0)**2/(2*(SIG**2)))
	synModelT = 0.01*synModelP
	synBBi = NP.dot(KKi,Basis1D.projectOnBasis(synModelP))
	synBBiN = synBBi.real+NP.random.randn(len(BBi))
	plt.figure()
	plt.plot(BBi.real)
	plt.plot(synBBiN)
	plt.plot(synBBi.real)

	plt.figure()
	plt.plot(Basis1D.x_*1e-6,Basis1D.reconstructFromBasis(Basis1D.projectOnBasis(synModelP)))
	plt.plot(TargetDepths*1e-6,NP.dot(coeffs[:,:-1],synBBi),'.r',label='No Noise')
	plt.plot(TargetDepths*1e-6,NP.dot(coeffs[:,:-1],synBBiN),'.k',label = 'With Noise')
	plt.legend()
	plt.xlim(-10,1)

elif CMD == 'ALPHA':
	print(text_special('WARNING: currently subsampling KK,NN and BB','r',True,True))

	def computeSOLA_coeffs(QX,QY,SIGMA,alpha,TEST = False):

		with h5py.File('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/Kernels/Kernels_%s_SOLA_SurfaceMaps.h5' % (BASIS),'r') as h5f:
			# QX,QY = [NP.array(h5f[x]) for x in ['QX','QY']]
			KKt = NP.array(h5f['Kernels']['QX%i' % QX]['QY%i' % QY]['data'])
			if TOROIDAL:
				KK = KKt.reshape(-1,KKt.shape[-1]).T
			else:
				KK = KKt[0].T
		

		BB = []
		for iFile in range(len(InversionData)):
			DATADIR = InversionData[iFile][0]
			tindl   = int(InversionData[iFile][1])
			tindr   = int(InversionData[iFile][2])
			BcoeffFile = DATADIR + '/%s/Surface_Flow_Maps/SurfaceFlows_Bcoeffs.h5' % (subDir)

			with h5py.File(BcoeffFile,'r') as h5f:
				xgrid,ygrid,dkx,dky = [NP.array(h5f[x]) for x in ['xgrid','ygrid','dkx','dky']]
				BB.append(NP.array(h5f['Bcoeffs']['QX%i' % QX]['QY%i' % QY]['SIGMA%i' % SIGMA]['data']))
				if iFile == 0:
					NN = NP.array(h5f['NoiseModel']['QX%i' % QX]['QY%i' % QY]['SIGMA%i' % SIGMA]['data'])
		BB = NP.array(BB).T


		good_inds = (KK[:,0] != 0)*(BB[:,0] != 0)*(NN != 0)*(~NP.isnan(NN))
		KKt = KK[good_inds][::2]
		NNt = NN[good_inds][::2]
		BBt = BB[good_inds][::2]

		if NP.sum(good_inds) < 100:
			return NP.zeros(len(InversionData)) * 1.j



		NNi = 1/NP.sqrt(NNt)
		KKi = NNi[:,None]*KKt;BBi = NNi[:,None]*BBt

		if TEST:
			coeffs = NP.ones((1,len(BBi)+1))
		else:

			TargetWidth = 2*NP.pi/NP.sqrt((2*NP.pi*0.0025)**2/7892**2 - 750**2/RSUN**2) /2
			TargetWidth = 0.65e6

			coeffs,KKz,Target = SOLA_coeffCalc_MCA(KKi,alpha,\
								Basis1D,-0.5e6,[TargetWidth],True)

		# return [coeffs,KKz,Target]

		return [NP.trapz((Target[0] - NP.dot(coeffs[0,:-1],KKz))**2,x=Basis1D.x_),NP.sqrt(NP.sum(coeffs[0,:-1]**2*NNt))]

	tmp = computeSOLA_coeffs(11,0,0,1e2)



	alphaGrid = NP.logspace(-10,10,40)
	sols = reduce(computeSOLA_coeffs,(11,0,0,alphaGrid,False),len(alphaGrid),min(len(alphaGrid),25),progressBar = True)

	plt.figure()
	plt.loglog(sols[0],sols[1],'b',lw=2)
	plt.loglog(sols[0],sols[1],'.k',mew=2)
	for ii in range(sols.shape[-1]):
		plt.text(sols[0,ii],sols[1,ii],'(%i,%1.2e)' % (ii,alphaGrid[ii]))

	plt.xlabel('Noise',fontsize = 15)
	plt.ylabel('Error in the Fit',fontsize=15)
	plt.tight_layout()

	# sols2 = NP.reshape(sols.T,qxg.shape + (-1,))

	# sols = NP.concatenate([NP.conj(NP.flip(sols2[1:],axis=(0,1,2))),sols2],axis=0)

elif CMD == 'COMPUTE':
	print(text_special('WARNING: currently subsampling KK,NN and BB','r',True,True))

	def computeSOLA_coeffs(QX,QY,SIGMA,alpha,TEST = False,SaveCoeffs = False):

		with h5py.File('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/Kernels/Kernels_%s_SOLA_SurfaceMaps.h5' % (BASIS),'r') as h5f:
			# QX,QY = [NP.array(h5f[x]) for x in ['QX','QY']]
			KKt = NP.array(h5f['Kernels']['QX%i' % QX]['QY%i' % QY]['data'])
			if TOROIDAL:
				KK = KKt.reshape(-1,KKt.shape[-1]).T
			else:
				KK = KKt[0].T
		

		BB = []
		for iFile in range(len(InversionData)):
			DATADIR = InversionData[iFile][0]
			tindl   = int(InversionData[iFile][1])
			tindr   = int(InversionData[iFile][2])
			BcoeffFile = DATADIR + '/%s/Surface_Flow_Maps/SurfaceFlows_Bcoeffs.h5' % (subDir)

			with h5py.File(BcoeffFile,'r') as h5f:
				xgrid,ygrid,dkx,dky = [NP.array(h5f[x]) for x in ['xgrid','ygrid','dkx','dky']]
				BB.append(NP.array(h5f['Bcoeffs']['QX%i' % QX]['QY%i' % QY]['SIGMA%i' % SIGMA]['data']))
				if iFile == 0:
					NN = NP.array(h5f['NoiseModel']['QX%i' % QX]['QY%i' % QY]['SIGMA%i' % SIGMA]['data'])
		BB = NP.array(BB).T


		good_inds = (KK[:,0] != 0)*(BB[:,0] != 0)*(~NP.isnan(NN))*(NN != 0)
		# print(NP.sum(good_inds),len(NN))
		KKt = KK[good_inds][::2]
		NNt = NN[good_inds][::2]
		BBt = BB[good_inds][::2]

		if NP.sum(good_inds) < 100:
			return NP.zeros((len(TargetDepths),len(InversionData))) * 1.j



		NNi = 1/NP.sqrt(NNt)
		KKi = NNi[:,None]*KKt;BBi = NNi[:,None]*BBt

		if TEST:
			coeffs = NP.ones((len(TargetDepths),len(BBi)+1))
			BBi    = BBi
		else:
			outDir = '/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/Kernels/SOLA_SurfaceCoeffs/QX_%i/QY_%i/SIGMA_%i/' % (QX,QY,SIGMA)

			if SaveCoeffs:

				TargetWidth  = 2*NP.pi/NP.sqrt((2*NP.pi*0.0025)**2/7892**2 - 750**2/RSUN**2) /2
				TargetWidth  = 0.35e6

				coeffs,KKz,Target = SOLA_coeffCalc_MCA(KKi,alpha,\
									Basis1D,TargetDepths,TargetWidth*NP.ones(len(TargetDepths)),True)

				mkdir_p(outDir)
				with h5py.File(outDir + '/SOLAcoeffs.h5','w') as h5f:
					h5f.create_dataset('coeffs',data = coeffs)
					h5f.create_dataset('TargetDepths',data = TargetDepths)
					h5f.create_dataset('TargetWidth',data = TargetWidth)
					h5f.create_dataset('Kernelz',data = KKz)
					h5f.create_dataset('TargetFunc',data = Target)

			else:
				with h5py.File(outDir + '/SOLAcoeffs.h5','r') as h5f:
					coeffs = NP.array(h5f['coeffs'])

		# return [coeffs,KKz,Target]

		return NP.dot(coeffs[:,:-1],BBi)

	# tmp = computeSOLA_coeffs(11,0,0,1e-7,False,True)
	# abort


	qxg,qyg,sigma= NP.meshgrid(QX,QY,SIGMA,indexing='ij')

	sols = reduce(computeSOLA_coeffs,(qxg.ravel(),qyg.ravel(),sigma.ravel(),1e-7,False,True),len(qxg.ravel()),25,progressBar = True)

	sols2 = NP.reshape(sols.T,qxg.shape + sols.shape[:-1])

	sols = NP.concatenate([NP.conj(NP.flip(sols2[1:],axis=(0,1,2))),sols2],axis=0)

	solsDiv = -NP.gradient(sols,TargetDepths,axis=-2)

	plt.figure()
	plt.pcolormesh(QY,QY,NP.mean(abs(solsDiv[:,:,0,NP.argmin(abs(TargetDepths +0.5e6))])**2,axis=-1))

	solsB = reduce(computeSOLA_coeffs,(qxg.ravel(),qyg.ravel(),sigma.ravel(),1e-7,True),len(qxg.ravel()),25,progressBar = True)
	solsB = NP.reshape(solsB.T,qxg.shape + solsB.shape[:-1])
	solsB = NP.concatenate([NP.conj(NP.flip(solsB[1:],axis=(0,1,2))),solsB],axis=0)
	POWB = NP.mean(abs(solsB[:,:,0,0])**2,axis=-1)

	plt.figure()
	plt.pcolormesh(QY,QY,POWB.T)

	# POW = NP.mean(abs(solsDiv[:,:,0,NP.argmin(abs(TargetDepths +0.5e6))])**2,axis=-1)
	# POW2 = NP.mean(abs(solsB[:,:,0,NP.argmin(abs(TargetDepths +0.5e6))])**2,axis=-1)
	# dkx = dky = 1.6123484392467422e-08
	# ABSQ = NP.sqrt((QY[:,None]*dkx*RSUN)**2+(QY[None,:]*dky*RSUN)**2)
	# ABSQ = ABSQ.ravel()

	# absk_bins = NP.histogram(ABSQ,bins=25)[1]
	# absk_bins = absk_bins[NP.argmin(abs(absk_bins-15)):NP.argmin(abs(absk_bins-415))]

	# power_nn = NP.zeros((len(absk_bins)-1));
	# power_nnB = NP.zeros((len(absk_bins)-1));

	# DAT       = POW.ravel();DAT2 = POW2.ravel()
	# for binInd in range(len(absk_bins)-1):
	# 	inds = (ABSQ > absk_bins[binInd])*(ABSQ < absk_bins[binInd+1])
	# 	power_nn    [binInd] = NP.nanmean(DAT       [inds])
	# 	power_nnB   [binInd] = NP.nanmean(DAT2      [inds])

