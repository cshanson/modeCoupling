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
plt.ion()

plt.close('all')

TOROIDAL = False
FILTER = False

OUTDIR = '/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/Solutions/'


subDir = '1Day'
InversionData = NP.genfromtxt('data/InversionDataDir.txt',dtype=str)
drmsSeries = 'mTrack_modeCoupling_3d_30deg'

tinds = NP.arange(1,3)*1920

Ngrid = NP.arange(8)

BASIS = 'Bspline'
LM    = 2 #Order of the Lmatrix


if BASIS == 'Bspline':
	BasisStr = '_Bspline'
else:
	BasisStr = ''
Basis1D    = NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/InversionData/Basis%s.npy' % (BasisStr),allow_pickle=True)[0]

print(text_special('Computing for %s' %['CHEBYSHEV','BSPLINE'][int(BASIS=='Bspline')],'g',True,True))


#-----------------Build Smoothness Matrix--------------
Lmatrix0  = Basis1D.createSmoothnessMatrix(0)
Lmatrix0  = Lmatrix0 / NP.amax(Lmatrix0)
Lmatrix   = Basis1D.createSmoothnessMatrix(LM)
Lmatrix   = Lmatrix/NP.amax(Lmatrix)

if TOROIDAL:
	Lmatrix0  = NP.block([[Lmatrix0,NP.zeros(Lmatrix0.shape)],[NP.zeros(Lmatrix0.shape),Lmatrix0]])
	Lmatrix   = NP.block([[Lmatrix,NP.zeros(Lmatrix.shape)],[NP.zeros(Lmatrix.shape),Lmatrix]])
Lmatrix = Lmatrix + 1e-5*Lmatrix0 #+ 5e-4*NP.eye(len(Lmatrix))


DATN = '/scratch/ch3246/OBSDATA/modeCouple/Cartesian/SG_INVERSION/%s[%i][090][090.0][+00.0][+00.0]/' % (drmsSeries,2200)
with h5py.File(DATN + '/%s/Bcoeffs/Bcoeffs_n%i_np%i_%i_%i%s.h5' % (subDir,0,0,tinds[0],tinds[1],DATN.split('/')[-2].split('g')[-1]),'r') as h5f:
	QX,QY,SIGMA = [NP.array(h5f[x]) for x in ['QX','QY','SIGMA']]
	dkx,dky = [NP.array(h5f[x]) for x in ['dkx','dky']]
	


#-----------------Load Data --------------------------
def RunInversion(iFile,alpha,qMax = 300,VERBOSE = False):
	if not hasattr(alpha,'__len__'):
		alpha = [alpha]
	if VERBOSE:
		print(text_special('Loading in Kernels and Bcoeffs','y'))
		PG = progressBar(len(QX),'serial')
	tini = time.time()

	res_tot = NP.zeros((len(QX),len(QY),len(SIGMA),Basis1D.nbBasisFunctions_,len(alpha)),complex)

	for qx in range(len(QX)):
		for qy in range(len(QY)):
			if NP.sqrt((QX[qx]*dkx*RSUN)**2 + (QY[qy]*dky*RSUN)**2) > qMax:
				continue
			for sig in range(len(SIGMA)):
				for ii in range(len(Ngrid)):
					nn = Ngrid[ii]; nnp = nn
					with h5py.File('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/Kernels/Kernels%s_n%i_np%i.h5' % (BasisStr,nn,nnp),'r') as h5f:
						KKt = NP.array(h5f['Kernels']['QX%i' % QX[qx]]['QY%i' % QY[qy]]['data'])
						if TOROIDAL:
							KKt = KKt.reshape(-1,KKt.shape[-1]).T
						else:
							KKt = KKt[0].T

					DATADIR = InversionData[iFile][0]
					tindl   = int(InversionData[iFile][1])
					tindr   = int(InversionData[iFile][2])
					BcoeffFile = DATADIR + '/%s/Bcoeffs/Bcoeffs_n%i_np%i_%i_%i%s.h5' % (subDir,nn,nnp,tindl,tindr,DATADIR.split('/')[-1].split('g')[-1])
					if not os.path.isfile(BcoeffFile):
						continue
					with h5py.File(BcoeffFile,'r') as h5f:
						BBt = NP.array(h5f['Bcoeffs_field']['QX%i' % QX[qx]]['QY%i' % QY[qy]]['SIGMA%i' % SIGMA[sig]]['data'])
						NNt = NP.array(h5f['NoiseModel']['QX%i' % QX[qx]]['QY%i' % QY[qy]]['SIGMA%i' % SIGMA[sig]]['data'])


					if ii == 0:
						KK = KKt
						BB = BBt
						NN = NNt
						ns = NP.ones(BBt.shape[-1])*nn
					else:
						KK = NP.concatenate([KK,KKt],axis=0)
						BB = NP.append(BB,BBt) 
						NN = NP.append(NN,NNt) 
						ns = NP.append(ns,NP.ones(BBt.shape[0])*nn) 


	# if VERBOSE:
	# 	del PG
	# 	print('Loading time: ',time.time() - tini)
	# 	print(text_special('Inverting for |q|Rsun = %1.2f' % (NP.sqrt((QX*dkx)**2+(QY*dky)**2)*RSUN),'g',True,True))

				tini = time.time()
				good_inds = (KK[:,0] != 0)*(BB != 0)*(NN != 0)*(~NP.isnan(NN))
				KKt = KK[good_inds]
				NNt = NN[good_inds]
				BBt = BB[good_inds]
				nst = ns[good_inds]

				if sum(good_inds) == 0:
					continue

	# if VERBOSE:
		# print('Kernel size: ', KKt.shape)

				# NNi = NP.diag(1/NP.sqrt(NNt))
				# KKi = NP.dot(NNi,KKt);BBi = NP.dot(NNi,BBt)

					# computationally efficient but works only if NoiseCov is diagonal
				NNi = 1/NP.sqrt(NNt)
				KKi = NNi[:,None]*KKt;BBi = NNi*BBt


				res  = RLSinversion_MCA(KKi,BBi.real,alpha,Lmatrix=Lmatrix)
				resI = RLSinversion_MCA(KKi,BBi.imag,alpha,Lmatrix=Lmatrix)
				res_tot[qx,qy,sig,:,:] = res + resI*1.j
		if VERBOSE:
			PG.update()

	res_tot = NP.moveaxis(res_tot,-1,0)
	if VERBOSE:
		del PG
		return res_tot


	outDir = OUTDIR  + '/SG_FIELD/FILT%s/' % (['_None','_%i'%int(FILTER)][FILTER is not False]) 
	mkdir_p(outDir)
	with h5py.File(outDir + '/Solutions_%s.h5' % (InversionData[iFile][0].split('/')[-1].split('g')[-1]),'w') as h5f:
		h5f.create_dataset('result',data = res_tot)
		h5f.create_dataset('alpha',data = alpha)
		h5f.create_dataset('QX',data = QX)
		h5f.create_dataset('QY',data = QY)
		h5f.create_dataset('SIGMA',data = SIGMA)
	

	return 1




# # # # --------------------------------------------------------
# # # # #			Test Rountine
# # # # #---------------------------------------------------------
# QX_ind = 11;QY_ind = 0;SIGMA_ind = 0
# tini = time.time()
# res = RunInversion(0,[5e3,5e4],300,True)
# print('Inversion time = ',time.time()-tini)

# # plt.plot(Basis1D.x_*1e-6,Basis1D.reconstructFromBasis(res.real[:,0]))
# # plt.plot(Basis1D.x_*1e-6,Basis1D.reconstructFromBasis(res.imag[:,0]))

# abort
#--------------------------------------------------------
#		Run Inversions
#--------------------------------------------------------


alphaGrid = NP.linspace(1,50,10)*1e3
res_total = reduce(RunInversion,(NP.arange(len(InversionData)),alphaGrid,300,False),len(InversionData),12,progressBar=True)

