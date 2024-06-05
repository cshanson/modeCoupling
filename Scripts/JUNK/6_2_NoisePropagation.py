pathToRoutines = '/home/ch3246/mps_montjoie/'
import sys
sys.path.insert(0,pathToRoutines)
print("Importing pythonRoutines")
import numpy as NP
from pyCompHelio import *
from matplotlib.pyplot import *
from astropy.io import fits
from scipy.sparse import coo_matrix,csr_matrix,hstack,vstack
import time
plt.ion()

plt.close('all')


cross_coupling=False
Ngrid = NP.arange(6)

kRmin  = 200
kRmax  = 1000
nu_min = 0.002
nu_max = 0.004
BASIS = 'Bspline'
LM    = 2 # Order of the Lmatrix

if cross_coupling:
	kRmin  = 50
	kRmax  = 750
	Ngrid  = NP.array([2])
if BASIS == 'Bspline':
	BasisStr = '_Bspline'
else:
	BasisStr = ''

print(text_special('Computing for %s' %['CHEBYSHEV','BSPLINE'][int(BASIS=='Bspline')],'g',True,True))

#----------------------------------------------------------------
INV = tikhonov(stopRule='Chosen',N=2,forceAlpha=[-6,-2],VERBOSE=False)
# INV = tikhonov(stopRule='Lcurve',N=20,VERBOSE=True)


#-----------------Load Kernels --------------------------

if 'KK' not in locals():
	print(text_special('Loading in Kernels','y'))
	Basis1D    = NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/Basis%s.npy' % (BasisStr),allow_pickle=True)[0]


	InvDICT = NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions//Inv_Results_test_v2_L%i%s.npz' % (LM,BasisStr))
	alphaGrid   = InvDICT['alphaGrid']
	ALPHA       = InvDICT['guessAlpha']
	Lmatrix     = InvDICT['Lmatrix']


	ii = 0
	PG = progressBar(len(Ngrid),'serial')
	for nn in Ngrid:
		KernelDICT = NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/Kernels_Noise_test%s_n%i.npz' % (BasisStr,nn),allow_pickle=True)
		KKt,NoiseCovt,QX,QY,SIGMA,kx,ky,omega,dkx,dky,dw,mask = [KernelDICT[x] for x in ['Kernels','NoiseCov','QX','QY','SIGMA','kx','ky','omega','dkx','dky','dw','mask']]
	
		if ii == 0:
			KK = KKt
			NoiseCov = NoiseCovt
		else:
			KK = NP.concatenate([KK,KKt],axis=-1)
			NoiseCov = NP.concatenate([NoiseCov,NoiseCovt],axis=-1)

		PG.update()
		ii += 1
	del PG

	print(text_special('Done!','g'))


# N_C = R_K N_B R_K'
# where
# R_K = (K’ K)^{-1} K’
# Where K is a kernel. ' is the Hermitian conjugate transpose. N_B noise on the Bcoeff and N_C is the noise on the basis coefficients
alpha = 1.0539e4
noiseProp = NP.zeros((len(QX),len(QY),len(SIGMA),2*Basis1D.nbBasisFunctions_,2*Basis1D.nbBasisFunctions_))
noiseProp2 = NP.zeros((len(QX),len(QY),len(SIGMA),2*Basis1D.nbBasisFunctions_,2*Basis1D.nbBasisFunctions_))


def noise_prop(alpha,Kernel,NoiseCov,standardForm=False):
	# tstart = time.time()

	# Assign the variable
	KKi = Kernel.T
	NNi = NoiseCov

	# get the kernels and noise with non-zero values
	good_inds = (NP.sum(KKi,axis=1) != 0)*(NNi != 0)*(~NP.isnan(NNi))
	KKi = KKi[good_inds]
	NNi = NNi[good_inds]

	if len(KKi) == 0:
		return NP.zeros((KKi.shape[1],KKi.shape[1]))

	# tend = time.time()
	# print('1 Time taken: %1.4f sec' % (tend-tstart))


	# tstart = time.time()

	# Normalize by lambda
	NNb = NP.diag(NNi)
	# tend1 = time.time()
	# print('2.1 Time taken: %1.4f sec' % (tend1-tstart))


	Lambda = NP.diag(1/NP.sqrt(NNi))
	# abort
	# tend2 = time.time()
	# print('2.2 Time taken: %1.4f sec' % (tend2-tstart))


	KKi = NP.dot(Lambda,KKi)
	# tend3 = time.time()
	# print('2.3 Time taken: %1.4f sec' % (tend3-tstart))

	# NNb = NP.dot(NP.dot(Lambda,NNb),Lambda)
	NNb = NP.eye(len(Lambda))

	# tend = time.time()
	# print('2 Time taken: %1.4f sec' % (tend-tstart))


	# tstart = time.time()
	# turn into standard form
	if standardForm:
		Lp  = NP.linalg.inv(Lmatrix)
		KKi = (Lp.dot(KKi.T)).T
		LL  = NP.eye(KKi.shape[1])
	else:
		LL = Lmatrix

	KK2 = NP.dot(KKi.conj().T,KKi)+alpha**2*NP.dot(LL.T,LL)
	# tend = time.time()
	# print('3 Time taken: %1.4f sec' % (tend-tstart))


	# tstart = time.time()

	Rk = NP.dot( NP.linalg.inv(KK2),KKi.conj().T)

	Nprop = solarFFT.testRealFFT(NP.dot(Rk,NP.dot(NNb,Rk.conj().T))).real
	# tend = time.time()
	# print('4 Time taken: %1.4f sec' % (tend-tstart))
	if standardForm:
		return NP.dot(Lp,Nprop).dot(Lp)
	else:
		return Nprop

KKdim = KK.shape
KKt   = NP.moveaxis(KK.reshape(-1,KK.shape[-2],KK.shape[-1]),0,-1)	
NNt   = NP.moveaxis(NoiseCov.reshape(-1,KK.shape[-1]),0,-1)	

# tstart = time.time()
# ind = 572#NP.random.randint(0,962)
# print(ind)
# tmp = noise_prop(ALPHA,KKt[...,ind],NNt[...,ind])
# tmp2 = noise_prop(ALPHA,KKt[...,ind],NNt[...,ind],standardForm=True)

# tend = time.time()
# print('Total Time taken: %1.4f sec' % (tend-tstart))
# abort

tmpN = reduce(noise_prop,(ALPHA,KKt,NNt),KKt.shape[-1],15,progressBar=True)

noiseProp  = NP.moveaxis(tmpN,-1,0).reshape((len(QX),len(QY),len(SIGMA),2*Basis1D.nbBasisFunctions_,2*Basis1D.nbBasisFunctions_))
# noiseProp2 = NP.moveaxis(tmpN[1],-1,0).reshape((len(QX),len(QY),len(SIGMA),2*Basis1D.nbBasisFunctions_,2*Basis1D.nbBasisFunctions_))


NP.savez_compressed('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/NoiseProp_test_v2_L%i%s.npz' % (LM,BasisStr),\
					QX = QX,QY=QY,SIGMA=SIGMA,\
					dkx = dkx,dky = dky,dw = dw,\
					res = noiseProp)#,\
					# res2=noiseProp2)

