pathToRoutines = '/home/ch3246/mps_montjoie/'
import sys
sys.path.insert(0,pathToRoutines)
print("Importing pythonRoutines")
import numpy as NP
from pyCompHelio import *
from matplotlib.pyplot import *
from astropy.io import fits
from scipy.sparse import coo_matrix,csr_matrix,hstack,vstack
import emcee

plt.ion()

plt.close('all')

TOROIDAL = False
FILTER = False

subDir = '1Day'
InversionData = NP.genfromtxt('data/InversionDataDir.txt',dtype=str)
tinds = NP.arange(1,3)*1920

# Ngrid = NP.arange(8)
# Npgrid = copy.copy(Ngrid)
Ngrid = NP.concatenate([NP.arange(8),NP.arange(2,6),NP.arange(2,4)])
Npgrid = NP.concatenate([NP.arange(8),NP.arange(2,6)+1,NP.arange(2,4)+2])

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

# abort

#-----------------Load Data --------------------------
if 'KK' not in locals():
	QX = 11;QY = 0;SIGMA = 0
	print(text_special('Loading in Kernels and Bcoeffs','y'))
	for ii in range(len(Ngrid)):
		nn = Ngrid[ii]; nnp = Npgrid[ii]
		with h5py.File('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/Kernels/Kernels%s_n%i_np%i.h5' % (BasisStr,nn,nnp),'r') as h5f:
			# QX,QY = [NP.array(h5f[x]) for x in ['QX','QY']]
			KKt = NP.array(h5f['Kernels']['QX%i' % QX]['QY%i' % QY]['data'])
			if TOROIDAL:
				KKt = KKt.reshape(-1,KKt.shape[-1]).T
			else:
				KKt = KKt[0].T
		# Flows_avg_SG = 0;Ncount = 0;BBt = 0

		with h5py.File('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/Bcoeffs_AVG' + '/Bcoeffs_n%i_np%i%s.h5' % (nn,nnp,['','_FILT%i' % int(FILTER)][int(FILTER is not False)]),'r') as h5f:
			xgrid,ygrid,Ncount,dkx,dky = [NP.array(h5f[x]) for x in ['xgrid','ygrid','NumberSGs','dkx','dky']]
			BBt = NP.array(h5f['Bcoeffs_avgSG']['QX%i' % QX]['QY%i' % QY]['SIGMA%i' % SIGMA]['data'])
			NNt = NP.array(h5f['NoiseModel']['QX%i' % QX]['QY%i' % QY]['SIGMA%i' % SIGMA]['data'])

		# for jj in range(len(InversionData)):
		# 	DATADIR = InversionData[jj][0]
		# 	tindl   = int(InversionData[jj][1])
		# 	tindr   = int(InversionData[jj][2])
		# 	BcoeffFile = DATADIR + '/%s/Bcoeffs/Bcoeffs_n%i_np%i_%i_%i%s.h5' % (subDir,nn,nnp,tindl,tindr,DATADIR.split('/')[-1].split('g')[-1])
		# 	if not os.path.isfile(BcoeffFile):
		# 		continue
		# 	with h5py.File(BcoeffFile,'r') as h5f:
		# 		xgrid,ygrid,NSGs,dkx,dky = [NP.array(h5f[x]) for x in ['xgrid','ygrid','NumberSGs','dkx','dky']]
		# 		BBt += NP.array(h5f['Bcoeffs_avgSG']['QX%i' % QX]['QY%i' % QY]['SIGMA%i' % SIGMA]['data'])
		# 		if jj == 0:
		# 			NNt = NP.array(h5f['NoiseModel']['QX%i' % QX]['QY%i' % QY]['SIGMA%i' % SIGMA]['data'])
		# 		Ncount += NSGs
		# BBt = BBt/Ncount

		if ii == 0:
			KK = KKt
			BB = BBt
			NN = NNt
			ns = NP.ones(BBt.shape[-1])*nn
		else:
			KK = NP.concatenate([KK,KKt],axis=0)
			BB = NP.append(BB,BBt) 
			NN = NP.append(NN,NNt) 
			ns = NP.append(ns,NP.ones(BBt.shape[-1])*nn) 
	print(text_special('Inverting for |q|Rsun = %1.2f' % (NP.sqrt((QX*dkx)**2+(QY*dky)**2)*RSUN),'g',True,True))


	del KKt,BBt,NNt


good_inds = (KK[:,0] != 0)*(BB != 0)*(NN != 0)*(~NP.isnan(NN))
KKt = KK[good_inds]
NNt = NN[good_inds]
BBt = BB[good_inds]
nst = ns[good_inds]



NNi = 1/NP.sqrt(NNt)
KKi = NNi[:,None]*KKt;BBi = NNi*BBt



res = RLSinversion_MCA(KKi,BBi.real,5e3,Lmatrix=Lmatrix)
res = res.squeeze()


#--------------------------------------------------------------------
#				MCMC
#--------------------------------------------------------------------

theta_guess  = res.real
priors       = NP.array([-1,1])[None,:] * NP.ones(Basis1D.nbBasisFunctions_)[:,None]


def ForwardModel(theta):
	return abs(NP.dot(KKi,theta))**2

def lnlike(theta,BBi):
    model = NP.dot(KKi,theta)
    return -0.5 * NP.sum((BBi - model) ** 2)

# def lnlike(theta,x,y):
# 	return -NP.sum(NP.log(ForwardModel(theta)) + y/Model_Lorenz(theta,x))
	
def lnprior(theta,priors):

	nChecks = 0
	for ii in range(len(theta)):
		if priors[ii][0] < theta[ii] < priors[ii][1]:
			nChecks += 1

	if nChecks == len(theta):
		return 0
	return - NP.inf


def lnprob(theta,BBi,priors):
    lp = lnprior(theta,priors)
    if not NP.isfinite(lp):
        return -NP.inf
    return lp + lnlike(theta,BBi)

ndim, nwalkers = len(priors), 100
pos = [theta_guess + 1e-2*NP.random.randn(ndim)*NP.diff(priors,axis=1).squeeze() for i in range(nwalkers)]


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(BBi.real,priors))
sampler.run_mcmc(pos, 1000,rstate0=NP.random.get_state())


samples = sampler.chain[:, 500:, :].reshape((-1, ndim))

samples_recon = Basis1D.reconstructFromBasis(samples,axis=1)


#------------------------------------------------------------------
#				Plots
#------------------------------------------------------------------

plt.errorbar(Basis1D.x_*1e-6,NP.mean(samples_recon,axis=0),yerr=NP.std(samples_recon,axis=0),capsize=3)
plt.plot(Basis1D.x_*1e-6,Basis1D.reconstructFromBasis(res.real),label='RLS')
plt.legend()