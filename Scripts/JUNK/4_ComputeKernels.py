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


nn = int(sys.argv[1])
# nn  = 2
nnp = nn + 2

ngrid = [[nn,nnp]]

BASIS = 'Bspline'

Basis1D    = NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/InversionData/Basis_%s.npy' % (BASIS),allow_pickle=True)[0]


print(text_special('Computing for %s' %['CHEBYSHEV','BSPLINE'][int(BASIS=='Bspline')],'g',True,True))

subDir = '1Day'

InversionData = NP.genfromtxt('data/InversionDataDir.txt',dtype=str)
DATADIR = InversionData[0][0]
tindl   = int(InversionData[0][1])
tindr   = int(InversionData[0][2])

#-----------------FIG 2--------------------------
if 'cart_MC' not in locals():
	cart_MC = cartesian_modeCoupling(DATADIR + '/V.fits',\
									os.getcwd() + '/Apodize.fits',\
									dxFactor=2,dyFactor=2,kMax=2000/RSUN,\
									OmegaMax = 0.006*2*NP.pi,\
									daysPad = None,timeSubSample=1,\
									timeInds = [tindl,tindr])
	cart_MC.computeFFT(storeInInstance=True,fitsPathLoad = DATADIR + '/%s/V_kw_%i_%i.fits' % (subDir,cart_MC.timeInds_[0],cart_MC.timeInds_[1]))


	with h5py.File(DATADIR + '/%s/Bcoeffs/Bcoeffs_n%i_np%i_%i_%i%s.h5' % (subDir,nn,nnp,tindl,tindr,DATADIR.split('/')[-1].split('g')[-1]),'r') as DICT:
		QX,QY,dkx,dky = [NP.array(DICT[x]) for x in ['QX','QY','dkx','dky']]
		inds_kx,inds_ky,mask = [NP.array(DICT[x]) for x in ['inds_kx','inds_ky','mask']]
		mask[:,:] = 0
		mask[inds_kx,inds_ky] = 1


KK      = cart_MC.compute_kernels(nn,0.003*2*NP.pi,inds_kx[1500],inds_ky[1500],QX,QY,\
									Basis1D,multiplyH=False,scaleFields=2)


kxm,kym,Kernels = cart_MC.compute_kernels_parallel([nn,nnp],None,mask,QX,QY,\
										Basis1D,reorg_k=False,multiplyH=False,nbProc=25,\
										scaleFields=2)



with h5py.File('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/Kernels/Kernels_%s_n%i_np%i.h5' % (BASIS,nn,nnp),'w') as h5f:
	h5f.create_dataset('QX',data = QX)
	h5f.create_dataset('QY',data = QY)

	h5f.create_dataset('kx',data = kxm)
	h5f.create_dataset('ky',data = kym)
	h5f.create_dataset('mask',data = mask)

	grp = h5f.create_group('Kernels')
	for ii in range(len(QX)):
		subgrp = grp.create_group('QX%i' % QX[ii])
		for jj in range(len(QY)):
			subsubgrp = subgrp.create_group('QY%i' % QY[jj])
			subsubgrp.create_dataset('data',data = Kernels[:,ii,jj,0])


