#!/scratch/ch3246/project/PythonEnv/miniconda3/envs/dalma-python3-CPU/bin/python

# Routine for computing the Kernels. Computationally expensive, use on cluster
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


# state the radial order and the shift of n'
nn  = 9
nnshift = 0
nnp = nn + nnshift
ngrid = [[nn,nnp]]

# What basis do you wish to use
BASIS = 'Bspline'
Basis1D    = NP.load('../../OUTPUTDIR/Basis_%s.npy' % (BASIS),allow_pickle=True)[0]


print(text_special('Computing for %s' %['CHEBYSHEV','BSPLINE'][int(BASIS=='Bspline')],'g',True,True))



#-----------------FIG 2--------------------------
if 'cart_MC' not in locals():
	# Initialize the mode coupling class
	cart_MC = cartesian_modeCoupling(DATADIR + '/V.fits',\
									os.getcwd() + '/Apodize.fits',\
									dxFactor=2,dyFactor=2,kMax=2000/RSUN,\
									OmegaMax = 0.006*2*NP.pi,\
									daysPad = None,timeSubSample=1)
	cart_MC.computeFFT(storeInInstance=True,fitsPathLoad = DATADIR + '/V_kw.fits')

	# Load in the kx,ky inds actually used in the Bcoefficient calculation
	with h5py.File(DATADIR + '/Bcoeffs_n%i_np%i.h5' % (nn,nnp),'r') as DICT:
		QX,QY,dkx,dky = [NP.array(DICT[x]) for x in ['QX','QY','dkx','dky']]
		inds_kx,inds_ky,mask = [NP.array(DICT[x]) for x in ['inds_kx','inds_ky','mask']]
		mask[:,:] = 0
		mask[inds_kx,inds_ky] = 1

# We run the kernel routine once of any qx, qy, just to initialize the eigenfunctions
qxind  = 41;qyind = 31
INDx = 0;INDy = 0
KK      = cart_MC.compute_kernels([nn,nnp],0.003*2*NP.pi,inds_kx[INDx],inds_ky[INDy],[qxind],[qyind],\
									Basis1D,['POLOIDAL'],multiplyH=False,scaleFields=2)


# COmpute the kernels in parallel
kxm,kym,Kernels = cart_MC.compute_kernels_parallel([nn,nnp],None,mask,QX,QY,\
										Basis1D,['POLOIDAL','UX','UY','UZ'],\
										reorg_k=False,multiplyH=False,nbProc=25,\
										scaleFields=2)


# Save the kernels
with h5py.File(DATADIR + '/Kernels_Slepians/Kernels_%s_n%i_np%i_flow.h5' % (BASIS,nn,nnp),'w') as h5f:
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


