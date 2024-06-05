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



FILTER = False

nn = int(sys.argv[1])
# nn = 0; 
nnp = nn

subDir = '1Day'
InversionData = NP.genfromtxt('data/InversionDataDir.txt',dtype=str)
Amplitudes_all    = NP.genfromtxt('data/SG_Amplitudes.dat')
percentLimits     = [0,20,40,60,80,100]
percentiles       = NP.percentile(Amplitudes_all,percentLimits)
TEST = False
OVERWRITE = True


QX = NP.arange(-30,31)
QY = NP.arange(-30,31)
SIGMA = NP.array([0])

Ncount = 0;Flow_Maps_avgSG = 0
print(text_special('Averaging for p%i-p%i' % (nn,nnp),'y'))
PG = progressBar(len(InversionData),'serial')
for iFile in range(len(InversionData)):
	DATADIR = InversionData[iFile][0]
	tindl   = int(InversionData[iFile][1])
	tindr   = int(InversionData[iFile][2])

	BcoeffFile = DATADIR + '/%s/Bcoeffs/Bcoeffs_n%i_np%i_%i_%i%s%s.h5' % (subDir,nn,nnp,tindl,tindr,DATADIR.split('/')[-1].split('g')[-1],['','_FILT%i' % int(FILTER)][int(FILTER is not False)])

	with h5py.File(BcoeffFile,'r') as h5f:
		if iFile == 0:
			dims = ((len(h5f['Bcoeffs_avgSG']),len(h5f['Bcoeffs_avgSG']['QX0']),len(h5f['Bcoeffs_avgSG']['QX0']['QY0']),) +h5f['Bcoeffs_avgSG']['QX0']['QY0']['SIGMA0']['data'].shape)
			BBt = NP.zeros(dims,complex)
			NNt = NP.zeros(dims)[...,0]
			xgrid,ygrid,dkx,dky,dw,kRmax,kRmin,mask,kxm,kym,nPad,WINDFACTOR = [NP.array(h5f[x]) for x in ['xgrid','ygrid','dkx','dky','dw','kRmax','kRmin','mask','kxm','kym','nPad','WINDFACTOR']]
		NSGs,Flow_Maps_avgSGtmp = [NP.array(h5f[x]) for x in ['NumberSGs','Flow_Maps_avgSG']]
		# Flow_Maps_avgSG += Flow_Maps_avgSGtmp
		for qx in range(len(QX)):
			for qy in range(len(QY)):
				for sig in range(len(SIGMA)):
					BBt[qx,qy,sig] += NP.array(h5f['Bcoeffs_avgSG']['QX%i' % QX[qx]]['QY%i' % QY[qy]]['SIGMA%i' % SIGMA[sig]]['data'])
					if iFile == 0:
						NNt[qx,qy,sig] = NP.array(h5f['NoiseModel']['QX%i' % QX[qx]]['QY%i' % QY[qy]]['SIGMA%i' % SIGMA[sig]]['data'])
	Ncount += NSGs



	PG.update()
del PG

BBt = BBt / Ncount
Flow_Maps_avgSG = NP.sum(BBt,axis=-2).squeeze()
Flow_Maps_avgSG = fft.ifftn(fft.ifftshift(NP.pad(Flow_Maps_avgSG,((nPad,nPad),(nPad,nPad),(0,0)),constant_values=0),axes=(0,1)),norm='forward',axes=(0,1)).real


if TEST:
	Btmp = NP.sum(BBt,axis=(-2,-1))
	Flow_Map = fft.ifftn(fft.ifftshift(NP.pad(Btmp,((nPad,nPad),(nPad,nPad)),constant_values=0)),norm='ortho').real
	plt.plot(ygrid[100],Flow_Map[100])
	plt.plot(ygrid[100],Flow_Maps_avgSG[100],'.')
	abort

with h5py.File('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/Bcoeffs_AVG' + '/Bcoeffs_n%i_np%i%s.h5' % (nn,nnp,['','_FILT%i' % int(FILTER)][int(FILTER is not False)]),'w') as h5f:
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

	h5f.create_dataset('nPad',data = nPad)
	h5f.create_dataset('WINDFACTOR',data = WINDFACTOR)
	h5f.create_dataset('NumberSGs',data=Ncount)

	h5f.create_dataset('Flow_Maps_avgSG',data = Flow_Maps_avgSG)

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
	
				BAVGsssubgrp.create_dataset('data',data = BBt[ii,jj,kk])
				Nsssubgrp.create_dataset('data',data = NNt[ii,jj,kk])


print(text_special('Averaging for p%i-p%i COMPLETE' % (nn,nnp),'g'))
