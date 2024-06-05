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
ALPHAInd = 3


BASIS = 'Bspline'


if BASIS == 'Bspline':
	BasisStr = '_Bspline'
else:
	BasisStr = ''
Basis1D    = NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/InversionData/Basis%s.npy' % (BasisStr),allow_pickle=True)[0]

print(text_special('Computing for %s' %['CHEBYSHEV','BSPLINE'][int(BASIS=='Bspline')],'g',True,True))

xFinal = Basis1D.x_[::3]

subDir = '1Day'
drmsSeries = 'mTrack_modeCoupling_3d_30deg'
tinds = NP.arange(1,3)*1920
DATN = '/scratch/ch3246/OBSDATA/modeCouple/Cartesian/SG_INVERSION/%s[%i][090][090.0][+00.0][+00.0]/' % (drmsSeries,2200)
with h5py.File(DATN + '/%s/Bcoeffs/Bcoeffs_n%i_np%i_%i_%i%s.h5' % (subDir,0,0,tinds[0],tinds[1],DATN.split('/')[-2].split('g')[-1]),'r') as h5f:
	dkx,dky = [NP.array(h5f[x]) for x in ['dkx','dky']]


InversionData = NP.genfromtxt('data/InversionDataDir.txt',dtype=str)



OUTDIR = '/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/Solutions/SG_FIELD/FILT%s' % (['_None','_FILT%i' % int(FILTER)][int(FILTER is not False)])
Ncount = 0;res = 0
for iFile in range(len(InversionData)):
	FileName = OUTDIR + '/Solutions_%s.h5' % (InversionData[iFile][0].split('/')[-1].split('g')[-1])

	if not os.path.isfile(FileName):
		continue
	else:
		Ncount += 1

	with h5py.File(FileName,'r') as h5f:
		if iFile == 0:
			QX    = NP.array(h5f['QX'])
			QY    = NP.array(h5f['QY'])
			SIGMA = NP.array(h5f['SIGMA'])
			alpha = NP.array(h5f['alpha'])
		res_tmp = NP.array(h5f['result'])[ALPHAInd]

	res_tmp = Basis1D.reconstructFromBasis(res_tmp.real,xFinal=xFinal,axis=-1) + 1.j*Basis1D.reconstructFromBasis(res_tmp.imag,xFinal=xFinal,axis=-1)
	res += NP.abs(res_tmp)**2

res = res/Ncount


qxgrid,qygrid = NP.meshgrid(QX,QY,indexing='ij')
ABSQ = NP.sqrt((qxgrid*dkx*RSUN)**2+(qygrid*dky*RSUN)**2)
ABSQ = ABSQ.ravel()

absQ_bins = NP.histogram(ABSQ,bins=30)[1]
absQ_bins = absQ_bins[NP.argmin(abs(absQ_bins-15)):NP.argmin(abs(absQ_bins-415))]


POW_POL_q_avg = NP.zeros((len(xFinal),len(absQ_bins)-1));

for ii in range(len(xFinal)):
	DATpol       = res[:,:,0,ii].ravel()
	for binInd in range(len(absQ_bins)-1):
		inds = (ABSQ > absQ_bins[binInd])*(ABSQ < absQ_bins[binInd+1])
		POW_POL_q_avg  [ii,binInd] = NP.nanmean(DATpol       [inds])


plt.pcolormesh(absQ_bins[:-1] + NP.diff(absQ_bins)[0]/2,xFinal*1e-6,POW_POL_q_avg,vmax=10,rasterized=True)
plt.contour(absQ_bins[:-1] + NP.diff(absQ_bins)[0]/2,xFinal*1e-6,POW_POL_q_avg,cmap='jet',levels=15)

plt.xlim(50,300)
plt.ylim(bottom=-25)

plt.xlabel(r'$|q|$R$_\odot$')
plt.ylabel('Height [Mm]')