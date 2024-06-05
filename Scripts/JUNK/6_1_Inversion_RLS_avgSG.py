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

QX = NP.arange(-30,31);
QY = NP.arange(-30,31);
SIGMA = NP.array([0])

OUTDIR = '/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/Solutions/SG_AVG/'
mkdir_p(OUTDIR)

Amplitudes_all    = NP.genfromtxt('data/SG_Amplitudes.dat')
percentLimits     = [0,20,40,60,80,100]
percentiles       = NP.percentile(Amplitudes_all,percentLimits)

subDir = '1Day'
InversionData = NP.genfromtxt('data/InversionDataDir.txt',dtype=str)
drmsSeries = 'mTrack_modeCoupling_3d_30deg'

tinds = NP.arange(1,3)*1920

Ngrid = NP.arange(8)
Npgrid = copy.copy(Ngrid)
# Ngrid = NP.concatenate([NP.arange(8),NP.arange(2,6),NP.arange(2,4)])
# Npgrid = NP.concatenate([NP.arange(8),NP.arange(2,6)+1,NP.arange(2,4)+2])

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
	dkx,dky = [NP.array(h5f[x]) for x in ['dkx','dky']]
	

#-----------------Load Data --------------------------
def RunInversion(QX,QY,SIGMA,alpha,urConstraint=False,divConstraint=False,VERBOSE = False):
	if not hasattr(alpha,'__len__'):
		alpha = [alpha]
	if VERBOSE:
		print(text_special('Loading in Kernels and Bcoeffs','y'))

	tini = time.time()
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
			ns = NP.ones(BBt.shape[0])*nn
		else:
			KK = NP.concatenate([KK,KKt],axis=0)
			BB = NP.concatenate([BB,BBt],axis=0)
			NN = NP.append(NN,NNt) 
			ns = NP.append(ns,NP.ones(BBt.shape[0])*nn) 
	if VERBOSE:
		print('Loading time: ',time.time() - tini)
		print(text_special('Inverting for |q|Rsun = %1.2f' % (NP.sqrt((QX*dkx)**2+(QY*dky)**2)*RSUN),'g',True,True))

	tini = time.time()
	good_inds = (KK[:,0] != 0)*(BB[:,0] != 0)*(NN != 0)*(~NP.isnan(NN))
	KKt = KK[good_inds]
	NNt = NN[good_inds]
	BBt = BB[good_inds]
	nst = ns[good_inds]

	if len(good_inds) < 500:
		return 0

	if VERBOSE:
		print('Kernel size: ', KKt.shape)

	# NNi = NP.diag(1/NP.sqrt(NNt))
	# KKi = NP.dot(NNi,KKt);BBi = NP.dot(NNi,BBt)


	BsplineCoeffs = [];dBsplineCoeffs = []
	for ii in range(Basis1D.nbBasisFunctions_):
		BsplineCoeffs.append(Basis1D(ii,x=NP.array([0])))
		dBsplineCoeffs.append(Basis1D(ii,x=NP.array([0]),derivative=1))
	BsplineCoeffs = NP.squeeze(BsplineCoeffs)
	dBsplineCoeffs = -NP.squeeze(dBsplineCoeffs) #Negative because -q^2 dz(P) = divh(u) 

	if divConstraint is not False:
		dBsplineCoeffs = dBsplineCoeffs*1e6 
		divConstraint  = divConstraint*1e6 # to avoid singular matrix

	# computationally efficient but works only if NoiseCov is diagonal
	NNi = 1/NP.sqrt(NNt)
	KKi = NNi[:,None]*KKt;BBi = NNi[:,None]*BBt




	res = [];resI = []
	for ii in range(BBi.shape[-1]):
		if urConstraint is False and  divConstraint is False:
			knotConstraint = None
			res.append(RLSinversion_MCA(KKi,BBi[:,ii].real,alpha,Lmatrix=Lmatrix))
			resI.append(RLSinversion_MCA(KKi,BBi[:,ii].imag,alpha,Lmatrix=Lmatrix))
			continue
		elif urConstraint is not False and divConstraint is False:
			knotConstraintR = [BsplineCoeffs,urConstraint[ii].real]
			knotConstraintI = [BsplineCoeffs,urConstraint[ii].imag]
		else:
			knotConstraintR = [[BsplineCoeffs,dBsplineCoeffs],[urConstraint[ii].real,divConstraint[ii].real]]
			knotConstraintI = [[BsplineCoeffs,dBsplineCoeffs],[urConstraint[ii].imag,divConstraint[ii].imag]]

		res_tmp = RLSinversion_MCA(KKi,BBi[:,ii].real,alpha,Lmatrix=Lmatrix,knotConstraint=knotConstraintR,GaussianScale = Ncount[ii])
		resI_tmp = RLSinversion_MCA(KKi,BBi[:,ii].imag,alpha,Lmatrix=Lmatrix,knotConstraint=knotConstraintI,GaussianScale = Ncount[ii])


		res.append(res_tmp)
		resI.append(resI_tmp)

	res = NP.moveaxis(res,0,-1)
	resI = NP.moveaxis(resI,0,-1)

	if VERBOSE:
		print('Kernel size: ', KKi.shape)
		print('Inversion Time: ',time.time() - tini)



	return res + resI*1.j

	# outDir = OUTDIR  + '/QX%i/QY%i' % (QX,QY) 
	# mkdir_p(outDir)
	# with h5py.File(outDir + '/SIGMA%i.h5' % (SIGMA),'w') as h5f:
		# h5f.create_dataset('result',data = res+resI*1.j)
		# h5f.create_dataset('alpha',data = alpha)
	

	# return 1

nPad = 70
with NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/InversionData/avgSupergranuleData_Doppler.npz') as npyDICT:
	ur_Doppler = -NP.moveaxis(npyDICT['DopplerAverage'],0,-1) # negative because z is outward, but doppler inward
	xgrid = npyDICT['xgrid']
	ygrid = npyDICT['ygrid']
with NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/InversionData/avgSupergranuleData_LCT.npz') as npyDICT:
	div_LCT = NP.moveaxis(npyDICT['divAverage'],0,-1)
	xgrid = npyDICT['xgrid']
	ygrid = npyDICT['ygrid']

ur_Doppler_q = fft.fftshift(fft.fftn(ur_Doppler,axes=(0,1),norm='forward'),axes=(0,1))
ur_Doppler_q = ur_Doppler_q[nPad:len(xgrid)-nPad,nPad:len(ygrid)-nPad]
div_LCT_q = fft.fftshift(fft.fftn(div_LCT,axes=(0,1),norm='forward'),axes=(0,1))
div_LCT_q = div_LCT_q[nPad:len(xgrid)-nPad,nPad:len(ygrid)-nPad]


# # # # --------------------------------------------------------
# # # # #			Test Rountine
# # # # #---------------------------------------------------------
QX_ind = 11;QY_ind = 3;SIGMA_ind = 0
ur_constraint = ur_Doppler_q[NP.argmin(abs(QX - QX_ind)),NP.argmin(abs(QY - QY_ind))]
divh_constraint = div_LCT_q[NP.argmin(abs(QX - QX_ind)),NP.argmin(abs(QY - QY_ind))]

tini = time.time()
res_test,std_test = RunInversion(QX_ind,QY_ind,SIGMA_ind,[5e3,5e4],urConstraint=ur_constraint,divConstraint=divh_constraint,VERBOSE=True)
print('Inversion time = ',time.time()-tini)

plt.figure()
for ii in range(6):
	plt.plot(Basis1D.x_*1e-6,Basis1D.reconstructFromBasis(res_test.real[:,0,ii],axis=0),color='C%i' % ii)
	plt.fill_between(Basis1D.x_*1e-6,Basis1D.reconstructFromBasis(res_test.real[:,0,ii],axis=0)-Basis1D.reconstructFromBasis(std_test.real[:,0,ii],axis=0),Basis1D.reconstructFromBasis(res_test.real[:,0,ii],axis=0)+Basis1D.reconstructFromBasis(std_test.real[:,0,ii],axis=0),color='C%i' % ii,alpha=0.3)

plt.figure()
for ii in range(6):
	plt.errorbar(Basis1D.x_*1e-6,Basis1D.reconstructFromBasis(res_test.imag[:,0,ii],axis=0),yerr=Basis1D.reconstructFromBasis(std_test.imag[:,0,ii],axis=0))

for ii in range(len(percentiles)):
	surfaceValues = Basis1D.reconstructFromBasis(res_test.real[:,1,ii],axis=0,xFinal=NP.array([0])) + 1.j*Basis1D.reconstructFromBasis(res_test.imag[:,1,ii],axis=0,xFinal=NP.array([0]))
	surfaceValues_divh = -(Basis1D.reconstructFromBasis(res_test.real[:,1,ii],axis=0,xFinal=NP.array([0]),derivative=1) + 1.j*Basis1D.reconstructFromBasis(res_test.imag[:,1,ii],axis=0,derivative=1,xFinal=NP.array([0])))
	print(norm2(surfaceValues - ur_constraint[ii])/norm2(surfaceValues),norm2(surfaceValues_divh - divh_constraint[ii])/norm2(divh_constraint))


#--------------------------------------------------------
#		Run Inversions
#--------------------------------------------------------


qxgrid,qygrid = NP.meshgrid(QX,QY,indexing='ij')


if LM ==1:
	alphaGrid = NP.linspace(1,50,10)
elif LM ==2:
	alphaGrid = NP.linspace(1,50,10)*1e3
	# alphaGrid = NP.array([5e3,5e4])

tmp = reduce(RunInversion,(qxgrid.ravel(),qygrid.ravel(),0,alphaGrid,ur_Doppler_q.reshape(-1,len(percentiles)).T,div_LCT_q.reshape(-1,len(percentiles)).T),len(qxgrid.ravel()),25,progressBar=True)
res_total,std_total = tmp

res_total = NP.moveaxis(NP.moveaxis(res_total,-1,0).reshape(len(QX),len(QY),len(SIGMA),Basis1D.nbBasisFunctions_,len(alphaGrid),len(percentiles)),(-2,-1),(0,1))
std_total = NP.moveaxis(NP.moveaxis(std_total,-1,0).reshape(len(QX),len(QY),len(SIGMA),Basis1D.nbBasisFunctions_,len(alphaGrid),len(percentiles)),(-2,-1),(0,1))

with h5py.File(OUTDIR + '/SG_InversionSolutions%s_LM%i.h5' % (['','_FILT%i' % int(FILTER)][int(FILTER is not False)],LM),'w') as h5f:
	h5f.create_dataset('result',data = res_total)
	h5f.create_dataset('std',data = std_total)
	h5f.create_dataset('alpha',data = alphaGrid)
	h5f.create_dataset('QX',data = QX)
	h5f.create_dataset('QY',data = QY)
	h5f.create_dataset('SIGMA',data = SIGMA)
