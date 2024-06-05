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
nPad = 70

# subDir = '1Day'
# InversionData = NP.genfromtxt('data/InversionDataDir.txt',dtype=str)
# tinds = NP.arange(1,3)*1920

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

# abort

#-----------------Load Data --------------------------
# if 'KK' not in locals():
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
		ns = NP.ones(len(BBt))*nn
	else:
		KK = NP.concatenate([KK,KKt],axis=0)
		BB = NP.concatenate([BB,BBt],axis=0)
		NN = NP.append(NN,NNt) 
		ns = NP.append(ns,NP.ones(len(BBt))*nn) 
print(text_special('Inverting for |q|Rsun = %1.2f' % (NP.sqrt((QX*dkx)**2+(QY*dky)**2)*RSUN),'g',True,True))


del KKt,BBt,NNt


good_inds = (KK[:,0] != 0)*(BB[:,-1] != 0)*(NN != 0)*(~NP.isnan(NN))
KKt = KK[good_inds]
NNt = NN[good_inds]
BBt = BB[good_inds]
nst = ns[good_inds]



NNi = 1/NP.sqrt(NNt)
KKi = NNi[:,None]*KKt;BBi = NNi[:,None]*BBt

BBi = BBi[...,-1]
Ncount = Ncount[-1]


#-----------------------------------------------------------------------
#						RLS Inversion
#-----------------------------------------------------------------------
print(text_special('Computing RLS inversion','y',True,True))




INV = tikhonov(stopRule='Lcurve',N=51,VERBOSE=True)
res = RLSinversion_MCA(KKi,BBi.real,INV.alpha_,Lmatrix=Lmatrix)

# INV = tikhonov(stopRule='Lcurve',N=25,forceAlpha=[5,18],VERBOSE=True)

INV.update(KKi,BBi.real,L=Lmatrix)
xr,alphaXr = INV.invert(standardForm=True)

# INV.update(KKi,BBi.imag,L=Lmatrix)
# xi,alphaXi = INV.invert(standardForm=False)
SOLAdir = '/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/SOLA_Coeffs/' + '/QX_%i/QY_%i/' % (QX,QY)
with NP.load(SOLAdir + '/SOLA_coeffs_SIGMA%i.npz' % (SIGMA)) as npyDICT:
	z0 = npyDICT['TargetDepths']*1e-6
	coeffs = npyDICT['coeffs']
	BB_sola    = npyDICT['Bcoefs']

ALPHA = 5e3
alphaI = NP.argmin(abs(INV.alpha_ - ALPHA))

plt.figure(1)
plt.plot(INV.res_[alphaI],INV.normLx_[alphaI],'.k',mew = 3)
plt.legend(['L curve','auto-selected point','Manual selection point'])
# plt.figure()
# test = []
# for ii in range(200):
# 	# test.append(RLSinversion(KKi.imag,BBi.imag,INV.alpha_[ii],Lmatrix=Lmatrix,Lcurve=True))
# 	test.append(RLSinversion(KKi.imag,BBi.imag,ALPHA + NP.linspace(-1,1)*ALPHA,Lmatrix=Lmatrix,Lcurve=True))
# 	plt.loglog(test[-1][0],test[-1][1],'x')
# 	plt.text(test[-1][0],test[-1][1],'(%1.2e,%i)' % (INV.alpha_[ii],ii))
# test=NP.array(test)
# plt.loglog(test[:,0],test[:,1])

res = RLSinversion_MCA(KKi,BBi.real,INV.alpha_,Lmatrix=Lmatrix)
residual = RLSinversion_MCA(KKi,BBi.real,NP.logspace(-10,10,101),Lmatrix=Lmatrix,Lcurve=True)
ind = NP.argmin(abs(NP.logspace(-10,10,101) - ALPHA))


plt.figure()
plt.loglog(NP.logspace(-10,10,101),residual,lw=2,mew=15)
plt.plot(ALPHA,residual[ind],'xr')


with NP.load('data/SOLA_coeff_test_QX%i_QY%i.npz' % (QX,QY)) as npyDICT:
	z02 = npyDICT['TargetDepths']*1e-6
	coeffs2 = npyDICT['coeffs']
	sols = npyDICT['sols']
plt.figure()
plt.plot(Basis1D.x_*1e-6,Basis1D.reconstructFromBasis(xr[:,alphaI][:Basis1D.nbBasisFunctions_]),'b',label='Re')
plt.plot(Basis1D.x_*1e-6,Basis1D.reconstructFromBasis(res[:,alphaI][:Basis1D.nbBasisFunctions_]),'--r',label='test')
# plt.plot(z02,NP.dot(coeffs2,BB_sola),'.-k',label='SOLA test')
plt.plot(z02,sols,'.-r',label='SOLA test subsample')
# plt.plot(z0,NP.dot(coeffs,BB_sola),'.-g',label='SOLA')


with NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/InversionData/avgSupergranuleData_Doppler.npz') as npyDICT:
	ur_Doppler = -npyDICT['DopplerAverage'][-1]
	xgrid = npyDICT['xgrid']
	ygrid = npyDICT['ygrid']
ur_Doppler_q = fft.fftshift(fft.fftn(ur_Doppler,axes=(0,1),norm='forward'))
ur_Doppler_q = ur_Doppler_q[nPad:len(xgrid)-nPad,nPad:len(ygrid)-nPad]

surface_constraint = ur_Doppler_q[QX,QY]
surKnot = NP.argmin(abs(Basis1D.knots_[Basis1D.order_-1:-(Basis1D.order_-1)]))

A2 = NP.dot(KKi.T,KKi)
Ab = NP.dot(KKi.T,BBi.real)
L2 = NP.dot(Lmatrix.T,Lmatrix)

mat = A2 + INV.alpha_[alphaI]**2*L2

sol = NP.linalg.solve(mat,Ab)

mat2 = NP.pad(mat,((0,1),(0,1)),constant_values=((0,0),(0,0)))

BsplineCoeffs = []
for ii in range(Basis1D.nbBasisFunctions_):
	BsplineCoeffs.append(Basis1D(ii,x=NP.array([0])))
BsplineCoeffs = NP.squeeze(BsplineCoeffs)

mat2[-1,:-1] = BsplineCoeffs;mat2[:-1,-1] = BsplineCoeffs

sol2 = NP.linalg.solve(mat2,NP.append(Ab,surface_constraint.real))

res3 = RLSinversion_MCA(KKi,BBi.real,INV.alpha_,Lmatrix=Lmatrix,knotConstraint=[BsplineCoeffs,surface_constraint.real])

plt.figure()
plt.plot(Basis1D.x_*1e-6,Basis1D.reconstructFromBasis(sol),'k',label='sol without constraint')
plt.plot(Basis1D.x_*1e-6,Basis1D.reconstructFromBasis(sol2[:-1]),'r',label='sol with manual constraint')
plt.plot(Basis1D.x_*1e-6,Basis1D.reconstructFromBasis(res3[:,alphaI][:Basis1D.nbBasisFunctions_]),'--g',label='sol with inbuilt constraint')



# plt.plot(Basis1D.x_*1e-6,Basis1D.reconstructFromBasis(xi[:,alphaXi][:Basis1D.nbBasisFunctions_]),'r',label='Im')
plt.title('Poloidal')
plt.legend()

plt.figure()
plt.plot(BBi.real,label='Obs')
plt.plot(NP.dot(KKi,xr[:,alphaI]).real,label = 'Forward model')
plt.legend()
#----------------------------------------------------------------
#   			Build Synthetic model
#----------------------------------------------------------------
plt.figure()
# X0 = -5; SIG = 2;AMP = 0.15
X0 = -15; SIG = 6;AMP = -0.4 / [1,50][FILTER == 175]

synModelP = AMP*NP.exp(-(Basis1D.x_*1e-6 - X0)**2/(2*(SIG**2)))
synModelT = 0.01*synModelP
plt.plot(Basis1D.x_*1e-6,Basis1D.reconstructFromBasis(xr[:,alphaI][:Basis1D.nbBasisFunctions_]),'b',label='Re')
plt.plot(Basis1D.x_*1e-6,synModelP)
synBBi = NP.dot(KKi,Basis1D.projectOnBasis(synModelP))
synBBiN = synBBi.real+NP.random.randn(len(BBi))/NP.sqrt(Ncount)
plt.figure()
plt.plot(BBi.real)
plt.plot(synBBiN)
plt.plot(synBBi.real)


INV.update(KKi,synBBiN.real,L=Lmatrix)
xrS,alphaXrS = INV.invert(standardForm=True)

plt.figure(8)
plt.plot(INV.res_[alphaI],INV.normLx_[alphaI],'.k',mew = 3)
plt.legend(['L curve','auto-selected point','Manual selection point'])

plt.figure()
plt.plot(Basis1D.x_*1e-6,synModelP,'b',label='Synthetic model')




SOLAsol = [];RLSsol = []
for ii in range(500):
	# synBBiN_sola = synBBi.real+NP.random.randn(len(BBi))/NP.sqrt(Ncount)
	# SOLAsol.append(NP.dot(coeffs2,synBBiN_sola))
	# INV.update(KKi,synBBiN_sola.real,L=Lmatrix)
	xrST = RLSinversion_MCA(KKi,synBBiN_sola.real,INV.alpha_,Lmatrix=Lmatrix)
	RLSsol.append(Basis1D.reconstructFromBasis(xrST[:,alphaI][:Basis1D.nbBasisFunctions_]))

# plt.plot(z02,NP.mean(SOLAsol,axis=0),'.-k',label='SOLA')
# plt.fill_between(z02,NP.mean(SOLAsol,axis=0)-NP.std(SOLAsol,axis=0),NP.mean(SOLAsol,axis=0)+NP.std(SOLAsol,axis=0),color='k',alpha=0.3)
plt.plot(Basis1D.x_*1e-6,NP.mean(RLSsol,axis=0),'r',label='Inversion result')
plt.fill_between(Basis1D.x_*1e-6,NP.mean(RLSsol,axis=0)-NP.std(RLSsol,axis=0),NP.mean(RLSsol,axis=0)+NP.std(RLSsol,axis=0),color='r',alpha=0.3)


plt.legend()


plt.figure()
plt.plot(NP.dot(KKi,xr[:,alphaI]).real,label = 'Obs Forward model')
plt.plot(synBBi.real,label = 'Synthetic Forward model')
plt.legend()


# #---------------------------------------------------------------------
# #					RLS Averaging Kernel
# #---------------------------------------------------------------------
# A = KKi
# KKz = NP.dot(KKi,NP.linalg.inv(Basis1D.mass_))
# KKz = Basis1D.reconstructFromBasis(KKz,axis=-1)

# A2 = NP.dot(A.T,A)
# L2 = NP.dot(Lmatrix.T,Lmatrix)

# mat = NP.dot(NP.linalg.inv(A2 + 2e4**2*L2),A.T)
# matP = Basis1D.reconstructFromBasis(mat,axis=0)


# ind = NP.argmin(abs(Basis1D.x_ - -5e6))
# plt.plot(Basis1D.x_*1e-6,NP.dot(matP[ind],KKz));


import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("output_testingInversion_L%i%s.pdf" % (LM,BasisStr))
for fig in range(1, figure().number): ## will open an empty extra figure :(
    pdf.savefig( fig )
pdf.close()

plt.close('all')

abort
#-------------------------------------------------------------------
# 				SOLA INVERSION
#-------------------------------------------------------------------
print(text_special('Computing SOLA inversion','y',True,True))

KKiT = copy.copy(KKi)[::4]
NNiT = copy.copy(NNi)[::4]
alphaGrid = NP.logspace(-10,10,40)

# # tmp = SOLA_coeffCalc_MCA(KKiT,4e-7,Basis1D,-2e6*NP.ones(10),NP.linspace(0.1,3,10)*1e6,True)

# WidthIdeal = SOLA_coeffCalc_MCA(KKiT,4e-7,Basis1D,-10e6*NP.ones(25),NP.linspace(0.1,10,25)*1e6,False,True,NNiT)
# abort

target = NP.arange(-30,1)*1e6
widths = NP.linspace(0.1,10,25)*1e6

tgrid,wgrid = NP.meshgrid(target,widths,indexing = 'ij')
LcurveDetails = reduce(SOLA_coeffCalc_MCA,\
						(KKiT,alphaGrid,Basis1D,tgrid.ravel(),wgrid.ravel(),False,True,NNiT),\
						len(alphaGrid),min(len(alphaGrid),10),progressBar=True)

# LcurveDetails = LcurveDetails.squeeze()
# plt.plot(LcurveDetails[0],LcurveDetails[1],lw=2,color='b')
# plt.plot(LcurveDetails[0],LcurveDetails[1],'.k',mew=2)
for ii in range(LcurveDetails.shape[-1]):
	plt.text(LcurveDetails[0,ii],LcurveDetails[1,ii],'(%i,%1.2e)' % (ii,alphaGrid[ii]))

coeffs,KKz,Target = SOLA_coeffCalc_MCA(KKiT,4e-7,\
					Basis1D,-5e6,1e6,True,False)

# plt.figure()
# plt.plot(Basis1D.x_*1e-6,synModelP,'b',label='Synthetic model')
# plt.plot(Basis1D.x_*1e-6,Basis1D.reconstructFromBasis(xrS[:,alphaI][:Basis1D.nbBasisFunctions_]),'--r',label='RLS Inversion result')
# plt.plot(TargetDepths*1e-6,NP.dot(coeffs[:,:-1],synBBiN.real),'.k',mew=2,label='SOLA Inversion result')
# plt.legend()


