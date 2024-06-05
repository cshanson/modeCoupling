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

# subDir = '1Day'
# InversionData = NP.genfromtxt('data/InversionDataDir.txt',dtype=str)
# tinds = NP.arange(1,3)*1920

# Ngrid = NP.arange(8)
# Npgrid = copy.copy(Ngrid)
Ngrid = NP.concatenate([NP.arange(8),NP.arange(2,6),NP.arange(2,4)])
Npgrid = NP.concatenate([NP.arange(8),NP.arange(2,6)+1,NP.arange(2,4)+2])

BASIS = 'Bspline'
LM    = 1 #Order of the Lmatrix


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
		kxt,kyt = [NP.array(h5f[x]) for x in ['kxm','kym']]

	good_inds_tmp = (KKt[:,0] != 0)*(BBt != 0)*(NNt != 0)*(~NP.isnan(NNt))

	KKt = KKt[good_inds_tmp]
	BBt = BBt[good_inds_tmp]
	NNt = NNt[good_inds_tmp]
	kxt = kxt[good_inds_tmp]
	kyt = kyt[good_inds_tmp]

	inds_tmp = NP.arange(len(BBt))
	NP.random.shuffle(inds_tmp)

	# if nn < 5:
		# Ncut = 2000
	# else:
	Ncut = 1500


	KKt_missing = KKt[inds_tmp[Ncut:]]
	BBt_missing = BBt[inds_tmp[Ncut:]]
	NNt_missing = NNt[inds_tmp[Ncut:]]
	kxt_missing = kxt[inds_tmp[Ncut:]]
	kyt_missing = kyt[inds_tmp[Ncut:]]

	KKt = KKt[inds_tmp[:Ncut]]
	BBt = BBt[inds_tmp[:Ncut]]
	NNt = NNt[inds_tmp[:Ncut]]
	kxt = kxt[inds_tmp[:Ncut]]
	kyt = kyt[inds_tmp[:Ncut]]



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
		nps = NP.ones(BBt.shape[-1])*nnp
		kxm = kxt
		kym = kyt

		inds_interp = inds_tmp[:Ncut]

		KK_missing = KKt_missing
		BB_missing = BBt_missing
		NN_missing = NNt_missing
		ns_missing = NP.ones(BBt_missing.shape[-1])*nn
		nps_missing = NP.ones(BBt_missing.shape[-1])*nnp
		kx_missing = kxt_missing
		ky_missing = kyt_missing

		inds_missing = inds_tmp[Ncut:]
	else:
		KK = NP.concatenate([KK,KKt],axis=0)
		BB = NP.append(BB,BBt) 
		NN = NP.append(NN,NNt) 
		ns = NP.append(ns,NP.ones(BBt.shape[-1])*nn)
		nps = NP.append(nps,NP.ones(BBt.shape[-1])*nnp) 
		kxm = NP.append(kxm,kxt)
		kym = NP.append(kym,kyt)

		KK_missing = NP.concatenate([KK_missing,KKt_missing],axis=0)
		BB_missing = NP.append(BB_missing,BBt_missing) 
		NN_missing = NP.append(NN_missing,NNt_missing) 
		ns_missing = NP.append(ns_missing,NP.ones(BBt_missing.shape[-1])*nn) 
		nps_missing = NP.append(nps_missing,NP.ones(BBt_missing.shape[-1])*nnp) 
		kx_missing = NP.append(kx_missing,kxt_missing)
		ky_missing = NP.append(ky_missing,kyt_missing)

		inds_interp = NP.append(inds_interp,inds_tmp[:Ncut])
		inds_missing = NP.append(inds_missing,inds_tmp[Ncut:])
print(text_special('Inverting for |q|Rsun = %1.2f' % (NP.sqrt((QX*dkx)**2+(QY*dky)**2)*RSUN),'g',True,True))


del KKt,BBt,NNt


# good_inds = (KK[:,0] != 0)*(BB != 0)*(NN != 0)*(~NP.isnan(NN))
# KKt = KK[good_inds]
# NNt = NN[good_inds]
# BBt = BB[good_inds]
# nst = ns[good_inds]
# kxm = kxm[good_inds]
# kym = kym[good_inds]
# NNi = 1/NP.sqrt(NNt)
# KKi = NNi[:,None]*KKt;BBi = NNi*BBt


NNi = 1/NP.sqrt(NN)
KKi = NNi[:,None]*KK;BBi = NNi*BB

NNi_missing = 1/NP.sqrt(NN_missing)
KKi_missing = NNi_missing[:,None]*KK_missing;BBi_missing = NNi_missing*BB_missing


#-------------------------------------------------------------------
# 				SOLA INVERSION
#-------------------------------------------------------------------
print(text_special('Computing SOLA inversion','y',True,True))

KKiT = copy.copy(KKi)
NNiT = copy.copy(NNi)
alphaGrid = NP.logspace(-10,10,40)

target = NP.linspace(-30,0,60)*1e6
widths = NP.linspace(0.1,10,25)*1e6

COMPUTE = False
if COMPUTE:



	tgrid,wgrid = NP.meshgrid(target,widths,indexing = 'ij')
	LcurveDetails = reduce(SOLA_coeffCalc_MCA,\
							(KKiT,alphaGrid,Basis1D,tgrid.ravel(),wgrid.ravel(),False,True,NNiT),\
							len(alphaGrid),min(len(alphaGrid),10),progressBar=True)

	LcurveDetails = NP.moveaxis(LcurveDetails,-1,1).reshape((2,len(alphaGrid)) + tgrid.shape)

	NP.savez_compressed('data/SOLA_details.npz',\
						results = LcurveDetails,\
						alphaGrid = alphaGrid,\
						targetDepths = target,\
						widths = widths)

else:
	with NP.load('data/SOLA_details.npz') as npyDICT:
		LcurveDetails = npyDICT['results']


alphaInd = 7

# plt.ioff()

# for jj in NP.arange(len(target)-1,0,-1):
# 	plt.figure()
# 	for ii in range(len(alphaGrid)):
# 		plt.semilogy(widths*1e-6,LcurveDetails[1,ii,jj,:],'.-',label = '%1.2e' % (alphaGrid[ii]))
# 	plt.legend()
# 	plt.title(r'$z_0=%1.2fMm$' % (target[jj]*1e-6))

# import matplotlib.backends.backend_pdf
# pdf = matplotlib.backends.backend_pdf.PdfPages("SOLA_Lcurves.pdf" )
# for fig in range(1, figure().number): ## will open an empty extra figure :(
#     pdf.savefig( fig )
# pdf.close()

# plt.close('all')

# plt.ion()


# Emperically  identified from SOLA.pdf
idealWidths = (target*1e-6 * 0.2589 + 0.68)*1e6



cs = NP.load('/scratch/ch3246/OBSDATA/gyreresult/eigenfunctions_combined/eigs%02d.npz' % 2)['cs']*1e3

idealWidths = []
zh = -1e6;wh = 0.75e6;ind0 = NP.argmin(abs(Basis1D.x_ - zh))
for ii in range(len(target)):
	ind = NP.argmin(abs(Basis1D.x_ - target[ii]))
	idealWidths.append(wh * (cs[ind]/cs[ind0])**1.2)
idealWidths = NP.array(idealWidths)


# coeffs,KKz,Targets = SOLA_coeffCalc_MCA(KKiT,alphaGrid[alphaInd],Basis1D,target,idealWidths,True)
# coeffs,KKz,Targets = SOLA_coeffCalc_MCA(KKi[::2],alphaGrid[alphaInd],Basis1D,target,idealWidths,True,SVDmethod=True,rcond=1e-6)
coeffs,KKz,Targets = SOLA_coeffCalc_MCA(KKi,alphaGrid[alphaInd],Basis1D,target,idealWidths,True,SVDmethod=True,rcond=1e-5)


KKiSOLA = NP.dot(KKi[:,:Basis1D.nbBasisFunctions_],NP.linalg.inv(Basis1D.mass_))
KKz_used = Basis1D.reconstructFromBasis(KKiSOLA,axis=-1)
KKiSOLA = NP.dot(KKi_missing[:,:Basis1D.nbBasisFunctions_],NP.linalg.inv(Basis1D.mass_))
KKz_missing = Basis1D.reconstructFromBasis(KKiSOLA,axis=-1)


coeffs_total = NP.zeros((len(target),1));KKz_total = [NP.zeros(len(Basis1D.x_))];ns_total = []
for ii in range(len(Ngrid)):
	inds = (ns == Ngrid[ii])*(nps == Npgrid[ii])
	inds_mis = (ns_missing == Ngrid[ii])*(nps_missing == Npgrid[ii])
	print(sum(inds_mis))
	if sum(inds_mis) == 0:
		coeffs_total = NP.append(coeffs_total,coeffs[:,inds],axis=1)
		KKz_total = NP.append(KKz_total,KKz_used[inds],axis=0)
		continue

	points = (kxm[inds]*RSUN,kym[inds]*RSUN)
	points_interp = (kx_missing[inds_mis]*RSUN,ky_missing[inds_mis]*RSUN)

	inds_sort  = NP.argsort(NP.append(inds_interp[inds],inds_missing[inds_mis]))

	KKz_interp = KKz_used[inds]
	KKz_mis    = KKz_missing[inds_mis]
	KKz_tmp    = NP.concatenate([KKz_interp,KKz_mis])[inds_sort]# * sum(inds)/sum(inds_mis)

	coeffs_total_depth = []
	for depth in range(len(coeffs)):

		coeffs_tmp = scipy.interpolate.griddata(points, coeffs[depth][inds], points_interp, method='linear',fill_value=0)

		coeffs_tmp = NP.append(coeffs[depth][inds],coeffs_tmp)
		coeffs_tmp = coeffs_tmp[inds_sort]

		coeffs_total_depth.append(coeffs_tmp * simps(abs(NP.dot(coeffs[depth][inds],KKz_interp)),x=Basis1D.x_) / simps(abs(NP.dot(coeffs_tmp,KKz_tmp)),x=Basis1D.x_))

	coeffs_total = NP.append(coeffs_total,coeffs_total_depth,axis=1)
	KKz_total = NP.append(KKz_total,KKz_tmp,axis=0)

	if len(ns_total) == 0:
		ns_total = NP.ones(len(KKz_tmp))*Ngrid[ii]
	else:
		ns_total = NP.append(ns_total,NP.ones(len(KKz_tmp))*Ngrid[ii])


coeffs_total = coeffs_total[:,1:]
KKz_total = KKz_total[1:]

NP.savez_compressed('data/SOLA_coeff_test_QX%i_QY%i.npz' % (QX,QY),\
						coeffs = coeffs_total,\
						sols = NP.dot(coeffs,BBi),\
						KKz    = NP.dot(coeffs_total,KKz_total),\
						KKz_og = NP.dot(coeffs,KKz),\
						TargetDepths = target,\
						TargetFunc   = Targets)
