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
OUTDIR = '/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/SOLA_Coeffs/'

QX = int(sys.argv[1])
QY = int(sys.argv[2])
# QX = -26;QY = -6;
SIGMA = 0



Ngrid = NP.arange(8)

BASIS = 'Bspline'


#---------------------Load Basis----------------------

if BASIS == 'Bspline':
	BasisStr = '_Bspline'
else:
	BasisStr = ''
Basis1D    = NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/InversionData/Basis%s.npy' % (BasisStr),allow_pickle=True)[0]

print(text_special('Computing for %s' %['CHEBYSHEV','BSPLINE'][int(BASIS=='Bspline')],'g',True,True))


#-----------------Load Data --------------------------

print(text_special('Loading in Kernels and Bcoeffs','y'))
for ii in range(len(Ngrid)):
	nn = Ngrid[ii]; nnp = nn

	# ----------------------Load Kernels --------------------
	with h5py.File('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/Kernels/Kernels%s_n%i_np%i.h5' % (BasisStr,nn,nnp),'r') as h5f:
		# QX,QY = [NP.array(h5f[x]) for x in ['QX','QY']]
		KKt = NP.array(h5f['Kernels']['QX%i' % QX]['QY%i' % QY]['data'])
		if TOROIDAL:
			KKt = KKt.reshape(-1,KKt.shape[-1]).T
		else:
			KKt = KKt[0].T

	# ---------------Load Bcoeffs and noise ----------------
	with h5py.File('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/Bcoeffs_AVG' + '/Bcoeffs_n%i_np%i%s.h5' % (nn,nnp,['','_FILT%i' % int(FILTER)][int(FILTER is not False)]),'r') as h5f:
		xgrid,ygrid,Ncount,dkx,dky = [NP.array(h5f[x]) for x in ['xgrid','ygrid','NumberSGs','dkx','dky']]
		BBt = NP.array(h5f['Bcoeffs_avgSG']['QX%i' % QX]['QY%i' % QY]['SIGMA%i' % SIGMA]['data'])
		NNt = NP.array(h5f['NoiseModel']['QX%i' % QX]['QY%i' % QY]['SIGMA%i' % SIGMA]['data'])
		kxt,kyt = [NP.array(h5f[x]) for x in ['kxm','kym']]


	# --------------- Determine the useful modes ----------------

	good_inds_tmp = (KKt[:,0] != 0)*(BBt != 0)*(NNt != 0)*(~NP.isnan(NNt))

	KKt = KKt[good_inds_tmp]
	BBt = BBt[good_inds_tmp]
	NNt = NNt[good_inds_tmp]
	kxt = kxt[good_inds_tmp]
	kyt = kyt[good_inds_tmp]

	# --------------- Shuffle the modes to get a nice spread in kx,ky ----------------


	inds_tmp = NP.arange(len(BBt))
	NP.random.shuffle(inds_tmp)

	if nn < 5:
		Ncut = 2000
	else:
		Ncut = 1000


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



	# ---------------Get the ones used in the SOLA Inversion ----------------

	if ii == 0:
		KK = KKt
		BB = BBt
		NN = NNt
		ns = NP.ones(BBt.shape[-1])*nn
		kxm = kxt
		kym = kyt

		inds_interp = inds_tmp[:Ncut]

		KK_missing = KKt_missing
		BB_missing = BBt_missing
		NN_missing = NNt_missing
		ns_missing = NP.ones(BBt_missing.shape[-1])*nn
		kx_missing = kxt_missing
		ky_missing = kyt_missing

		inds_missing = inds_tmp[Ncut:]
	else:
	# ---------------Collect the ones not used in the SOLA Inversion, but to be interpolated ----------------

		KK = NP.concatenate([KK,KKt],axis=0)
		BB = NP.append(BB,BBt) 
		NN = NP.append(NN,NNt) 
		ns = NP.append(ns,NP.ones(BBt.shape[-1])*nn) 
		kxm = NP.append(kxm,kxt)
		kym = NP.append(kym,kyt)

		KK_missing = NP.concatenate([KK_missing,KKt_missing],axis=0)
		BB_missing = NP.append(BB_missing,BBt_missing) 
		NN_missing = NP.append(NN_missing,NNt_missing) 
		ns_missing = NP.append(ns_missing,NP.ones(BBt_missing.shape[-1])*nn) 
		kx_missing = NP.append(kx_missing,kxt_missing)
		ky_missing = NP.append(ky_missing,kyt_missing)

		inds_interp = NP.append(inds_interp,inds_tmp[:Ncut])
		inds_missing = NP.append(inds_missing,inds_tmp[Ncut:])
print(text_special('Inverting for |q|Rsun = %1.2f' % (NP.sqrt((QX*dkx)**2+(QY*dky)**2)*RSUN),'g',True,True))


del KKt,BBt,NNt


# --------------- Scale the Kernels and Bcoeffs by the noise model ----------------

NNi = 1/NP.sqrt(NN)
KKi = NNi[:,None]*KK;BBi = NNi*BB

NNi_missing = 1/NP.sqrt(NN_missing)
KKi_missing = NNi_missing[:,None]*KK_missing;BBi_missing = NNi_missing*BB_missing


#-------------------------------------------------------------------
# 				SOLA INVERSION
#-------------------------------------------------------------------
print(text_special('Computing SOLA inversion','y',True,True))

#--------- Select the target depths --------------------------
target = NP.linspace(-35,0,100)*1e6

#--------- Choose the target widths ---------------------------

cs = NP.load('/scratch/ch3246/OBSDATA/gyreresult/eigenfunctions_combined/eigs%02d.npz' % 2)['cs']*1e3

idealWidths = []
zh = -1e6;wh = 0.75e6;ind0 = NP.argmin(abs(Basis1D.x_ - zh))
# We select a good width at some depth then scale by cs[z]/cs[z0]
for ii in range(len(target)):
	ind = NP.argmin(abs(Basis1D.x_ - target[ii]))
	idealWidths.append(wh * (cs[ind]/cs[ind0])**1.2)
idealWidths = NP.array(idealWidths)



#----------Perform the SOLA inversion--------------------------
coeffs,KKz,Targets = SOLA_coeffCalc_MCA(KKi,10,Basis1D,target,idealWidths,True,SVDmethod=True,rcond=1e-5)


#----------Build K(z) for the used and interpolated modes-------------------------------
KKiSOLA = NP.dot(KKi[:,:Basis1D.nbBasisFunctions_],NP.linalg.inv(Basis1D.mass_))
KKz_used = Basis1D.reconstructFromBasis(KKiSOLA,axis=-1)
KKiSOLA = NP.dot(KKi_missing[:,:Basis1D.nbBasisFunctions_],NP.linalg.inv(Basis1D.mass_))
KKz_missing = Basis1D.reconstructFromBasis(KKiSOLA,axis=-1)



#----------For each n, interpolate coeffs across the kx ky grid ------------------------
coeffs_total = NP.zeros((len(target),1));KKz_total = [NP.zeros(len(Basis1D.x_))];
BB_total = NP.array([0]);ns_total = NP.array([999])
for nn in arange(8):
	inds = ns == nn
	inds_mis = ns_missing == nn
	# print(sum(inds_mis))
	if sum(inds_mis) == 0:
		coeffs_total = NP.append(coeffs_total,coeffs[:,inds],axis=1)
		KKz_total = NP.append(KKz_total,KKz_used[inds],axis=0)
		BB_total = NP.append(BB_total,BBi[inds])
		ns_total = NP.append(ns_total,NP.ones(len(BBi[inds]))*nn)
		continue

	points = (kxm[inds]*RSUN,kym[inds]*RSUN)
	points_interp = (kx_missing[inds_mis]*RSUN,ky_missing[inds_mis]*RSUN)

	inds_sort  = NP.argsort(NP.append(inds_interp[inds],inds_missing[inds_mis]))

	KKz_interp = KKz_used[inds]
	KKz_mis    = KKz_missing[inds_mis]
	KKz_tmp    = NP.concatenate([KKz_interp,KKz_mis])[inds_sort]

	BB_tmp = NP.append(BBi[inds],BBi_missing[inds_mis])[inds_sort]

	# ---------------For each depth perform the interpolation----------------
	coeffs_total_depth = []

	# coeffs_tmp_lstsq = NP.linalg.lstsq(KKz_mis.T,NP.dot(coeffs[:,inds],KKz_interp).T)[0]

	for depth in range(len(coeffs)):

		coeffs_tmp = scipy.interpolate.griddata(points, coeffs[depth][inds], points_interp, method='linear',fill_value=0)
		coeffs_tmp = NP.append(coeffs[depth][inds],coeffs_tmp)

		# coeffs_tmp = NP.append(coeffs[depth][inds],coeffs_tmp_lstsq[:,depth])
		coeffs_tmp = coeffs_tmp[inds_sort]

		coeffs_total_depth.append(coeffs_tmp * simps(abs(NP.dot(coeffs[depth][inds],KKz_interp)),x=Basis1D.x_) / simps(abs(NP.dot(coeffs_tmp,KKz_tmp)),x=Basis1D.x_))

	coeffs_total = NP.append(coeffs_total,coeffs_total_depth,axis=1)
	KKz_total = NP.append(KKz_total,KKz_tmp,axis=0)
	BB_total = NP.append(BB_total,BB_tmp)
	ns_total = NP.append(ns_total,NP.ones(len(KKz_tmp))*nn)



#----------- remove the null row due wanting to use the NP.append---------------------
coeffs_total = coeffs_total[:,1:]
KKz_total = KKz_total[1:]
BB_total = BB_total[1:]
ns_total = ns_total[1:]

#------------ Save the coefficients and the averaging kernels ------------------------

SAVEDIR = OUTDIR + '/QX_%i/QY_%i/' % (QX,QY)

mkdir_p(SAVEDIR)

NP.savez_compressed(SAVEDIR + '/SOLA_coeffs_SIGMA%i_wider.npz' % (SIGMA),\
						coeffs = coeffs_total,\
						KKz    = NP.dot(coeffs_total,KKz_total),\
						KKz_og = NP.dot(coeffs,KKz),\
						Bcoefs     = BB_total,\
						TargetDepths = target,\
						TargetFunc   = Targets,\
						inds_n = inds_interp,\
						ns = ns)
