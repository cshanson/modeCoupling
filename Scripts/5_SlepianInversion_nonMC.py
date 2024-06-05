#!/scratch/ch3246/project/PythonEnv/miniconda3/envs/dalma-python3-CPU/bin/python

# Routine to perform inversion
# VERY VERY MEMORY INTENSIVE. USE HPC!

pathToRoutines = '/home/ch3246/mps_montjoie/'
import sys
sys.path.insert(0,pathToRoutines)
print("Importing pythonRoutines")
import numpy as NP
from pyCompHelio import *
from matplotlib.pyplot import *
from astropy.io import fits
import matplotlib.gridspec as gridspec

plt.ion()
plt.close('all')

# Solves AX = b

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------


# Use pre computed A matrix (computed here and saved if False) 
usePreLoadAmatrix = True

# Padding for real space calc
nPad = 70

# Do you want to scale the L matrix by rho
SCALED = True

# Ignore the constraints?
IGNORE_UR_constraint = False

# Slepian contour size
SlepContour = 60 #Mm

# Invert for
Measurement = 'FLOW'

# Which flow model?
FlowModel = 10

# Outflows or inflows
OUTFLOWS = True

# Which basis and Lmatrix order
BASIS = 'Bspline'
LM    = 2 #Order of the Lmatrix

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# Load in the Basis and Slepian functions
if BASIS == 'Bspline':
	BasisStr = '_Bspline'
Basis1D    = NP.load('../../OUTPUTDIR//Basis%s.npy' % (BasisStr),allow_pickle=True)[0]
SlepianDict = scipy.io.loadmat('Slepian_github/Slepian_%iMm.mat' % SlepContour)
Lambda_alpha = SlepianDict['V'].squeeze()

CoeffsKeep = int(NP.ceil(NP.sum(Lambda_alpha)))
Slepians = SlepianDict['G'][...,:CoeffsKeep]


# Load in the background parameters
rho = NP.load('../..//eigenfunctions_combined/eigs%02d.npz' % 2)['rho']
z_tmp = NP.load('../..//eigenfunctions_combined/eigs%02d.npz' % 2)['z']


# Load in some Bcoef information
with h5py.File(DATADIR + '/Bcoeffs_n%i_np%i%s.h5' % (0,0,['','_cs'][int(Measurement.upper() == 'SOUNDSPEED')]),'r') as h5f:
	xgrid,ygrid,Ncount,dkx,dky,QX,QY = [NP.array(h5f[x]) for x in ['xgrid','ygrid','NumberSGs','dkx','dky','QX','QY']]
absX = NP.sqrt(xgrid**2+ygrid**2)


# Load in the ur
with NP.load(DATADIR + '/avgSupergranuleData_Doppler.npz') as npyDICT:
	ur_Doppler = -NP.moveaxis(npyDICT['DopplerAverage'],0,-1)[...,-1] # negative because z is outward, but doppler inward
	xgridD = npyDICT['xgrid']
	ygridD = npyDICT['ygrid']


#-----------------Build Smoothness Matrix--------------
if SCALED:
	Lmatrix0  = Basis1D.createSmoothnessMatrix(0,scaling=rho)
else:
	Lmatrix0  = Basis1D.createSmoothnessMatrix(0)
Lmatrix0  = Lmatrix0 / NP.amax(Lmatrix0)
Lmatrix1  = Basis1D.createSmoothnessMatrix(1)
Lmatrix1  = Lmatrix1 / NP.amax(Lmatrix1)
Lmatrix2   = Basis1D.createSmoothnessMatrix(LM)
Lmatrix2   = Lmatrix2/NP.amax(Lmatrix2)

# If flow, let's build constrains
if Measurement.upper() == 'FLOW':


	# Load in the hor and vertical flow maps
	with NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/InversionData/LCT/avgSupergranuleData_LCT%s.npz' % (['_INFLOW',''][int(OUTFLOWS is not False)])) as npyDICT:
		uxy_LCT = NP.moveaxis(npyDICT['LCTAverage'],0,-1)[...,FlowModel]
		div_LCT = NP.moveaxis(npyDICT['divAverage'],0,-1)[...,FlowModel]
	with NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/InversionData/avgSupergranuleData_DopplerLCT%s.npz' % (['_INFLOW',''][int(OUTFLOWS is not False)])) as npyDICT:
		ur_Doppler = -NP.moveaxis(npyDICT['DopplerAverage'],0,-1)[...,FlowModel] # negative because z is outward, but doppler inward

	# Vector with flow constraint maps
	flows_obs = NP.array([uxy_LCT[0],uxy_LCT[1],ur_Doppler])
	# Project on Slepians
	constraint_values = simps(simps(flows_obs[...,None]*Slepians[None,...],x=xgridD[:,0]/amax(xgridD),axis=1),x=ygridD[0,:]/amax(ygridD),axis=1)
	
	# Build mass matrix
	MassMatrix = NP.zeros((Slepians.shape[-1],Slepians.shape[-1]))
	for ii in range(Slepians.shape[-1]):
		for jj in range(Slepians.shape[-1]):
			MassMatrix[ii,jj] = simps(simps(Slepians[...,ii]*Slepians[...,jj],x=xgridD[:,0]/amax(xgridD),axis=0),x=ygridD[0,:]/amax(ygridD),axis=0)
	constraint_values = NP.dot(constraint_values,NP.linalg.inv(MassMatrix))
	
	# Reconstruct and check with original maps
	recon = NP.sum(Slepians * constraint_values[:,None,None,:],axis=-1)

	#-----------------Load some bacground parameters---------
	print(text_special('Smoothing rho','r'))
	rho = NP.exp(Basis1D.reconstructFromBasis(Basis1D.projectOnBasis(NP.log(rho))))
	dz = FDM_Compact(z_tmp)
	drhodz = dz.Compute_derivative(NP.log(rho)).real
	drhodz_surf = dz.Compute_derivative(NP.log(rho))[NP.argmin(abs(z_tmp))].real


	# The bcoefficients on which we want to apply the constraints
	BsplineCoeffs = [];dBsplineCoeffs = []
	for ii in range(Basis1D.nbBasisFunctions_):
		BsplineCoeffs.append(Basis1D(ii,x=NP.array([0])))
		dBsplineCoeffs.append(Basis1D(ii,x=NP.array([0]),derivative=1) + drhodz_surf*Basis1D(ii,x=NP.array([0])))
		# print(Basis1D(ii,x=NP.array([0]),derivative=1), drhodz_surf)
	BsplineCoeffs = NP.squeeze(BsplineCoeffs)
	dBsplineCoeffs = -NP.squeeze(dBsplineCoeffs) #Negative because -q^2 dz(P) = divh(u) 

	constraint_basisCoeffs = []
	for ii in range(Slepians.shape[-1]):
		tmp = NP.zeros(Slepians.shape[-1])
		tmp[ii] = 1
		constraint_basisCoeffs.append(NP.reshape(BsplineCoeffs[:,None] * tmp[None,:],-1))
	constraint_basisCoeffs = NP.array(constraint_basisCoeffs)
	constraint_basisCoeffs = NP.block([[constraint_basisCoeffs,NP.zeros(constraint_basisCoeffs.shape),NP.zeros(constraint_basisCoeffs.shape)],\
										[NP.zeros(constraint_basisCoeffs.shape),constraint_basisCoeffs,NP.zeros(constraint_basisCoeffs.shape)],\
										[NP.zeros(constraint_basisCoeffs.shape),NP.zeros(constraint_basisCoeffs.shape),constraint_basisCoeffs]])


	if IGNORE_UR_constraint:
		print(text_special('Ignoring ur constraint in solar inversion','y',True,True))
		constraint_basisCoeffs = constraint_basisCoeffs[:2*len(constraint_basisCoeffs)//3]
		constraint_values = constraint_values[:2]

	constraint_values = constraint_values.reshape(-1)

global Kernel


# Load in and build the uiltimate A matrix
if 'Kernel' not in locals():
	print(text_special('Loading data','g'))

	Kernel = []

	# Load in the kernels and Bcoeffs of interest
	for Measurement in ['UX','UY','UZ']:
		with h5py.File('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/Kernels/SlepianKernels_%s_%s_%iMm.h5' % (BASIS,Measurement.upper(),SlepContour),'r') as h5f:
			KKt = NP.array(h5f['Kernels'])
			if Measurement == 'UX':
				BBt = NP.array(h5f['Bcoeffs'])#[...,FlowModel]
				NNb = NP.array(h5f['NoiseModel'])
				NNt = NP.array(h5f['Bcoeffs_var'])/Ncount[-1]

				QXinds = NP.array(h5f['QXinds'])
				QYinds = NP.array(h5f['QYinds'])
				Nx     = NP.array(h5f['Nx'])
				SlepiansCoeffs = (NP.array(h5f['SlepianCoeffs'])[...,:]).astype('complex64')/Nx**2# FFT Normalization 
				radial_orders   = NP.array(h5f['radial_order'])
				radial_orders_p = NP.array(h5f['radial_order_p'])

		NNt = NP.where(NNt <= 1e-10/Ncount[FlowModel],1e10,NNt)
		NN = 1/NP.sqrt(NNt)
		Kernel.append((NN[:,None] * KKt).astype('complex64'));
		

		if Measurement == 'UX':
			BB = NN[:,None] * BBt;
			BB = BB.astype('complex64')

	Kernel = NP.array(Kernel)
	KKq = copy.copy(Kernel)

	# Build Slepian kernels	
	print('Computing Slepian Kernel')
	Kernel = (Kernel[...,None] * SlepiansCoeffs[None,:,None,:])	
	Kernel = Kernel.reshape((Kernel.shape[0],Kernel.shape[1],-1))
	Kernel = NP.concatenate(Kernel,axis=-1)

	# If already available load in A matrix
	if usePreLoadAmatrix:
		print('Loading Amatrix')
		Amatrix = NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/InversionData/Amatrix_%iMm.npz' % SlepContour)['Amatrix']
	else:
		# Build A martix VERY HEAVY
		print('Computing KKT')
		KKT = Kernel.T.conj()

		print('Computing Amatrix')

		Amatrix = NP.dot(KKT,Kernel)#.real
		del KKT
		NP.savez_compressed('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/InversionData/Amatrix_%iMm.npz' % SlepContour,Amatrix = Amatrix)

	# Build b matrix
	print('Computing bmatrix')

	BBH = NP.conj(BB).T
	bmatrix = NP.conj(NP.dot(BBH,Kernel)).T
	print('Loading complete')


Slepians = Slepians[...,:SlepiansCoeffs.shape[-1]]

# Build L matrix, has L2 andsmall L0 component
print(text_special('CAREFUL OF RHO SCALING IN L MATRIX','y'))
if SCALED:
	FACTOR = 1e-4
else:
	FACTOR = 1e-5
Lmatrix = Lmatrix2 + 0*Lmatrix1+ FACTOR*Lmatrix0 #+ 1e-3*NP.eye(len(Lmatrix))


# Since we are solving for three components, build block of L matrix
Lscaling = 1
Lmatrix_tmp = NP.kron(Lmatrix,MassMatrix)
Lmatrix_tmp_hor = Lscaling * NP.kron(Lmatrix2 + FACTOR*Lmatrix0,MassMatrix)

zerosBlock = NP.zeros(Lmatrix_tmp.shape)

Lmatrix_total = NP.block([[Lmatrix_tmp_hor,zerosBlock,zerosBlock],\
							[zerosBlock,Lmatrix_tmp_hor,zerosBlock],\
							[zerosBlock,zerosBlock,Lmatrix_tmp]])

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------


# -----------------------Compute the L curve ------------------------------
# COmpute the L curve for the opulation outlined by FlowModel

# Initialize some matriced
Amatrix2 = copy.copy(Amatrix)
L2matrix = copy.copy(Lmatrix_total)
bmatrix2 = copy.copy(bmatrix)[:,FlowModel]
OBS = BB[:,FlowModel]


# Apply Constraints
nParams = len(Amatrix)
Amatrix2 = NP.pad(Amatrix2,((0,len(constraint_values)),(0,len(constraint_values))),constant_values=((0,0),(0,0)))
for cc in range(len(constraint_values)):
	Amatrix2[nParams+cc,:nParams] = constraint_basisCoeffs[cc];Amatrix2[:nParams,nParams+cc] = constraint_basisCoeffs[cc]

L2matrix = NP.pad(Lmatrix_total,((0,len(constraint_values)),(0,len(constraint_values))),constant_values=((0,0),(0,0)))
bmatrix2 = NP.append(bmatrix2,constraint_values)


# Compute the L curve using the regularization grid alphaGrid
sols_list = [];		residual = [];normLx = []
alphaGrid = NP.logspace(-2,12,21)


print('Computing L curve')


PG = progressBar(len(alphaGrid),'serial')
for ii in range(len(alphaGrid)):
	LHS = Amatrix2 + alphaGrid[ii] * L2matrix
	sols_tmp = NP.linalg.solve(LHS,bmatrix2.real).astype('complex64')

	# Remove the lagrange multiplier
	sols_tmp = sols_tmp[:-len(constraint_values)]

	# Store solutions
	sols_list.append(sols_tmp)

	# Compute mismatch and residual
	mismatch = NP.dot(Kernel,sols_tmp)
	mismatch = mismatch - OBS
	residual.append((mismatch.conj().T @ mismatch).real)
	normLx.append((sols_list[-1][:,None].T@Lmatrix_total@sols_list[-1][:,None]).squeeze())
	PG.update()
del PG


# Plot the L curve
residual= NP.array(residual,dtype=float)
normLx= NP.array(normLx,dtype=float)


plt.figure()
plt.loglog(residual,normLx)
for ii in range(len(alphaGrid)):
	plt.plot(residual[ii],normLx[ii],'x')
	plt.text(residual[ii],normLx[ii],'%i,%1.2e' % (ii,alphaGrid[ii]))
plt.savefig('FIGURES/Lcurves/Lcurve%s_FlowGroup%i.png' % (['_INFLOWS',''][int(OUTFLOWS)],FlowModel))


#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# Invert for all populations


#---------------------Compute all sizes-----------------------
sols_list_size = [];residual_size = []; normLx_size = [];recon_size = []
print('Computing for different size supergranules')
PG = progressBar(len(Ncount),'serial')
for nSize in range(len(Ncount)):

	#---------------------------------------------------------------------

	# Compute constraints for each pop group

	with NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/InversionData/LCT/avgSupergranuleData_LCT%s.npz' % (['_INFLOW',''][int(OUTFLOWS is not False)])) as npyDICT:
		uxy_LCT = NP.moveaxis(npyDICT['LCTAverage'],0,-1)[...,nSize]
		div_LCT = NP.moveaxis(npyDICT['divAverage'],0,-1)[...,nSize]
	with NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/InversionData/avgSupergranuleData_DopplerLCT%s.npz' % (['_INFLOW',''][int(OUTFLOWS is not False)])) as npyDICT:
		ur_Doppler = -NP.moveaxis(npyDICT['DopplerAverage'],0,-1)[...,nSize] # negative because z is outward, but doppler inward


	flows_obs = NP.array([uxy_LCT[0],uxy_LCT[1],ur_Doppler])
	constraint_values = simps(simps(flows_obs[...,None]*Slepians[None,...],x=xgridD[:,0]/amax(xgridD),axis=1),x=ygridD[0,:]/amax(ygridD),axis=1)
	MassMatrix = NP.zeros((Slepians.shape[-1],Slepians.shape[-1]))
	for ii in range(Slepians.shape[-1]):
		for jj in range(Slepians.shape[-1]):
			MassMatrix[ii,jj] = simps(simps(Slepians[...,ii]*Slepians[...,jj],x=xgridD[:,0]/amax(xgridD),axis=0),x=ygridD[0,:]/amax(ygridD),axis=0)
	constraint_values = NP.dot(constraint_values,NP.linalg.inv(MassMatrix))
	recon_size.append(NP.sum(Slepians * constraint_values[:,None,None,:],axis=-1))

	if IGNORE_UR_constraint:
		print(text_special('Ignoring ur constraint in solar inversion','y',True,True))
		constraint_values = constraint_values[:2]

	constraint_values = constraint_values.reshape(-1)

	#---------------------------------------------------------------------


	# Build matrices for inverion
	Amatrix2 = copy.copy(Amatrix)
	L2matrix = copy.copy(Lmatrix_total)
	bmatrix2 = copy.copy(bmatrix)[:,nSize+len(Ncount)*int(not OUTFLOWS)]
	OBS = BB[:,nSize]

	nParams = len(Amatrix)
	Amatrix2 = NP.pad(Amatrix2,((0,len(constraint_values)),(0,len(constraint_values))),constant_values=((0,0),(0,0)))
	for cc in range(len(constraint_values)):
		Amatrix2[nParams+cc,:nParams] = constraint_basisCoeffs[cc];Amatrix2[:nParams,nParams+cc] = constraint_basisCoeffs[cc]

	L2matrix = NP.pad(Lmatrix_total,((0,len(constraint_values)),(0,len(constraint_values))),constant_values=((0,0),(0,0)))
	bmatrix2 = NP.append(bmatrix2,constraint_values)

	#---------------------------------------------------------------------

	# Perfmorm inversion for each regularization

	sols_tmp = [];residual_tmp = []; normLx_tmp = []
	for ii in range(len(alphaGrid)):
		LHS = Amatrix2 + alphaGrid[ii] * L2matrix
		sols_tmp.append(NP.linalg.solve(LHS,bmatrix2.real)[:-len(constraint_values)])
	sols_tmp = NP.array(sols_tmp)


	sols_list_size.append(sols_tmp)

	residual_size.append(residual_tmp)
	normLx_size.append(normLx_tmp)


	sols_list_size = NP.moveaxis(sols_list_size,0,1)
	residual_size = NP.moveaxis(residual_size,0,1)
	normLx_size = NP.moveaxis(normLx_size,0,1)


# Save all data

NP.savez_compressed('Result_data/Flow_Results%s_%iMm%s.npz' % (['','Scaledrho'][int(SCALED)],SlepContour,['_INFLOW',''][int(OUTFLOWS is not False)]),
						residual = residual,\
						residual_size = residual_size,\
						normLx = normLx,\
						normLx_size = normLx_size,\
						alphaGrid = alphaGrid,\
						alphaI = alphaI,\
						sols_list = sols_list,\
						sols_list_size = sols_list_size)#,\
