import numpy             as NP
import matplotlib.pyplot as PLOT

from ..Common       import *
from ..Parameters   import *
from ..Observations import *
from ..Filters      import *

# define function for 1 frequency and ell
def Extract_Eigenfunction(initFile,Freq,L,FWHM = 0.,typeOfObservable = TypeOfObservable.rhoc2DivXi,Pl = None):
	# initFile in the initFile used to run the simulation
	# Freq = frequency in Hz
	# L = ell of eigenfunction
	ell  = int(L)

	params = parameters(initFile,TypeOfOutput.Polar2D)
	outDir = params.config_('OutDir')
	ID = int(NP.random.rand(1)*1.e32)
	dampingFile = outDir + '/rFWHM_%d.txt' % ID
	params.config_.set('Frequencies','SINGLE %1.16e' % Freq)
	if hasattr(FWHM,'__len__'):
		FWHM = NP.squeeze(FWHM)
		print('using radial profiles')
		if len(FWHM) != 2:
			raise Exception('rFWHM must be list [r,FWHM(r)]')
		rfile = params.bgm_.getRadius()
		dampSpatial = NP.interp(rfile,FWHM[0],FWHM[1]/2)
		NP.savetxt(dampingFile,NP.array([rfile,dampSpatial]).T)
		params.config_.set('Damping','CONSTANT 1.')
		params.config_.set('DampingSpatial','RADIAL %s' % dampingFile)
	else:
		print('using constant profile')
		params.config_.set('Damping','CONSTANT %1.16e' % (FWHM/2.))
	if params.unidim_:
		params.config_.set('MaximumDegree','%i' % (ell+2))
		params.config_.set('SaveEllCoeffs','ALL')

	mkdir_p(params.config_('OutDir'))

	configNEW = outDir + '/CurrentInitFile_%d.init' % ID
	params.config_.save(configNEW)
	paramsNEW = parameters(configNEW,TypeOfOutput.Polar2D)
	G1 = Green(paramsNEW,onTheFly=configNEW,onTheFlyDel = True,observable=typeOfObservable)

	Green_rt = G1.get(ifreq = 0)[0]
	if paramsNEW.unidim_:
		Green_rell = NP.squeeze(Green_rt)[...,ell]/NP.sqrt(2*NP.pi) # 2pi for normalization
	else:
		if Pl is None:
			Green_rell = projectOnLegendre(Green_rt,NP.arange(ell+5),normalized=True,axisTheta=-1,pgBar=True,theta = paramsNEW.geom_.theta())
			Green_rell = Green_rell[:,ell]
		else:
			Green_rell = simps(Green_rt*Pl*NP.sin(paramsNEW.geom_.theta()),x=paramsNEW.geom_.theta(),axis=-1)\
			                /simps(Pl**2*NP.sin(paramsNEW.geom_.theta()),x=paramsNEW.geom_.theta(),axis=-1)

	remove(configNEW)
	if os.path.isfile(dampingFile):
		remove(dampingFile)

	return Green_rell

def Extract_Eigenfunction_Variance(initFile,Freq,L,FWHM = 0.,dFWHM = 0.,Nrealizations=100,typeOfObservable = TypeOfObservable.rhoc2DivXi,Pl = None,RealImag='real'):
	# runs the Extract_Eigenfunction command with noise realization, then computes variance

	Greens_grid = []
	damping     = []
	for ii in range(Nrealizations):
		damping.append(abs(FWHM + NP.random.randn()*dFWHM))
		G_tmp = Extract_Eigenfunction(initFile,Freq,L,damping[-1]*2,typeOfObservable,Pl)
		Greens_grid.append(G_tmp)
	Greens_grid = NP.array(Greens_grid);damping = NP.array(damping)


	if RealImag.upper() == 'REAL':
		return NP.var(NP.real(Greens_grid),axis=0)
	elif RealImag.upper() == 'IMAG':
		return NP.var(NP.imag(Greens_grid),axis=0)
	elif RealImag.upper() == 'GAMMAGG*':
		return NP.var(damping[:,None]*NP.conj(Greens_grid)*Greens_grid,axis=0)


def FWHM_eigenFunction_Kernel(initFile,Freq,L,FWHM=0.,typeOfObservable = TypeOfObservable.cDivXi,Pl = None):
	# Compute and Extract the eigenfunction for this kernel
	Gl = Extract_Eigenfunction(initFile,Freq,L,FWHM,typeOfObservable,Pl)

	# obtain some term for the kernel	
	params = parameters(initFile,TypeOfOutput.Polar2D)
	omega  = Freq*2*NP.pi
	R      = params.geom_.r()*RSUN
	rho    = params.bgm_.rho(geom=params.geom_)[:,0]
	c   = params.bgm_.c(geom=params.geom_)[:,0]

	#multiply by 2 to get FWHM kernels

	if typeOfObservable == TypeOfObservable.cDivXi:
		return NP.array(NP.real(2*NP.conj(Gl)*Gl*rho/simps(NP.conj(Gl)*Gl*(R**2)*rho,x=R)))
		# return NP.array([NP.real(2*NP.conj(Gl)*Gl*rho/simps(NP.conj(Gl)*Gl*(R**2)*rho,x=R)),\
			             # NP.real(2*NP.conj(Gl)*Gl*rho*omega**3/(2*c**2)/simps(NP.conj(Gl)*Gl*(R**2)*rho,x=R))])
	elif typeOfObservable == TypeOfObservable.rhoc2DivXi:
		raise Warning('rhoc2DivXi has not been tested')
		return NP.real(2*NP.conj(Gl)*Gl*rho/simps(NP.conj(Gl)*Gl*(R**2)*rho,x=R))


def Run_MJ_EigenProblem_Setup(Lmax = 999,BCString = 'ATMO RBC 1',Run = True,params = None):
	if params is None:
		f = open('EigenValues_Vectors.init','w')
		f.write('[dataParam]\n')
		f.write('OutDir              = /scratch/seismo/%s/TMP/eigenValues/\n' % getpass.getuser())
		f.write('InteractiveMode     = YES\n')
		f.write('MontjoiePrintLevel  = 3\n')
		f.write('TypeEquation        = HELMHOLTZ\n')
		f.write('TypeElement         = EDGE_LOBATTO\n')
		f.write('BoundaryCondition   = %s\n' % BCString)
		f.write('Source              = SRC_DIRAC 1.0 0. 0. spherical\n')
		f.write('MeshOptions         = LAYERED 0 0.9995 1.0007126 MANUAL 1000 100\n')
		f.write('OrderDiscretization = 5\n')
		f.write('MaximumDegree       = %i\n' % Lmax)
		f.write('Modes               = SINGLE 0\n')
		f.write('Frequencies         = SINGLE 3e-12\n') # Because zero == NAN
		f.write('BackgroundFile      = /home/%s/mps_montjoie/data/background/modelS_Bspline_EXT_expRho_CUT_500km.txt\n' % getpass.getuser())
		f.write('Damping             = CONSTANT 0.\n')
		f.write('FileOutputCircle    = greenCircle R 1.0 THETA UNIFORM 5000\n')
		f.write('StorageMatrix       = YES')
		f.close()
	else:
		params.config_.save('EigenValues_Vectors.init')

	if Run:
		os.system('cd /home/%s/mps_montjoie/pyCompHelio/RunMontjoie/; ./runMontjoie.py %s/%s -v -c' % (getpass.getuser(),os.getcwd(),'EigenValues_Vectors.init'))

	return os.getcwd() + '/EigenValues_Vectors.init'



def Determine_MJ_EigenValues_EigenVectors(initFile,ell,nuMax = None,nuIndMax = None,ReturnEigenValues = False):
	params = parameters(initFile,TypeOfOutput.Surface1D)
	G1     = Green(params)

	Matrix = loadComplexMat(params.config_('OutDir') + '/STORAGE_MATRIX/Mat_ell%i.dat' % ell) / RSUN**2
	mass_matrix_omega = load1D(params.config_('OutDir') + '/STORAGE_MATRIX/mass_matrix_omega.dat') / (-(params.time_.omega_*RSUN)**2)
	rtilde = load1D(params.config_('OutDir') + '/STORAGE_MATRIX/r_tilde.dat')

	Matrix = NP.diag(1/mass_matrix_omega)*Matrix
	Matrix = NP.nan_to_num(Matrix)

	eigenvalues,eigenvectors = scipy.linalg.eig(NP.real(Matrix))

	eigenvaluesnu = NP.sqrt(eigenvalues - (params.time_.omega_)**2)/(2*pi)
	eigenvaluesnu = NP.nan_to_num(eigenvaluesnu)

	eigenvectors  = eigenvectors[:,NP.argsort(eigenvaluesnu)]
	eigenvaluesnu = NP.sort(eigenvaluesnu)

	if nuMax is not None or nuIndMax is not None:
		if nuMax is not None:
			ind = NP.argmin(abs(eigenvaluesnu - nuMax))
		else:
			ind = nuIndMax
		eigenvaluesnu = eigenvaluesnu[:ind]
		eigenvectors  = eigenvectors[:,:ind]

	if ReturnEigenValues:
		return NP.real(eigenvaluesnu)
	else:
		return rtilde,NP.real(eigenvaluesnu),eigenvectors
