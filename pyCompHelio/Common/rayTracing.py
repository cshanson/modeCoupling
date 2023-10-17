import numpy  as NP
import scipy.interpolate as ITP
from .     import *
from ..Parameters import *
from ctypes import c_float,c_double,c_int,POINTER,byref,CDLL
from numpy.ctypeslib import ndpointer
import numpy as NP
import matplotlib.pylab as plt


# First we specify the c functions in the shared library

ClusterName = ['HelioCluster','DalmaCluster'][getClusterName() == 'slurm']
_rayTrace = CDLL(pathToMPS() + '/bin/%s/rayTracing/csh_ray_path_sobj.so' % ClusterName)

rtvinfo_c = _rayTrace.raytravelinfo
rtvinfo_c.argtypes = (c_float,c_double,\
	              c_double,c_int,\
	              c_int,c_int,c_int,
	              POINTER(c_float),POINTER(c_double),POINTER(c_double),POINTER(c_double))

raypath_c = _rayTrace.raypath
raypath_c.argtypes = (c_float,c_float,c_int,c_int,c_int,c_int,\
	              ndpointer(dtype='f8', ndim=1, flags='aligned, c_contiguous'),\
	              ndpointer(dtype='f8', ndim=1, flags='aligned, c_contiguous'),\
	              ndpointer(dtype='f8', ndim=1, flags='aligned, c_contiguous'),\
	              ndpointer(dtype='f8', ndim=1, flags='aligned, c_contiguous'),\
	              ndpointer(dtype='f8', ndim=1, flags='aligned, c_contiguous'),\
	              ndpointer(dtype='f8', ndim=1, flags='aligned, c_contiguous'))

# then we build python code for calling the c program

def raytrvinfo(freq,distance,dist_err = 0.001,mtype = 1,actype = 2,crdtype = 1,npts = 100):
	'''
	Given an input frequency and distance computes the info of the ray
	freq:     frequency of ray (Hz)
	distance: ray travel distance (deg)
	dist_err: stopping criteria for distance in the loop in ell (deg)
	mtype:    "model type (1:jcd;2:smh;3:txp;4:csm;5:csm_a)"
	actype:   "acoustic cutoff type (1: genac, 2: isoac)"
	crdtype:  "coordinate type (1: polr, 2: cart)"
	npts:     number of points in the ray for calculation

	returns: [ell,time,rdepth] = [ell of ray, travel time [min],depth of ray in Mm]
	'''

	# allocate the output to memory
	ell    = c_float(0.0);	time   = c_double(0.0)
	rdepth = c_double(0.0);	vp     = c_double(0.0)

	# Call C program
	rtvinfo_c(freq,distance,dist_err,mtype,actype,crdtype,npts,byref(ell),byref(time),byref(rdepth),byref(vp))

	# print details
	print('ray info:\nfreq     = %1.2fuHz\nell      = %1.2f\ndistance = %1.2fdeg\ntime     = %1.2fmin\nrdepth   = %1.2fMm' % (freq*1e6,ell.value,distance,time.value,rdepth.value))

	#return details
	return [ell.value,time.value,rdepth.value]


def raypath(freq,ell,npts = 100,mtype = 1,actype = 2,crdtype = 1):
	'''
	Given an input frequency and ell computes the ray path assuming starting at N pole
	freq:     frequency of ray (Hz)
	ell:      Harmonic degree ell of ray
	npts:     number of points in the ray for calculation
	mtype:    "model type (1:jcd;2:smh;3:txp;4:csm;5:csm_a)"
	actype:   "acoustic cutoff type (1: genac, 2: isoac)"
	crdtype:  "coordinate type (1: polr, 2: cart)"

	returns: [rr,thr,cssqr,vpsqr,vgsqr] = [normalized r of ray,latitude of each point of path,
	                                             sound speed^2, phase velocity ^2,group velocity ^2]
	'''
	
	# establish size of output arrays
	rr     = NP.empty(npts);	thr    = NP.empty(npts)
	cssqr  = NP.empty(npts);	vpsqr  = NP.empty(npts)
	vgsqr  = NP.empty(npts);	ptime  = NP.empty(npts)

	# Calls c program
	raypath_c(freq,ell,npts,mtype,actype,crdtype,rr,thr,cssqr,vpsqr,vgsqr,ptime)

	#symmetrize the ray path and center on pole
	rr    = NP.concatenate((rr[1:][::-1],rr))
	thr   = NP.concatenate((-thr[1:][::-1],thr)) + thr[-1]
	cssqr = NP.concatenate((cssqr[1:][::-1],cssqr))
	vpsqr = NP.concatenate((vpsqr[1:][::-1],vpsqr))
	vgsqr = NP.concatenate((vgsqr[1:][::-1],vgsqr))

	return rr,thr,cssqr,vpsqr,vgsqr



# ell,t,rd = raytrvinfo(0.003,45.5)
# rr,thr,cssqr,vpsqr,vgsqr = raypath(0.003,100)




# plt.plot(rr*NP.sin(thr*NP.pi/180),rr*NP.cos(thr*NP.pi/180),'b')
# plt.plot(NP.sin(NP.linspace(0,NP.pi,100)),NP.cos(NP.linspace(0,NP.pi,100)),'k')
# plt.show()



def ray_computeTravelTime(rayPath,params,FlowString):
  '''
  Computes the travel time difference for a ray (or rays) through a a flow cell
  rayPath   : [nRays,3,Np] or [3,Np] matrix with the ray paths in spherical coordinates
  params    : params2D instance of montjoie 
  FlowString: montjoie string for a flow model
  '''
  rayPath = NP.array(rayPath)
  if rayPath.ndim == 3:
    nRays = len(rayPath)
  elif rayPath.ndim == 2:
    nRays = 1
    rayPath = NP.array(rayPath)[NP.newaxis,:,:]
  else:
    raise Exception('rayPath must be matrix of [nRays,3,Npts] or [3,Npts] in spherical coords')

  print('Building Axisymmetric Flow Model')  
  fb      = Flow(flowString=FlowString)
  print('Done')

  print('Computing Unit Tangent Vector')
  dtau = []
  for i in range(nRays):
    print('for ray %i/%i' % (i+1,nRays))
    r_path,th_path,ph_path = rayPath[i]
    x_path,y_path,z_path = sphericalToCartesian(rayPath[i])

    #compute the vector of the ray length
    lineSeg = [0];
    for i in range(1,len(r_path)):
      lineSeg.append(lineSeg[i-1]+NP.sqrt(NP.diff(x_path)[i-1]**2 + NP.diff(y_path)[i-1]**2 + NP.diff(z_path)[i-1]**2))
    lineSeg = NP.array(lineSeg)

    #then we compute the tangent and normalize
    dt = FDM_Compact(lineSeg)
    dxdt = dt.Compute_derivative(x_path)
    dydt = dt.Compute_derivative(y_path)
    dzdt = dt.Compute_derivative(z_path)
    tangent_lines = NP.array([x_path+dxdt,y_path+dydt,z_path+dzdt]) - NP.array([x_path,y_path,z_path])
    tangent_hat = tangent_lines/NP.linalg.norm(tangent_lines,axis=0)

    # Caluculate the flow and sound speed along the path
    u_vec   = fb(params.bgm_.rho,points = NP.array([x_path,y_path,z_path])/RSUN,coordsSystem = 'cartesian')
    cs_path = params.bgm_.c.getCoef(points = NP.array([x_path,y_path,z_path])/RSUN)

    # Then the travel time difference
    dtau.append(-2*simps(NP.sum(u_vec*tangent_hat,axis=0)/cs_path**2,x=lineSeg))
    print('   dt_x',-2*simps(u_vec[0]*tangent_hat[0]/cs_path**2,x=lineSeg))
    print('   dt_y',-2*simps(u_vec[1]*tangent_hat[1]/cs_path**2,x=lineSeg))
    print('   dt_z',-2*simps(u_vec[2]*tangent_hat[2]/cs_path**2,x=lineSeg))
  print('Done')

  return dtau
