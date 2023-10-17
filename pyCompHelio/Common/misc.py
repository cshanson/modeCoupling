import os
import configparser
import errno 
import re
import pickle
import dill
import shutil
import socket
import numpy as NP
import scipy
#from peakdet import *
from scipy.optimize import curve_fit
from scipy.integrate   import simps
import matplotlib.pylab as plt
import time
import datetime
from inspect import currentframe, getframeinfo
import sys
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure, binary_erosion

#==============================================================================
# File management 

def wipeFigs():
  plt.close('all');plt.figure();plt.close('all')
def which(file):
  for path in os.environ["PATH"].split(":"):
    if os.path.exists(path + "/" + file):
      return path + "/" + file
  return None

def cutExtension(filename):
  return ".".join(filename.split('.')[:-1])

def getExtension(filename):
  return filename.split('.')[-1]

def removeDoubleSlashes(path):
    return '/'.join(path.split('//'))

def mkdir_p(path):
    ''' Equivalent of mkdir -p '''
    try:
      os.makedirs(path)
    except OSError as exc:  # Python >2.5
      if exc.errno == errno.EEXIST and os.path.isdir(path):
        pass
      else:
        raise

def remove(filename):
  ''' thanks StackOverflow '''
  if os.path.isfile(filename):
    try:
      os.remove(filename)
    except OSError as e:
      if e.errno != errno.ENOENT:
        raise
  elif os.path.isdir(filename):
    shutil.rmtree(filename,ignore_errors=True)

def purge(dir,pattern):
  ''' thanks StackOverflow '''
  for f in os.listdir(dir):
    if re.search(pattern,f):
      os.remove(os.path.join(dir,f))
     
def getHostname():
  if socket.gethostname().find('.')>=0:
    return socket.gethostname()
  else:
    return socket.gethostbyaddr(socket.gethostname())[0]

#==============================================================================
# Pickle /Marshal

def saveObject(thing,filename):
    with open(filename,'wb') as output:
      # pickle.dump(thing,output,pickle.HIGHEST_PROTOCOL)
      dill.dump(thing,output,pickle.HIGHEST_PROTOCOL)


def loadObject(filename):
    with open(filename,'rb') as input:
      # return pickle.load(input)
      return dill.load(input)

#def saveFunc(func,filename):
#    code_string = pickle.dumps(func)
#    saveObject(code_string,filename)

#def loadFunc(filename,funcname):
#    code_string = loadObject(filename)
#    code        = pickle.loads(code_string)
#    return types.FunctionType(code,globals(),funcname)

def testfunc(a):
  return NP.array(a*NP.ones((10,)))

#==============================================================================

class bColors:
    ''' Bold colors escape characters '''
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def warning():
      return bColors.WARNING + 'Warning: ' + bColors.ENDC

#==============================================================================
# Extension of ConfigParser to get easier access...
class myConfigParser(configparser.RawConfigParser):
    ''' extension of rawConfigParser to avoid if config.has_option, then config.get... ''' 

    def __init__(self,filename):

      configparser.RawConfigParser.__init__(self)
      self.optionxform = str
      self.fileName_ = filename
      self.read(filename)

    def __call__(self,key,default_value=None):

      if default_value is not None:
        res = default_value

      if self.has_option(self.sections()[0],key):
        res = self.get(self.sections()[0],key)
      else:
        if default_value is None:
          raise IOError('No key '+key+' found in .init file and no default value was given.')

      # remove eventual line breaks from res
      if isinstance(res,str):
        return res.replace('\n',' ')
      else:
        return res

    def set(self,key,string):
      ''' Sets or adds an option '''
      configparser.RawConfigParser.set(self,self.sections()[0],key,string)

    def remove_option(self,key):
      ''' removes an option '''
      configparser.RawConfigParser.remove_option(self,self.sections()[0],key)

    def save(self,fileName):
      with open(fileName,'w') as of:
        self.write(of)
      self.fileName_ = fileName

    def update(self,keys,options,fileName=None):

      if len(keys) != len(options):
        raise Exception("Options and keys must have the same length")

      if fileName is not None:
        newC = myConfigParser(self.fileName_)
        toUpdate = newC
      else:
        toUpdate = self

      for (k,opt) in zip(keys,options):
        toUpdate.set(k,opt)

      if fileName is not None:
        newC.fileName_  = fileName
        newC.save(fileName)
        return newC

    def waitCompletion(self, nFiles, typeOfOutput = 'FileOutputCircle', test = False):
      '''Wait for the cluster to have finished computing all the Green's function for the type of output given as argument. If test = True, just check if the computation was already done.'''
      FilePrefix = self(typeOfOutput).split()[0]
      outDir = self('OutDir')
      command = 'ls %s/%s*| wc -l' % (outDir + '/results/source0/',FilePrefix)
      if test:
        return int(os.popen(command).read()) == nFiles
      else:
        while int(os.popen(command).read()) < nFiles:
          time.sleep(30)
        return True


#==============================================================================
def updateConfig(initFileName,ObsHeight, angle, suffix,ComputeGrad = False,Mode = 0):
    '''update the configuration file by changing the source location and creating the output directory. If sourceLocation = None, it is created for the source at the pole. Returns the name of the new configuration file.'''
    if len(angle) != len(ObsHeight):
      raise Exception('Vector containing sources heights must equal vector of angles')
    config = myConfigParser(initFileName)
    outDir = config('OutDir')
    SourceTerm = ''
    for i in range(len(angle)):
      if i != 0:
        SourceTerm = SourceTerm +', '
      SourceTerm = SourceTerm + 'SRC_DIRAC %s %s 0. spherical' % (ObsHeight[i], angle[i])
    config.set('Source',SourceTerm)
    outDir = '%s/source_%s/' % (outDir, suffix)
    if ComputeGrad:
      config.set('ComputeGradient', 'YES') 

    config.set('Modes','SINGLE %i' % Mode)
    mkdir_p(outDir)
    config.set('OutDir', outDir)
    filename = '%s/config.init' % outDir
    config.save(filename)
    return filename

#==============================================================================
# Norm stuff

def dotProd(a,b,axis=None):
  a = NP.array(a)
  b = NP.array(b)
  return NP.sum(a*b,axis=axis)

def gradDot(a,b):
  return dotProd(a,b,axis=0)

def norm(p,U,X=None):
  # L^p(X) norm 
  if not X is None:
    dX      = NP.zeros(len(X))
    dX[:-1] = X[1:]-X[:-1]
    l       = NP.zeros(len(X))
    l[1:-1] = (dX[:-2]+dX[1:-1])/2
    l[1]    = dX[1]/2
    l[-1]   = dX[-2]/2
  else:
    l = 1.e0
  
  return (NP.sum(l*(NP.fabs(U))**(p*1.0)))**(1/(p*1.0))

def norm2(U,X=None):
  if X is None:
    return NP.sqrt(NP.sum(NP.abs(NP.asarray(U))**2))
  else:
    return norm(2,U,X)


def getExtrema(U,x=None,type=None,interp=True):
  
  if x is None:
    x = list(range(len(U)))

  res  = []
  resf = []

  for i in range(1,len(U)-1):
    f0 = U[i-1];
    f1 = U[i];
    f2 = U[i+1];

    bool = ((f1-f0)*(f1-f2) >= 0)
    if type=='MIN':
      bool = bool and (f1<f0)
    elif type=='MAX':
      bool = bool and (f1>f0)

    if bool:
      if interp:

        x1 = x[i];
        x0 = x[i-1];
        x2 = x[i+1];

        c0 = f0/((x0-x1)*(x0-x2));
        c1 = f1/((x1-x0)*(x1-x2));
        c2 = f2/((x2-x0)*(x2-x1));

        xmax = (x0*(c1+c2)+x1*(c0+c2)+x2*(c0+c1))/(2*(c0+c1+c2))
        res.append(xmax)
        resf.append(c0*(xmax-x1)*(xmax-x2) + c1*(xmax-x0)*(xmax-x2) + c2*(xmax-x0)*(xmax-x1))

      else:
        res.append(i)
        resf.append(f1)

  return NP.asarray(res),NP.asarray(resf)

def getZeros(U,x=None,interp=True):
  
  if x is None:
    x = list(range(len(U)))

  res  = []

  for i in range(0,len(U)-1):
    f1 = U[i];
    f2 = U[i+1];

    bool = ((f2*f1)<= 0)

    if bool:
      if interp:

        x1 = x[i];
        x2 = x[i+1];
        xzero = x1-f1*(x2-x1)/(f2-f1)
        res.append(xzero)

      else:
        res.append(i)

  return NP.asarray(res)


def getMaximumRadiusMesh(fileMesh,fullName=False):

  if fullName:
    FM=fileMesh
  else:
    FM=os.getcwd().split('mps_montjoie')[0]+'/mps_montjoie/data/meshes/'+fileMesh

  with open(FM) as fm:

    lines = fm.readlines()
    i =-1
    for l in range(len(lines)):
      if lines[l][:5]=='Edges':
        i =l
        break
    points = lines[5:i]
    rmax = -1
    for pStr in points:
      p    = pStr.split()[:2]
      rmax = max(rmax,norm2([float(p[0]),float(p[1])]))
    return rmax

#==============================================================================
# Sample vectors
def subSampleVector(vector, subSampling):
  if subSampling > 1:
    newSizeR  = int(NP.floor(len(vector) / subSampling))
    newVector = NP.zeros(newSizeR)
    for i in range(0, newSizeR):
      newVector[i] = (vector[i*subSampling] + vector[i*subSampling+1])/2

    # Why not return vector[::int(subSampling)] ??

  elif subSampling == 1:
    newVector = vector
  else:
    cpt = 0
    sampling  = int(NP.round(1. / subSampling))
    newVector = NP.zeros(((len(vector) -1) * sampling + 1))
    for i in range(0, len(vector)-1):
      for j in range(0, sampling):
        newVector[cpt] = vector[i] + j * subSampling * (vector[i+1] - vector[i])
        cpt = cpt + 1
    newVector[cpt] = vector[-1]
  return newVector


#=============================================================================
# Some commonly used function profiles

def smoothRectangle(x,wmin,wmax,ratio_t):
  ''' rectangle window function with smooth junctions '''

  d  = 0.5*ratio_t*(wmax-wmin)
  x0 = wmin-d/2.
  x1 = wmin+d/2.
  x2 = wmax-d/2.

  if hasattr(x,'shape'):
    res  = NP.zeros(x.shape)
    res  = NP.where( abs(x-wmin)<=0.5*d, 0.5*(1.e0 - NP.cos( (x-x0)/d *NP.pi)),0.e0)
    res += NP.where( (x>x1)*(x<x2) , 1.e0,0.e0)
    res += NP.where( abs(x-wmax)<=0.5*d, 0.5*(1.e0 + NP.cos( (x-x2)/d * NP.pi)),0.e0)
    return res
  else:
    if abs(x-wmin)<0.5*d:
      return 0.5*(1.e0 - NP.cos( (x-x0)/d *NP.pi))
    elif abs(x-wmax)<0.5*d:
      return 0.5*(1.e0 + NP.cos( (x-x2)/d * NP.pi))
    elif (x>x1) and (x<x2):
      return 1.e0
    else:
      return 0.e0

def gaussian(x,mean,sd,amp=1):
  ''' Gaussian profile '''
  return amp*(NP.exp(-(x-mean)**2/(2.e0*sd**2)))

def lorenz(x,mean,sd,amp=1):
  # Note that the FWHM = 2*sqrt(2)*sd
  return amp*1.e0/(1.e0+((x-mean)/(NP.sqrt(2.)*sd))**2)

def lorenz_FWHM(x,x0,FWHM,amp=1):
  return amp*(FWHM/2.)**2/((x-x0)**2+(FWHM/2)**2)

def multi_lorenz_FWHM(x, *params):
  y = NP.zeros_like(x)
  params = NP.squeeze(params)
  for i in NP.arange(0, len(params), 3):
    ctr = params[i]
    wid = params[i+1]
    amp = params[i+2]
    y += lorenz_FWHM(x,ctr,wid,amp)
  return y

def asym_lorenz(x,xc,FWHM,amp = 1.,asym = 1.,asym2 = 1.):
  # Compute an asymmetric lorentzian with FWHM of FWHM, cntered at xc
  # Formula taken from korzennik 2013
  # Set up the symmetric lorentzian function
  xi = (x-xc)/(FWHM/2)
  lor = amp/(1+xi**2)
  # Find the indice of the centre of lorentzain
  ind = NP.argmin(abs(x-xc))
  # Establish output grid
  out = NP.zeros(x.shape)
  # Apply asymmetry to the left and the right
  out[:ind] = lor[:ind]**asym
  out[ind:] = lor[ind:]**asym2
  return out

def sigmoid(x,threshold,slope):
  ''' Smooth Heaviside function'''
  return 0.5e0*(1+NP.tanh(slope*(x-threshold)))

#==============================================================================

def integrateBoole(f,x=None,axis=-1):
  ''' adapatation of Boole's Newton Cotes formula to non equally spaced points,
      as scipy.integrate.simps '''

  itgd,x = prepareIntegration(f,x,axis)

  if (N-1)%4!=0:
    istop  = 4*(N-1)/4
    res1   = integrateBoole(itgd[...,:istop],x[:istop],axis=-1) \
             + scipy.integrate.simps(itgd[...,istop-1:],x[istop-1:],axis=-1)
    istart = (N-1)%4
    res2   = integrateBoole(itgd[...,istart:],x[istart:],axis=-1) \
             + scipy.integrate.simps(itgd[...,:istart+1],x[:istart+1],axis=-1)
    
    res    = 0.5*(res1+res2) 

  else:
    
    res  = NP.zeros(itgd.shape[:-1],dtype=f.dtype)

    for i in range(5):
      pximxk  = NP.ones(x[:-1:4].shape)
      pxk     = 1.e0
      sxk     = 0.e0
      sxkxj   = 0.e0
      sxjxkxl = 0.e0

      if i==0:
        islice = slice(i,-1,4)
      else:
        islice = slice(i,None,4)

      for k in range(5):

        if k==0:
          kslice = slice(k,-1,4)
        else:
          kslice = slice(k,None,4)

        # 3 terms product
        tpxjxkxl = NP.ones(x[:-1:4].shape)
        for j in range(5):
          if j==0:
            jslice = slice(j,-1,4)
          else:
            jslice = slice(j,None,4)
          if j!=i and j!=k:
            tpxjxkxl = tpxjxkxl * x[jslice]

        if k!= i:
          sxjxkxl = sxjxkxl + tpxjxkxl
        #----------------

        if k!=i:
          pximxk = pximxk*(x[islice]-x[kslice])
          pxk    = pxk   * x[kslice]
          sxk    = sxk   + x[kslice]
          tsxj   = 0.e0
          for j in range(k+1,5):

            if j==0:
              jslice = slice(j,-1,4)
            else:
              jslice = slice(j,None,4)

            if j!=i:
              tsxj = tsxj + x[jslice]
          sxkxj   = sxkxj   + tsxj*x[kslice]

      x4   = x[4::4]
      x0   = x[:-1:4]

      num  = 0.2e0*(x4**5-x0**5) - 0.25e0*(x4**4-x0**4)*sxk + 1.e0/3.e0*(x4**3-x0**3)*sxkxj - 0.5*(x4*x4-x0*x0)*sxjxkxl + pxk*(x4-x0)
      res  = res + NP.sum(itgd[...,islice]*(num/pximxk),axis=-1)
   
  return res

def integrateSimpsons38(f,x=None,axis=-1):
  ''' see above
      if the total interval cannot be divided by 3,
      simpsons rules is apply on the last interval '''


  itgd,x = prepareIntegration(f,x,axis)

  if (N-1)%3!=0:
    istop  = 3*(N-1)/3
    res1   = integrateSimpsons38(itgd[...,:istop],x[:istop],axis=-1) \
             + scipy.integrate.simps(itgd[...,istop-1:],x[istop-1:],axis=-1)
    istart = (N-1)%3 
    res2   = integrateSimpsons38(itgd[...,istart:],x[istart:],axis=-1) \
             + scipy.integrate.simps(itgd[...,:istart+1],x[:istart+1],axis=-1)
    
    res    = 0.5*(res1+res2) 

  else:
    
    H  = x[3::3]-x[:-1:3]
    h0 = x[1::3]-x[:-1:3]
    h1 = x[2::3]-x[1::3]
    h2 = x[3::3]-x[2::3]

    div  = ((h1+h2)*(h2-h1-h0)-h0*h2)/(h0*(h0+h1))
    div2 = ((h1+h0)*(h0-h1-h2)-h0*h2)/(h2*(h2+h0))

    res = NP.sum( H/12.e0 *(itgd[...,:-1:3] * (3.e0+div)\
                          + itgd[..., 1::3] * (H*H*(h0+h1-h2)/(h0*h1*(h1+h2)))\
                          + itgd[..., 2::3] * (H*H*(h2+h1-h0)/(h2*h1*(h1+h0)))\
                          + itgd[..., 3::3] * (3.e0+div2)))

  return res  

def prepareIntegration(f,x,axis):

  f   = NP.asarray(f)
  N   = f.shape[axis]
  if x is None:
    x = NP.arange(N)
  else:
    x = NP.asarray(x)

  if len(x.shape)!=1 or len(x)!=N:
    raise Exception('x must be a 1D vector of the same size as f.shape[axis].')

  if axis!=-1 and axis!=f.ndim-1:
    itgd = NP.rollaxis(f,axis,f.ndim) 
  else:
    itgd = f

  return itgd,x

def cumsimps(integrand, x, even='first', **kwargs):
    assert integrand.size == x.size
    out = NP.empty(x.size)
    for i in range(x.size):
        out[i] = simps(integrand[:i+1], x=x[:i+1], even=even, **kwargs)
    return out

#======================================================================
# Coordinates options in init file

def readCoordinatesOptions(options,BGfile=None):

  if options[1] == 'UNIFORM':
    Nc = int(options[2])
    try:
      cmin = evalFloat(options[3])
      cmax = evalFloat(options[4])
    except:
      if options[0] == 'R':
        cmin = 0
        cmax = 1
      if options[0] == 'THETA':
        cmin = 0
        cmax = NP.pi
      if options[0] == 'PHI':
        cmin = 0
        cmax = 2.e0*NP.pi - 2.e0*NP.pi/Nc
      if options[0] in ['X','Z']:
        cmin = -1.e0
        cmax = 1.e0
    return NP.linspace(cmin,cmax,Nc)

  elif options[1] == 'SAMPLE':
    try:
      sample = int(options[2])
    except:
      sample = 1
    try:
      filename = options[3]
      if filename == "CUT":
        filename = BGfile
    except:
      filename = BGfile

    plop = NP.loadtxt(filename,comments='#')
    if plop.ndim >1:
      coords = plop[::sample,0]
    else:
      coords =  plop[::sample]
  
    if 'CUT' in options:
      icut = options.index('CUT')
      try:      
        cmin = evalFloat(options[icut+1])
        cmax = evalFloat(options[icut+2])
        return coords[(coords>=cmin)*(coords<=cmax)]
      except:
        return coords
    else:
      return coords

def evalFloat(strf):
  return eval(strf.replace('pi',repr(NP.pi)))

def evalInt(strf):
  return int(eval(strf))
#==============================================================================
# Coordinates conversion

def cartesianToSpherical(MC):
  MC = NP.asarray(MC)
  return NP.array([NP.sqrt(MC[0,...]**2+MC[1,...]**2+MC[2,...]**2),\
                   NP.arctan2(NP.sqrt(MC[0,...]**2+MC[1,...]**2),MC[2,...]),\
                   NP.arctan2(MC[1,...],MC[0,...]) % (2*NP.pi)])

def sphericalToCartesian(MS):
  MS = NP.asarray(MS)
  return NP.array([MS[0,...]*NP.sin(MS[1,...])*NP.cos(MS[2,...]),\
                   MS[0,...]*NP.sin(MS[1,...])*NP.sin(MS[2,...]),\
                   MS[0,...]*NP.cos(MS[1,...])])

def cartesianToCopolar(MC):
  r = NP.sqrt(MC[0,...]**2+MC[1,...]**2)
  theta = NP.arctan2(MC[0,...],MC[1,...])
  return NP.array([r,(NP.arctan2(MC[0,...],MC[1,...]) + 2*NP.pi) % (2*NP.pi)])

def copolarToCartesian(MS):
  return NP.array([MS[0,...]*NP.sin(MS[1,...]),\
          MS[0,...]*NP.cos(MS[1,...])])

def cylindricalToCartesian(MC):
  MC = NP.asarray(MC)
  return NP.array([MC[0,...]*NP.cos(MC[1,...]),\
                   MC[0,...]*NP.sin(MC[1,...]),\
                   MC[2,...]])

def cartesianToCylindrical(MC):
  MC = NP.asarray(MC)
  return NP.array([NP.sqrt(MC[0,...]**2+MC[1,...]**2),\
                   NP.arctan2(MC[1,...],MC[0,...]),\
                   MC[2,...]])

def cylindricalToSpherical(MC):
  MC = NP.asarray(MC)
  return NP.array([NP.sqrt(MC[0,...]**2+MC[2,...]**2),\
                   NP.arctan2(MC[0,...],MC[2,...]),\
                   MC[1,...]])

def sphericalToCylindrical(MS):
  MS = NP.asarray(MS)
  return NP.array([MS[0,...]*NP.sin(MS[1,...]),\
                    MS[2,...],\
                    MS[0,...]*NP.cos(MS[1,...])])

# Conversion of vectors between coordinates systems.
# The input arrays must be (3,...)
# theta and phi are arrays of locations of the vectors
# use geom.GetXXXXXXXCoordsList() or geom.GetXXXXXCoordsMeshGrid() 
# to get these arrays

def sphericalToCartesianVector(VS,theta,phi):

  VS = NP.asarray(VS)
  VC = NP.zeros(VS.shape,dtype=VS.dtype)

  ct = NP.cos(theta)[NP.newaxis,:,NP.newaxis]
  st = NP.sin(theta)[NP.newaxis,:,NP.newaxis]
  cf = NP.cos(phi)[NP.newaxis,NP.newaxis,:]
  sf = NP.sin(phi)[NP.newaxis,NP.newaxis,:]

  VC[0,...] = st*cf*VS[0,...] + ct*cf*VS[1,...] - sf*VS[2,...]
  VC[1,...] = st*sf*VS[0,...] + ct*sf*VS[1,...] + cf*VS[2,...]
  VC[2,...] =    ct*VS[0,...] -    st*VS[1,...]

  return VC

def sphericalToCylindricalVector(VS,theta):

  VS = NP.asarray(VS)
  VC = NP.zeros(VS.shape,dtype=VS.dtype)
  
  ct = NP.cos(theta)
  st = NP.sin(theta)

  VC[0,...] = st*VS[0,...] + ct*VS[1,...]
  VC[1,...] = VS[2,...]
  VC[2,...] = ct*VS[0,...] - st*VS[1,...]

  return VC

def cartesianToSphericalVector(VC,theta,phi):

  VC = NP.asarray(VC)
  VS = NP.zeros(VC.shape,dtype=VC.dtype)

  ct = NP.cos(theta)[NP.newaxis,:,NP.newaxis]
  st = NP.sin(theta)[NP.newaxis,:,NP.newaxis]
  cf = NP.cos(phi)[NP.newaxis,NP.newaxis,:]
  sf = NP.sin(phi)[NP.newaxis,NP.newaxis,:]

  VS[0,...] = st*cf*VC[0,...] + st*sf*VC[1,...] + ct*VC[2,...]
  VS[1,...] = ct*cf*VC[0,...] + ct*sf*VC[1,...] - st*VC[2,...]
  VS[2,...] =   -sf*VC[0,...] +    cf*VC[1,...]

  return VS

def cartesianToCylindricalVector(VCart,phiCyl):

  VCart = NP.asarray(VCart)
  VCyl  = NP.zeros(VCart.shape,dtype=VCart.dtype)

  cf = NP.cos(phiCyl)
  sf = NP.sin(phiCyl)

  VCyl[0,...] =  cf*VCart[0,...] + sf*VCart[1,...]
  VCyl[1,...] = -sf*VCart[0,...] + cf*VCart[1,...]
  VCyl[2,...] = VCart[2,...]

  return VCyl

def cylindricalToSphericalVector(VC,theta):

  VC = NP.asarray(VC)
  VS = NP.zeros(VC.shape,dtype=VC.dtype)

  ct = NP.cos(theta)
  st = NP.sin(theta)

  VS[0,...] = st*VC[0,...] + ct*VC[2,...]
  VS[1,...] = ct*VC[0,...] - st*VC[2,...]
  VS[2,...] = VC[1,...]

  return VS

def cylindricalToCartesianVector(VCyl,phiCyl):
  # phiCyl : polar angle (cylindrical coords) of location of vectors

  VCyl  = NP.asarray(VCyl)
  VCart = NP.zeros(VCyl.shape,dtype=VCyl.dtype)

  cf = NP.cos(phiCyl)
  sf = NP.sin(phiCyl)

  VCart[0,...] = cf*VCyl[0,...] - sf*VCyl[1,...]
  VCart[1,...] = sf*VCyl[0,...] + cf*VCyl[1,...]
  VCart[2,...] = VCyl[2,...]

  return VCart

def power_bit_length(x):
  return 2**(x-1).bit_length()




def theta_refine(theta,c1,c2,nearpoint_ratio = 0.7,source_refine_width=NP.pi/32.,PLOT = False,silent = False):
  # Model takes theta, and the two points of interest c1 and c2 (which will have theta refined around)

  # nearpoint_ratio ratio of refinement around and between the points (in near field)
  # Source_refine_width: Width of window refinement functions
  # PLOT: Show plot of refine mesh
  # silent: No printing
  
  # fix cases where points are swapped
  if c2<c1:
    cen1 = c2
    cen2 = c1
  else:
    cen1 = c1
    cen2 = c2

  # fix case where points are the same (i.e only want to refine a single point)
  # Compute refinement windows around the sources based on the widths given
  if c1 == c2:
    indr   = NP.argmin(abs(theta - c1 -0.1*c1))
    indl   = NP.argmin(abs(theta - c1 +0.1*c1))
    wind = (smoothRectangle(theta,cen2-source_refine_width,cen2+source_refine_width,0.5))
  else:
    indr   = NP.argmin(abs(theta - cen2 -0.1*cen2))
    indl   = NP.argmin(abs(theta - cen1 +0.1*cen1))
    wind = (smoothRectangle(theta,cen1-source_refine_width,cen1+source_refine_width,0.5) 
      + smoothRectangle(theta,cen2-source_refine_width,cen2+source_refine_width,0.5)) 
  for i in range(indl,indr):
    if wind[i] < nearpoint_ratio:
      wind[i] = nearpoint_ratio
  wind = 1.1 - wind
  wind = wind/NP.amax(wind)
  # sort out for negative values
  wind[wind<0.05] = 0.05

  # Compute dtheta
  dtheta = wind / NP.trapz(wind)
  
  # Compute new refined grid
  theta_end = [0]
  theta_now = 0
  while theta_now < NP.pi:
    theta_end.append(theta_end[-1] + NP.interp(theta_end[-1],theta,dtheta)*NP.pi)
    theta_now = theta_end[-1]
  theta_end[-1] = NP.pi
  if not silent:
    print(('Number of points in theta =', len(theta_end)))

  # Plot grid
  if PLOT == True:
    # plt.figure()
    # plt.plot(dtheta*NP.pi)

    plt.figure()
    x = NP.sin(NP.array(theta_end))
    z = NP.cos(NP.array(theta_end))
    for i in NP.arange(0,len(theta_end),int(len(theta_end)/100.)):
      plt.plot([0.7*x[i],x[i]],[0.7*z[i],z[i]],'k')
    plt.plot([0,NP.sin(cen2)],[0,NP.cos(cen2)],'r')
    plt.plot([0,NP.sin(cen1)],[0,NP.cos(cen1)],'r')
    plt.xlim([0,2])

  return NP.array(theta_end)
  # return wind

def theta_refine_nPts(theta,points,refine_width,numCells,PLOT=False,inputUnits = 'rads'):
  # given the vector of 'theta', refine around the 'points' with a 'refinement_width' of 'numCells'
  # PLOT returns a plot of the domain
  # inputRads: input can be 'rads' or 'degs' 
  if not hasattr(points,'__len__'):
    points = NP.array([points])
  else:
    points = NP.array(points)

  if inputUnits.upper() != 'RADS':
    theta        = theta * NP.pi/180
    points       = points * NP.pi/180
    refine_width = refine_width *NP.pi/180
  

  THETA = theta
  for i in range(len(points)):
    lp = points[i] - refine_width/2
    rp = points[i] + refine_width/2
    if lp < 0:
      lp = 0
    if rp > NP.pi:
      rp = NP.pi
    THETA = NP.concatenate([THETA,NP.linspace(lp,rp,numCells)])

  THETA = NP.sort(THETA)

  if PLOT == True:
    plt.figure()
    x = NP.sin(NP.array(THETA))
    z = NP.cos(NP.array(THETA))
    for i in NP.arange(0,len(THETA),int(len(THETA)/200.)):
      plt.plot([0.7*x[i],x[i]],[0.7*z[i],z[i]],'k')
    for i in range(len(points)):
      plt.plot([0,NP.sin(points[i])],[0,NP.cos(points[i])],'r')
    plt.xlim([0,2])
  
  return THETA

def dummy(a):
  return a*NP.ones((5,))

def fart():
  print ('prout')




def text_special(string,color = 'k',underline = False,bold = False,bright=False,background_colored=False):
  '''
  routine to print a string with ANSI customization
  color : is the color that the text will be printed in
  underline: underlines the text
  bold: bold font
  bright: bright colors
  background_colored: swaps the background and text color
  '''

  ENDC      = '\033[0m'
  BOLD      = '\033[1m'
  UNDERLINE = '\033[4m'
  REVERSED  = '\033[7m'

  BLACK   = '\033[30'
  RED     = '\033[31'
  GREEN   = '\033[32'
  YELLOW  = '\033[33'
  BLUE    = '\033[34'
  MAGENTA = '\033[35'
  CYAN    = '\033[36'
  WHITE   = '\033[37'

  #------------
  # first add color

  if color.upper() in ['BLACK','K']:
    special_string = BLACK
  elif color.upper() in ['RED','R']:
    special_string =  RED
  elif color.upper() in ['GREEN','G']:
    special_string =  GREEN
  elif color.upper() in ['YELLOW','Y']:
    special_string =  YELLOW
  elif color.upper() in ['BLUE','B']:
    special_string =  BLUE
  elif color.upper() in ['MAGENTA','M']:
    special_string =  MAGENTA
  elif color.upper() in ['CYAN','C']:
    special_string =  CYAN
  elif color.upper() in ['WHITE','W']:
    special_string = WHITE

  if bright:
    special_string += ';1m'
  else:
    special_string += 'm'

  # then add formatting
  if underline:
    special_string += UNDERLINE
  if bold:
    special_string += BOLD
  if background_colored:
    special_string += REVERSED

  # then add the string
  special_string += string

  # finally close the formatting
  special_string += ENDC

  # and print

  return special_string


def dt_to_dec(dt):
    """Convert a datetime to decimal year."""
    year_start = datetime.datetime(dt.year, 1, 1)
    year_end = year_start.replace(year=dt.year+1)
    return dt.year + ((dt - year_start).total_seconds() /  # seconds so far
        float((year_end - year_start).total_seconds()))  # seconds in year


def rebin(a, *args):
    shape = a.shape
    lenShape = len(shape)
    factor = NP.asarray(shape)//NP.asarray(args)
    evList = ['a.reshape('] + \
             ['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
             [')'] + ['.mean(%d)'%(i+1) for i in range(lenShape)]
    # print(''.join(evList))
    return eval(''.join(evList))

def abort_sc():
  sys.exit(text_special("Script aborted",'r',True,True))



def detect_peaks(image,lower_limit=0):

    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    detected_peaks = NP.where(abs(detected_peaks)*image > lower_limit,detected_peaks,False)

    return detected_peaks