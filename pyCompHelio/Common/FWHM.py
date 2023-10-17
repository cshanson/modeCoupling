import numpy as NP
import scipy
from .peakdet import *
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from .misc import *
from ..Common     import *
from ..Background import *

def find_FWHM_power_spec(freq,POW,delta = 0.2,guesswidth = 100):
  # Computes the FWHM of the modes in power spectrum POW for each ell and in the range of freq
  # A point is considered a maximum peak if it has the maximal
  # value, and was preceded (to the left) by a value lower by
  # DELTA.
  # guess width = rough width of peaks (in no. of elements of freq)

  # initialise
  L = NP.arange(POW.shape[0])
  FWHM     = []
  Ls       = []
  ModeFreq = []
  Amp      = []
  j=0
  # determine the frequency separation
  dfreq = freq[1]-freq[0]

  def lorentz_FWHM_lsq(p,x):
    return p[2]*(p[1]/2.)**2/((x-p[0])**2+(p[1]/2)**2)

  def errorfunc_lsq(p,x,z):
    return lorentz_FWHM_lsq(p,x) - z

  for i in L:
    # determine the peak positions and values for the chosen ell
    peaks = peakdet(POW[i,:],delta*NP.amax(POW[i,:]))[0]
    if len(peaks) != 0:
      for ii in range(len(peaks)):
        # For each peak detected fit a lorentzian and obtain x0, sd and amplitude
        peak = peaks[ii]
        try:
          if (peak[0] < 0.05*len(freq)) or (peak[0] > 0.95*len(freq)):
            raise Exception()
          p_opt,pp = curve_fit(lorenz,NP.arange(len(freq)),POW[i,:],p0=[peak[0],guesswidth,peak[1]])
          # p_opt,pp = leastsq(errorfunc_lsq,NP.array([peak[0],guesswidth,peak[1]]),args=(NP.arange(len(freq)),POW[i,:]))
          # convert the width of the lorentzian to the FWHM
          FWHM.append(abs(2*NP.sqrt(2)*p_opt[1]*dfreq))
          ModeFreq.append(peak[0]*dfreq + freq[0])
          Ls.append(i)
          Amp.append(p_opt[2])

          # FWHM.append(abs(p_opt[1]*dfreq))
          # ModeFreq.append(peak[0]*dfreq + freq[0])
          # Ls.append(i)
          # Amp.append(p_opt[2])
        except:
          FWHM.append(-1)
          Ls.append(i)
          ModeFreq.append(-1)
          Amp.append(-1)
    else:
      # If no peak detected then return the value of -1
      FWHM.append(-1)
      Ls.append(-1)
      ModeFreq.append(-1)
      Amp.append(-1)
    j = j+1
  
  # remove all failed detections
  ll = NP.array(Ls)[NP.array(FWHM) > 0]
  ff = NP.array(FWHM)[NP.array(FWHM) > 0]
  mf = NP.array(ModeFreq)[NP.array(FWHM) > 0]
  Amp = NP.array(Amp)[NP.array(FWHM) > 0]
  
  # return the harmonic degree, FWHM, mode Frequency
  return ll,ff,mf,Amp

def fit_L_FWHM(POW,delta=0.05,ls = None,PLOT = False,VERBOSE = False):
  # Fit the FWHM of the modes given a slice of the power spectrum at specific omega
  if ls is None:
    lmin = 0; lmax = -1
    ls = NP.arange(POW.shape[0])
    POWn = POW
  else:
    lmin  = ls[0];lmax = ls[-1] + 1
    POWn  = POW[lmin:lmax]
  peaks = peakdet(POWn,delta*NP.amax(POWn))[0]
  if PLOT:
    plt.plot(ls,POWn)
    plt.plot(peaks[:,0]+lmin,peaks[:,1],'*')
  ELL   = []
  FWHM  = []
  AMP   = []

  for ii in range(peaks.shape[0]):
    peak = peaks[ii]
    pmode = len(peaks) - ii

    try:
      limits =  Determine_Windows_L(NP.array(peaks[:,0]),Lmax=ls[-1])[-1][ii]
      li = int(limits[0]); ri = int(limits[1])
      # li = int(peak[0]-5); ri = int(peak[0]+5)
      p_opt,pp = curve_fit(lorenz_FWHM,ls[li:ri],POWn[li:ri],p0=[peak[0]+lmin,(ri-li)/2,peak[1]])
      if p_opt[1] > 100:
        raise Exception('FWHM unreasonable')
      if PLOT:
        plt.plot(ls,lorenz_FWHM(ls,*p_opt))
      ELL.append(p_opt[0])
      FWHM.append(p_opt[1])
      AMP.append(p_opt[2])
    except:
      if VERBOSE:
        print(( 'fail for p_%i mode' % pmode))
      ELL = []
      FWHM =[]
      AMP = []
  return ELL,FWHM,AMP,(peaks[:,0]+lmin)


def Determine_Windows_L(peaks, Lmax = 1000, PLOT = False):
  # Using the given peaks, build symmetric windows around the modes
  # PLOT plots the window functions

  # Deterimine the separation distances between modes
  windowhalfwidth = (peaks[1:] - peaks[:-1])/2
  windowhalfwidth = NP.concatenate([windowhalfwidth,[windowhalfwidth[-1]]])

  # Using the separation distance, determine the closest mode and use that as width of window
  newBounds = []
  for i in range(len(peaks)):
    if i == 0:
      newBounds.append(peaks[i] - windowhalfwidth[0])
      newBounds.append(peaks[i] + windowhalfwidth[0])
    elif i == (len(peaks)-1):
      newBounds.append(peaks[i] - windowhalfwidth[-1])
      newBounds.append(peaks[i] + windowhalfwidth[-1])
    else:
      newBounds.append(peaks[i] - min([windowhalfwidth[i],windowhalfwidth[i-1]]))
      newBounds.append(peaks[i] + min([windowhalfwidth[i],windowhalfwidth[i-1]]))

  newBounds = NP.array(newBounds)
  newBounds = newBounds.reshape((len(peaks),2))

  # Build array of smooth rectangle window functions
  WINDOWS = NP.zeros((len(peaks),Lmax))
  for i in range(len(peaks)):
    WINDOWS[i,:] = smoothRectangle(NP.arange(Lmax),newBounds[i,0]+0.05*peaks[i],newBounds[i,1]-0.05*peaks[i],0.5)

    if PLOT:
      plt.plot(WINDOWS[i]*0.5)

  return WINDOWS,newBounds


def matchModes_kernel_FWHMdata(L0,F0,N0,Kernel = None,Noise=False,dataSet = 'Total'):
  if dataSet.upper() == 'KORZENNIK':
    FWHM_data = NP.genfromtxt('/home/hanson/mps_montjoie/data/Observations/FWHM_OBS/FWHM_DATA_k.dat')
  elif dataSet.upper() == 'LARSON':
    FWHM_data = NP.genfromtxt('/home/hanson/mps_montjoie/data/Observations/FWHM_OBS/FWHM_DATA_l.dat')
  elif dataSet.upper() == 'SYNTHETIC':
    FWHM_data = NP.genfromtxt('/home/hanson/mps_montjoie/data/Observations/FWHM_OBS/FWHM_DATA_synth.dat')
  elif dataSet.upper() == 'TOTAL':
    FWHM_data = NP.genfromtxt('/home/hanson/mps_montjoie/data/Observations/FWHM_DATA.dat')
  else:
    FWHM_data = NP.genfromtxt(dataSet)

  FWHM      = NP.zeros(L0.shape)
  dFWHM     = NP.zeros(L0.shape)
  j=0
  missing = []
  for i in range(len(L0)):
    ell = L0[i]; nn = N0[i];
    MAT = (FWHM_data[:,0].astype(int) == ell)&(FWHM_data[:,1] == nn)
    FWHM_tmp  = FWHM_data[MAT,-2]
    dFWHM_tmp = FWHM_data[MAT,-1]
    try:
      if FWHM_tmp[0] < 0:
        FWHM_tmp[0] = 0
        j = j+1
        missing.append(i)
      if dFWHM_tmp[0] > 50:
        FWHM_tmp[0] = 0
        j = j+1
        missing.append(i)
      if (FWHM_tmp[0] - dFWHM_tmp[0] < 0):
        FWHM_tmp[0] = 0
        j = j+1
        missing.append(i)
      FWHM[i] = FWHM_tmp[0]
      dFWHM[i] = dFWHM_tmp[0]
    except: 
      j = j+1
      missing.append(i)
  print(('Number of missing data %i/%i' % (j,len(L0))))
  N0       = NP.delete(N0,missing,axis=0)
  L0       = NP.delete(L0,missing,axis=0)
  F0       = NP.delete(F0,missing,axis=0)
  FWHM     = NP.delete(FWHM,missing,axis=0)
  dFWHM     = NP.delete(dFWHM,missing,axis=0)

  if Kernel is not None:
    Kernel   = NP.delete(Kernel,missing,axis=0)
    if Noise:
      return Kernel,L0,F0,N0,FWHM*1e-6,dFWHM*1e-6,missing
    else:
      return Kernel,L0,F0,N0,FWHM*1e-6,missing
  else:
    if Noise:
      return L0,F0,N0,FWHM*1e-6,dFWHM*1e-6,missing
    else:
      return L0,F0,N0,FWHM*1e-6,missing


def damping_callForwardSolver(baseinitFile,dampingProfile,r=None,omega=None,clean=True,outFolderName = None,Run=True,Iteration=0):
  params = parameters(baseinitFile,TypeOfOutput.Surface1D)

  # clear the previous damping notes
  params.config_.remove_option('Damping')
  params.config_.remove_option('DampingSpatial')
  params.config_.remove_option('DampingRW')


  if not hasattr(dampingProfile,'__len__') and not isinstance(dampingProfile,str):
    params.config_.set('Damping','CONSTANT %1.16e' % dampingProfile)
  elif isinstance(dampingProfile,str):
    params.config_.set('Damping','%s' % dampingProfile)
  elif dampingProfile.ndim == 1:
    if r is None:
      raise Exception('1D damping profile used, must provide r grid')
    outFile = os.getcwd() + '/dampingProfile.txt'
    NP.savetxt(outFile,NP.array([r,dampingProfile]).T)
    params.config_.set('Damping','CONSTANT 1.')
    params.config_.set('DampingSpatial','RADIAL %s' % outFile)
  elif dampingProfile.ndim == 2:
    if omega is None or r is None:
      raise Exception('2D damping profile used, must provide omega and r grid')
    if dampingProfile.shape != (len(omega),len(r)) and dampingProfile.shape != (len(r),len(omega)):
      raise Exception('2D Damping profile shape does not match given r and omega')
    elif dampingProfile.shape == (len(r),len(omega)):
      dampingProfile = dampingProfile.T
    outFile = os.getcwd() + '/dampingProfile_%i.npy' % Iteration
    NP.save(outFile,[omega,r,dampingProfile])
    params.config_.set('DampingRW','FILE %s' % outFile)

  if outFolderName is not None:
    params.config_.set('OutDir',outFolderName)


  initFiletmp = os.getcwd() + '/initFile_%i.init' % Iteration
  params.config_.save(initFiletmp)

  if Run:
    os.system('cd %s/pyCompHelio/RunMontjoie/; ./runMontjoie.py %s -v -c -np8 %s' % (pathToMPS(),initFiletmp,['','-dc'][int(clean)]))
  

  paramsNEW = parameters(initFiletmp,TypeOfOutput.Surface1D)
  initFileNEW = paramsNEW.config_('OutDir') + initFiletmp.split('/')[-1]

  if clean:
    os.system('rm -rf %s %s' % (outFile,initFiletmp))

  return initFileNEW



def sort_Modes(L0,F0,FWHM0,L1,F1,FWHM1,Noise=None):
  Fc,Lc,Nc = NP.load('/home/hanson/mps_montjoie/CompHelioWork/DAMPING_INVERSIONS/1_CURRENT/Modes_MJ_Complete.npy')
  N0 = []
  for i in range(len(L0)):
    modeRange = (Lc == L0[i])
    N0.append(Nc[modeRange][NP.argmin(abs(Fc[modeRange] - F0[i]))])
  N1 = []
  for i in range(len(L1)):
    modeRange = (Lc == L1[i])
    N1.append(Nc[modeRange][NP.argmin(abs(Fc[modeRange] - F1[i]))])
  N0 = NP.array(N0)
  N1 = NP.array(N1)
  deleteInd = []
  for i in range(len(L0)):
    tmp = NP.sum((L1 == L0[i])*(N1 == N0[i]))
    if not tmp:
      deleteInd.append(i)

  deleteInd2 = []
  for i in range(len(L1)):
    tmp = NP.sum((L0 == L1[i])*(N0 == N1[i]))
    if not tmp:
      deleteInd2.append(i)

  if Noise is None:
    return NP.delete(L0,deleteInd),NP.delete(F0,deleteInd),NP.delete(FWHM0,deleteInd),NP.delete(N0,deleteInd),NP.delete(L1,deleteInd2),NP.delete(F1,deleteInd2),NP.delete(FWHM1,deleteInd2),NP.delete(N1,deleteInd2)
  else:
    return NP.delete(L0,deleteInd),NP.delete(F0,deleteInd),NP.delete(FWHM0,deleteInd),NP.delete(N0,deleteInd),NP.delete(L1,deleteInd2),NP.delete(F1,deleteInd2),NP.delete(FWHM1,deleteInd2),NP.delete(N1,deleteInd2),NP.delete(NP.delete(Noise,deleteInd,axis=1),deleteInd,axis=0)
