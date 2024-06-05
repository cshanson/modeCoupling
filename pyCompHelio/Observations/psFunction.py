import numpy             as NP
import matplotlib.pyplot as PLOT

from scipy.optimize import leastsq

from ..Common     import *
from ..Parameters import *

class psFunction(object):
    ''' Class defining a callable Ps Function for other structure
        such as crossCorrelation 
    '''

    def __init__(self,params,pstype,mean=0.e0,sd=1.e0,amp=1.e0):

      self.omega_ = params.time_.omega_
      self.type_  = pstype
      if not pstype.lower() in ['one','gaussian','lorenz','lorenz_fwhm','stein','hanson']:
        print(bColors.warning() + 'ps function type not known.')
        print("Known types are 'one','gaussian','lorenz','lorenz_FWHM','stein','hanson'.")
        print('Will be set to 1.')
        self.type_ = 'one'
      self.mean_  = mean
      self.sd_    = sd
      self.amp_   = amp

    def __call__(self,freq=None,ifreq=None,omega=None):

      if freq is not None:
        x = freq
      elif ifreq is not None:
        x = self.omega_[ifreq]/(2.e0*NP.pi)
      elif omega is not None:
        x = omega/(2.e0*NP.pi)
      else:
        x = self.omega_/(2.e0*NP.pi)
          
      if self.type_.lower() == 'one':
        if hasattr(x,'shape'):
          return NP.ones(x.shape)
        else:
          return 1
      elif self.type_.lower() == 'gaussian':
        return PsGaussian(x,self.mean_,self.sd_,self.amp_)
      elif self.type_.lower() == 'lorenz':
        return PsLorenz(x,self.mean_,self.sd_,self.amp_)
      elif self.type_.lower() == 'lorenz_fwhm':
        return PsLorenz(x,self.mean_,self.sd_,self.amp_,True)
      elif self.type_.lower() == 'stein':
        return PsStein(x,self.mean_,self.sd_,self.amp_)
      elif self.type_.lower() == 'hanson':
        return PsHanson(x,self.mean_,self.sd_,self.amp_)
      elif self.type_.lower() == 'smoothrectangle':
        return PsSmoothRectangle(x,self.mean_,self.sd_,self.amp_)
     
def PsGaussian(x,mean,sd,amp=1.e0):
    return gaussian(abs(x),mean,sd,amp)

def PsLorenz(x,mean,sd,amp=1.e0,FWHM = False):
    if FWHM:
      return lorenz_FWHM(abs(x),mean,sd,amp)
    else:
      return lorenz(abs(x),mean,sd,amp)

def PsStein(x,alpha,sd,amp=1.e0):
    return amp*(2.e0/NP.sqrt(NP.pi))*(x**2/sd**3)*NP.exp(-(x/sd)**2)
  
def PsHanson(x,alpha,sd,amp=1.e0):
    return amp*(2.e0/NP.sqrt(NP.pi))*(x**alpha/sd**3)*NP.exp(-(x/sd)**2)

def PsSmoothRectangle(x,alpha,sd,amp=1.e0):
    return smoothRectangle(x,alpha-sd,alpha+sd,amp)

def sumPsFunctions(params,means,sigmas,amps,distType='GAUSSIAN'):
    ''' returns the sum of several Gaussians or Lorenzians '''

    freq = params.time_.omega_/(2.e0*NP.pi)
    Ps   = NP.zeros(freq.shape)
    for i in range(len(means)):
      if distType == 'GAUSSIAN':
        Ps += PsGaussian(x,means[i],sigmas[i],amps[i])
      elif distType == 'LORENZ':
        Ps += PsLorenz(x,means[i],sigmas[i],amps[i])
      else:
        raise Exception('Please choose a valid distribution option (GAUSSIAN/LORENZ)')
    return Ps/np.amax(Ps)

def fitPsToDistribution(x,y_target,initGuess=[0,1.,1.],distType='Gaussian'):
    ''' Performs a fit of a sum of Ps Profiles towards a given value y_target.
        Parameters of ps profiles are listed as follows:
        [mean0, mean1, ..., sd0, sd1, ..., amp0, amp1, ...]
    '''

    def psFittingError(p,y_target,x,dist='Gaussian'):
      # NOTE THAT mean == alpha in the case of Lorenz2
      nDist = len(p)/3
      means = p[       :  nDist]
      sds   = p[  nDist:2*nDist]
      amps  = p[2*nDist:3*nDist]
      y_fit = np.zeros(len(x))
      for i in range(len(means)):
        if dist == 'Gaussian':
          y_fit += PsGaussian(x,means[i],sds[i],amps[i])
        elif dist == 'Lorenz':
          y_fit += PsLorenz(x,means[i],sds[i],amps[i])
        elif dist == 'Stein':
          y_fit += PsStein(x,means[i],sds[i],amps[i])
        elif dist == 'Hanson':
          y_fit += PsHanson(x,means[i],sds[i],amps[i])
      return y_target-y_fit

    distParams = leastsq(psFittingError,initGuess,args=(y_target,x,distType))[0]
    
    nDist = len(distParams)/3
    means = distParams[       :  nDist]
    sds   = distParams[  nDist:2*nDist]
    amps  = distParams[2*nDist:3*nDist]

    # Reconstruct estimated Ps
    y_est = NP.zeros(len(x))

    for i in range(nDist):
      print('\nDistribution %1i: '%i,)
      if distType == 'Gaussian':
        y_est += PsGaussian(x,means[i],sds[i],amps[i])
        print('x0 =',)
      elif distType == 'Lorenz':
        y_est += PsLorenz  (x,means[i],sds[i],amps[i])
        print('x0 =',)
      elif distType == 'Stein':
        y_est += PsStein(x,means[i],sds[i],amps[i])
        print('alpha = ',)
      elif distType == 'Hansons':
        y_est += PsHanson(x,means[i],sds[i],amps[i])
        print('alpha =',)
      print('%1.4e, SD = %1.4e, AMP = %1.4e\n' % (means[i],sds[i],amps[i]))
   
    # Plot observations and fit 
    PLOT.figure()
    PLOT.rc('text',usetex=True)
    PLOT.rc('font',family='serif')
    PLOT.plot(x,y_target,'b-' ,label=r'Real Data')
    if   distType == 'Gaussian':
      PLOT.plot(x,y_est,'g.-',label=r'Fitted Gaussians')
    elif distType == 'Lorenz':
      PLOT.plot(x,y_est,'r'  ,label='Fitted Lorenzians')
    elif distType == 'Stein':
      PLOT.plot(x,y_est,'k.-',label=r'$P_s=\omega^\alpha\times$ Gaussian',markevery=15)
    elif distType == 'Hanson':
      PLOT.plot(x,y_est,'c*' ,label=r'$P_s=\omega^\alpha\times$ Gaussian',markevery=15)
    PLOT.legend()
    try:
      PLOT.show()
    except:
      pass
    
    return [means,sds,amps]
