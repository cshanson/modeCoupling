import numpy             as NP
import matplotlib.pyplot as PLOT
import copy
import scipy.interpolate

from ..Common     import *
from ..Parameters import *
from ..Observations import *

class windowFunction:
  ''' This class contains the window function used to select a wave packet. 
      Types are:
       - Heaviside function 
       - Gaussian (center and/or length of the window can be given)
       - Rectangle (idem) 
       - Rectangle (with interactive pick of left and right limits of the window)
  ''' 

  def __init__(self,windowType,params,XS=None,center=None,length=None,\
                    method='fit',customWindow=None,\
                    checkWindow=False,\
                    limitFit=0.20,numSkip=1,tmax=30000):
    ''' WindowType: type of window function (see TypeOfWindowFunction in Enum), 
        params:     object of type parameter that contains information 
                    about the time parameters and the geometry. 

        point, length: if given the window is centered around point and has a width of lengthWindow

        method: method used to select the center of the window:
          - interactive: the user clicks to select its window
          - fit: the center and length of the window are determined by using the time-distance diagram and selecting a window that isolates the pick. It does not work properly for small and large distances.
          - ray: it uses the ray theory to find the center of the window. The length must be specified by hand. It only works for the first skip. 

        customWindow: if type is TypeOfWindowFunction.CUSTOM, given function for W+
    '''

    self.type_      = windowType
    self.params_    = params
    self.center_    = center
    self.length_    = length
    if XS is None:
      self.XS_ = toyModelCrossCorrelation(params)
    else:
      self.XS_ = XS

    self.method_       = method
    self.customWindow_ = customWindow
    self.plot_         = checkWindow
    self.limitFit_     = limitFit
    self.numSkip_      = numSkip
    self.tmax_         = tmax

  #============================================================================================================

  def __call__(self,coords=None,G=None,fittmax = 30000):
    return self.compute(coords,G,fittmax)

  def compute(self,coords=None,G=None,fittmax = 30000):
    ''' G is an instance of green's function class 
        coords is a tuple in the same coordinates system
        as G.params_.geom_,
        G can be replaced with the reference cross covariance array (at chosen distance)'''

    if VERBOSE:
      print("\nComputing window function")
    t  = self.params_.time_

    self.FPlus_  = NP.zeros((t.Nt_))
    self.FMinus_ = NP.zeros((t.Nt_))

    # ============================================
    # User specified window function

    if self.type_ == TypeOfWindowFunction.CUSTOM:
    
      if self.customWindow_ is None:
        print("\033[93mWARNING\033[0m")
        print("Custom window was not given as argument of windowFunction.compute with type CUSTOM.")
        print("Will set window function to 1.")
        return 1
      else:
        self.FPlus_         = self.customWindow_ 
        self.FMinus_[:0:-1] = self.customWindow_[1:]

    # ==============================================
    # Heaviside

    elif self.type_ == TypeOfWindowFunction.HEAVISIDE:

      self.FPlus_ [t.t_ >= 0] = 1
      self.FMinus_[t.t_ <= 0] = 1

    # ================================================
    # Other types : if not specified, point and length
    # must be either picked of guessed from fit

    else:

      if (self.center_ is None) or (self.length_ is None) or self.method_ == 'ray':
        if self.method_ == 'interactive':
          self.center_,self.length_ = self.selectWindow(G,coords,self.tmax_)
        elif self.method_ == 'fit':
          self.center_,self.length_ = self.fitWavelet(G,coords,self.numSkip_,self.limitFit_,fittmax)
        elif self.method_ == 'ray':
          if self.numSkip_ != 1:
            raise NotImplementedError('The selection of the window using the ray theory is only implemented for the first skip.')
          self = self.useRayTheory(coords)
          if self.length_ is None:
            raise Error('You need to specify the length of your window function')
        else:
          raise NotImplementedError('This type of method to select the center of the window function is not implemented')

      # ===================================================
      # Rectangle

      if self.type_ == TypeOfWindowFunction.RECTANGULAR:

        self.FPlus_ [abs(t.t_ - self.center_) < self.length_/2.e0] = 1 
        self.FMinus_[abs(t.t_ + self.center_) < self.length_/2.e0] = 1

      # ===================================================
      # Smoothed rectangle

      elif self.type_ == TypeOfWindowFunction.SMOOTHRECTANGLE:

        tm = self.center_-self.length_/2.e0
        tp = self.center_+self.length_/2.e0

        self.FPlus_  = smoothRectangle(t.t_, tm, tp,0.2)
        self.FMinus_ = smoothRectangle(t.t_,-tp,-tm,0.2)

      # ===================================================
      # Gaussian

      elif self.type_ == TypeOfWindowFunction.GAUSSIAN:

        self.FPlus_  = NP.exp(-(t.t_ - self.center_)**2 / (2.e0 * self.length_**2))
        self.FMinus_ = NP.exp(-(t.t_ + self.center_)**2 / (2.e0 * self.length_**2))
       
      else:
        raise Exception('Window type not known')

    if VERBOSE:
      self.display()

    if self.plot_:
      self.plot(G,coords)

  #============================================================================================================

  def selectWindow(self,G,coords,tmax=30000):
    ''' Sets a window centered between two points selected interactively '''

    t     = self.params_.time_
    itmax = NP.argmin(abs(tmax*t.ht_-t.t_))

    if type(G).__module__ == 'numpy':
      Cxt = solarFFT.temporalIFFT(G,t)
    else:
      Cxt = solarFFT.temporalIFFT(self.XS_(data=G(coords=coords)[0]),t)

    def onclick(event):
      global ix, iy
      ix, iy = event.xdata, event.ydata
      # print 'x = %d'%ix

      global Coords
      Coords.append((ix, iy))

      # close after 2 clicks
      if len(Coords) == 2:
          fig.canvas.mpl_disconnect(cid)
          plt.close(1)
      return

    fig = plt.figure()
    plt.plot(NP.real(Cxt[:itmax]))
    plt.show()
    fig.canvas.manager.window.raise_()

    global Coords
    Coords = []
    cid    = fig.canvas.mpl_connect('button_press_event',onclick)
    print('Please click on the Plot Twice for the LEFT/RIGHT points of the Window')
    input('press ENTER to Continue')

    wLength = abs(Coords[0][0]-Coords[1][0])
    wCentre = wLength/2. + min(abs(NP.array([Coords[0][0],Coords[1][0]])))

    return wCentre*t.ht_,wLength*t.ht_

  #============================================================================================================

  def fitWavelet(self,G,coords,numSkip=1,limit=0.18,tmax=30000):
    ''' fits a gaussian wavelet to detect the position of the center of first skip
        be aware that the skips must be clearly distinct to obtain a nice fit.
    '''
    if VERBOSE:
      print("Fitting cross-correlation with a gaussian wavelet.")
    t = self.params_.time_

    # Fourier transform the signal in time:
    if type(G).__module__ == 'numpy':
      G = G
    else:
      G = self.XS_(data=G(coords=coords)[0])
    G = NP.nan_to_num(G)
    Cxt = solarFFT.temporalIFFT(G,t)
    solarFFT.testRealFFT(Cxt,message='Window function, fit wavelet')
    Cxt = NP.real(Cxt)

    # Get enveloppe of signal
    from scipy import signal
    from scipy import optimize
    H = NP.imag(scipy.signal.hilbert(Cxt))
    E = NP.sqrt(Cxt**2+H**2)

    # List maxima of E and get rid of noise maxima
    indz    = NP.argmin(abs(t.t_ - tmax))
    Emax    = list(getExtrema(E[:indz],t.t_[:indz],'MAX',True))
    vmax    = NP.amax(Emax[1])

    Emax[0] = Emax[0][Emax[1]>limit*vmax]
    Emax[1] = Emax[1][Emax[1]>limit*vmax]

    if DEBUG:
      PLOT.ion()
      PLOT.figure()
      PLOT.plot(t.t_[:indz],Cxt[:indz])
      PLOT.plot(t.t_[:indz],E[:indz])
 
    if len(Emax[1])<2:
      raise Exception("Less than 2 maxima were detected in the enveloppe of F-1(Cxw)), meaning that the window function cannot be determined with a fit with a Gaussian wavelet.")

    # Get indices of lower and upper boundaries of window function in time
    if numSkip == 1:
      im = 0
    else:
      im = NP.argmin(abs(0.5*(Emax[0][numSkip-1]+Emax[0][numSkip-2])-t.t_))
    ip   = NP.argmin(abs(0.5*(Emax[0][numSkip  ]+Emax[0][numSkip-1])-t.t_))

    def gaussian(x,a,c,s):
      return a*NP.exp(-(x-c)**2/s)
    try:
      pms = scipy.optimize.curve_fit(gaussian,t.t_[im:ip],NP.real(E[im:ip]),p0=[Emax[1][numSkip-1],Emax[0][numSkip-1],100])
      # Get center and window limits
      if len(pms[0])==3:
        c = pms[0][1]
        if numSkip == 1:
          l = Emax[0][numSkip]-c
          if (c-l/2.) < 0:
            l = 2*c
        else:
          l = 0.5*(Emax[0][numSkip]-Emax[0][numSkip-2])

        return c,l
    except:
      raise Exception("Impossible to perform the fit of the Gaussian wavelet")

  #============================================================================================================

  def useRayTheory(self,coords, fRay=3.e-3, readFile = True):
    '''Find the center of the window function by computing the time it takes for the wave to reach the receiver using the Ray theory.'''
    w = fRay*2.*NP.pi
    if len(coords) > 1:
      raise NotImplementedError('Computation of the center of the window function only implemented for phi=0')
    thR = coords[0]

    lMax = 501
    ls = NP.arange(1,lMax)

    if not hasattr(self, 'thetas_'):
      fileName = '%s/data/Observations/lThRayPath.npy' % pathToMPS()
      if readFile:
        self.thetas_ = NP.load(fileName)
      else:      
        # Get the correspondance between l and the receiver location given by its angle theta
        #thetas = NP.zeros(len(ls))
        #for l in range(len(ls)):
        #  thetas[l] = getThetaLCorrespondance(self.params_,w,ls[l])
        thetas = reduce(getThetaLCorrespondance,(self.params_,w,ls),len(ls),8,progressBar=True)
        self.thetas_ = NP.array(thetas)
        NP.save(fileName, self.thetas_)

    # Returns the time associated with thR
    indL = NP.argmin(NP.abs(self.thetas_-thR))
    outRay = ray_path(w,ls[indL],self.params_)
    self.center_ = outRay[2][-2]
    return self


  #============================================================================================================

  def plot(self,G,coords,filename='windowCheck.png'):
      ''' Plots the window function over a reference signal Cxt '''

      t = self.params_.time_
      
      if type(G).__module__ == 'numpy':
        Cxt = solarFFT.temporalIFFT(G,t)
      else:
        Cxt = solarFFT.temporalIFFT(self.XS_(data=G(coords=coords)[0]),t)

      PLOT.ion()      
      fig  = PLOT.figure()
      ax   = fig.add_subplot(111)
      tmax = NP.argmin(abs(t.t_-300*60))
      PLOT.plot(t.t_[:tmax]/60, NP.real(Cxt[:tmax] / NP.amax(NP.real(Cxt[:tmax]))),'r', label='Reference signal')
      PLOT.plot(t.t_[:tmax]/60, self.FPlus_[:tmax],label='Window function')
      if self.type_ != TypeOfWindowFunction.HEAVISIDE:
        ax.set_xlim([(self.center_-1.5e0*self.length_)/60., (self.center_+1.5e0*self.length_)/60.])
      ax.set_ylim([-1.1,1.1])
      PLOT.legend(loc='upper right')
      PLOT.show()
      PLOT.savefig(filename)

  #============================================================================================================

  def display(self):
      print("---------------------------")
      print("Window function properties:")
      print("  Type:",TypeOfWindowFunction.toString(self.type_))
      print("  Center:", self.center_)
      print("  Length:", self.length_) 
      print("---------------------------")

  #============================================================================================================

  def createWithNewTimeParameters(self,newParams):

    # Interpolate custom function if needed
    newCustom = None
    if self.type_ == TypeOfWindowFunction.CUSTOM:
      FP        = scipy.interpolate.UnivariateSpline(self.params_.time_.t_,self.FPlus_)
      newCustom = FP(newParams.time_.t_)

    return windowFunction(self.type_,newParams,XS=self.XS_,center=self.center_,length=self.length_,customWindow=newCustom)

  #============================================================================================================
  #============================================================================================================

def getThetaLCorrespondance(params,w,l):
  '''Get the location of the receiver when the source is at the pole by following the ray path associated to the frequency w and the harmonic degree l.'''
  try:
    outRay = ray_path(w,l,params)
    return outRay[1][-1] # Get the last value of theta on the ray path
  except:
    return NP.inf
