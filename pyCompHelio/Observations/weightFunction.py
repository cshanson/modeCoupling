import numpy             as NP
import matplotlib.pyplot as PLOT
import scipy.integrate   as ITG
import copy

from ..Common     import *
from ..Parameters import *

class weightFunction:
  ''' This class contains the weight function to compute kernels and travel times. 
      The definition is according to (Gizon & Birch 2004) for travel time 
      and (Nagashima et al. 2017) for amplitude.
  '''

  def __init__(self,windowFunction,typeOfMeas='tau',checkPlot=False):
    self.windowFunction_ = windowFunction
    self.typeOfMeas_     = typeOfMeas
    self.XS_             = windowFunction.XS_
    self.checkPlot_      = checkPlot

  def __call__(self,travelTimeType,G=None,Cw=None,coords=None,reversePoints=False):
    return self.Ww(travelTimeType,G,Cw,coords,reversePoints)

  def Ww(self,travelTimeType,G=None,Cw=None,coords=None,reversePoints=False):
    ''' Returns the weight function in frequency domain using 
        the cross covariance of G at point coords
        coords must be a tuple of coordinates matching params.geom_
    '''
    time = self.windowFunction_.params_.time_
    Wxw  = 2.e0*NP.pi*solarFFT.temporalFFT(self.Wt(travelTimeType,G,Cw,coords,reversePoints),time)
    return Wxw

  def Wt(self,travelTimeTypes,G=None,Cw=None,coords=None,reversePoints=False):   
    ''' Returns the weight functions for the different type of travel times computed from 
        a time CrossCorrelation at a point  times the window function 
    '''

    if VERBOSE:
      print('Computing weight function in t-space')

    # Compute the cross-covariance in the time domain from the Green's function
    time = self.windowFunction_.params_.time_
    if Cw is None:
      green = G(coords=coords)[0]
      if self.windowFunction_.params_.unidim_:
        # Put 0 for the frequency 0 in the 1D code
        green[0] = 0.
      Cw   = self.XS_(data=green)
    Ct   = solarFFT.temporalIFFT(Cw,time)
    # Test that the time XS is real
    Ct   = solarFFT.testRealFFT(Ct,message='In weighting function: FFT of XS')

    # For travel time weighting function compute the derivative in time of C
    if self.typeOfMeas_ == 'tau':
      dCw = -1j * Cw * time.omega_
      dCt = solarFFT.temporalIFFT(dCw, time)
      dCt = solarFFT.testRealFFT(dCt,message='In weighting function: FFT of dXS')

    # Format the window function to be of the same size that C 
    #if G is None:
      # If the cross-covariance is given as argument, the window function is the same for all distances
    #  Wplus = NP.zeros(Cw.shape)
    #  Wminus = NP.zeros(Cw.shape)
    #  self.windowFunction_.compute()
    #  for i in range(Cw.shape[0]):
    #    Wplus[i,:]  = self.windowFunction_.FPlus_
    #    Wminus[i,:] = self.windowFunction_.FMinus_

    #else:
    #  # Compute the window function for all distances
    #  if coords is None:
    #    coords = G.params_.geom_.theta()
    #  Wplus = NP.zeros((len(coords), time.Nt_))
    #  Wminus = NP.zeros((len(coords), time.Nt_))
    #  for i in range(len(coords)):
    #    self.windowFunction_.compute(G,coords[i])
    #    Wplus[i,:]  = self.windowFunction_.FPlus_
    #    Wminus[i,:] = self.windowFunction_.FMinus_

    self.windowFunction_.compute(coords,G)
    Wplus  = self.windowFunction_.FPlus_
    Wminus = self.windowFunction_.FMinus_
    # Compute W_+ and W_- for travel time or amplitude
    if self.typeOfMeas_ == 'tau':
      Wtp    = -Wplus*dCt /(NP.trapz(Wplus *dCt**2)*time.ht_)
      Wtm    = Wminus*dCt /(NP.trapz(Wminus*dCt**2)*time.ht_)
    elif self.typeOfMeas_ == 'amplitude':
      Wtp    = Wplus*Ct /(NP.trapz(Wplus *Ct**2)*time.ht_)
      Wtm    = Wminus*Ct /(NP.trapz(Wminus*Ct**2)*time.ht_)
    else:
      raise Exception('This type of measurement %s is not implemented. The possibilities are tau or amplitude.' % self.typeOfMeas_)


    Wt = []
    if (isinstance(travelTimeTypes, int)):
      travelTimeTypes = [travelTimeTypes]

    # If reversePoints, switch source and receiver so W_+ becomes W_- and reciprocally
    if reversePoints:
      Wtp_tmp = Wtp
      Wtp     = Wtm
      Wtm     = Wtp_tmp

    for i in range(len(travelTimeTypes)):
      if travelTimeTypes[i]   == TypeOfTravelTime.PLUS: 
        Wt.append(Wtp)
      elif travelTimeTypes[i] == TypeOfTravelTime.MINUS:
        Wt.append(Wtm)
      elif travelTimeTypes[i] == TypeOfTravelTime.MEAN:
        Wt.append( (Wtp + Wtm) / 2)
      elif travelTimeTypes[i] == TypeOfTravelTime.DIFF:
        Wt.append(Wtp - Wtm)
  
    # Check window and weighting function
    if self.checkPlot_:
      self.plot(Ct,dCt,Wt[0],Wplus)

    if (len(travelTimeTypes) == 1):
      Wt = Wt[0]

    return Wt

    
  def plot(self,Ct,dCt,Wt,Wplus):
    time = self.windowFunction_.params_.time_
    PLOT.ion()
    fig = PLOT.figure()
    ax  = fig.add_subplot(111) 
    ax.plot(NP.fft.fftshift(time.t_)/60,NP.real(NP.fft.fftshift(Ct )/max(abs(Ct ))),'r',label='Crefxt' )
    ax.plot(NP.fft.fftshift(time.t_)/60,NP.real(NP.fft.fftshift(dCt)/max(abs(dCt))),'g',label='dCrefxt')
    ax.plot(NP.fft.fftshift(time.t_)/60,NP.real(NP.fft.fftshift(Wt )/max(abs(Wt ))),'b',label='Wxt'    )
    ax.plot(NP.fft.fftshift(time.t_)/60,NP.fft.fftshift(Wplus), 'k', label='window')
    ax.legend(loc='upper right')
    PLOT.savefig('testW.png')
    if VERBOSE:
      print('Plot of weight functions saved in testW.png')


