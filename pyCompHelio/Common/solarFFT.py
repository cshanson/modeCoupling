import numpy as NP
import scipy as SP
import copy

# This module contains the definition of fft and ifft 
# with the correct conventions and constants for helioseismology

# It supposes that field contains first the spatial coordinates then the temporal ones and 
# that the zero frequencies are first (in real and Fourier space).

from .misc import *

def temporalFFT(field,time,N=None,nextPower2=False,returnNewTime=False,axis=-1):
    ''' FFT in time domain. 
        if N is greater thant the number of points of axis -1,
        adds zeros before and after +/- maxT to get more points 
    '''

    if N is not None or nextPower2:
      data = padZeros(field,N,axis,nextPower2)
    else:
      data = field

    newField = NP.fft.ifft(data,axis=axis)*time.ht_ / (2*NP.pi)*field.shape[axis]

    if returnNewTime:
      newTime = copy.deepcopy(time)
      newTime.setObservationalTime(time.T_,newField.shape[axis])
      return newField,newTime 
    else:
      return newField

def temporalIFFT(field,time,N=None,nextPower2=False,returnNewTime=False,axis=-1,itmin=0,itmax=-1):
    ''' IFFT in time domain. 
        if N is greater thant the number of points of axis -1,
        adds zeros before and after +/- maxT to get more points 
    '''

    if N is not None or nextPower2:
      data = padZeros(field,N,axis,nextPower2)
    else:
      data = field

    newField = NP.fft.fft(data,axis=axis)


    if itmax==-1: # to have the last element after slice
      itmax = newField.shape[axis] 
   
    s       = [slice(None)]*newField.ndim
    s[axis] = slice(itmin,itmax)
    if returnNewTime:
      newTime = copy.copy(time)
      newTime.setObservationalTime(time.T_,newField[s].shape[axis])
      return newField[s]* newTime.homega_, newTime 
    else:
      return newField[s]* time.homega_

def spatialFFT(field,geom):
  return NP.fft.fft2(field,axes=(0,1))*geom.h_[0]*geom.h_[1]/(2*NP.pi)**2

def spatialIFFT(field,geom):
  return NP.fft.ifft2(field,axes=(0,1))*geom.hk_[0]*geom.N_[0]*geom.hk_[1]*geom.N_[1]

def FFTn(field,time,geom):
  fieldkt = spatialFFT(field,geom)
  return temporalFFT(fieldkt,time)

def IFFTn(field,time,geom):
  fieldxw = spatialIFFT(field,geom)
  return temporalIFFT(fieldxw,time)

def testRealFFT(field,eps=1e-10,message=''):
    ''' Tests if the FFT is real and remove the small imaginary part if needed
    '''
    if (NP.amax(NP.imag(field)) > eps * NP.amax(NP.real(field))):
      print((bColors.warning() + ' ' + message + ' field is not real, max(real) = %.2E, max(imag) = %.2E \n'\
            %(NP.amax(NP.real(field)),  NP.amax(NP.imag(field)))))
    return NP.real(field)

def polarFFT(field, geom):
    ''' Field is 2D (r,w). Return f(k,w) = \int r f(r,w) J_0(k r) dr 
    '''
    midX      = int(geom.N_[0]/2) # to integrate only for 0 for L
    J0        = SP.special.jv(0,geom.k_[0][NP.newaxis,NP.newaxis,:] * geom.coords_[0][midX:,NP.newaxis,NP.newaxis])
    integrand = geom.coords_[0][midX:,NP.newaxis,NP.newaxis]*field[midX:,:,NP.newaxis]*J0
    fourier   = SP.integrate.simps(integrand,geom.coords_[0][midX:],axis=0)
    fourier   = NP.swapaxes(fourier,0,1)/(2.e0*NP.pi)
    return fourier

def polarIFFT(field, geom):
    ''' Returns the polar ifft for a 2D field in (k,w) that does not depend on the angle. 
        Return f(r,w) = \int k f(k,w) J_0(k r) dk
    '''
    midX      = int(geom.N_[0]/2) # to integrate only for 0 for k_max
    J0        = SP.special.jv(0, geom.k_[0][0:midX, NP.newaxis,NP.newaxis]*geom.coords_[0][NP.newaxis, NP.newaxis,:])
    J0       *= NP.ones((midX,field.shape[1],field.shape[0]))
    integrand = J0*geom.k_[0][0:midX,NP.newaxis, NP.newaxis]*field[0:midX,:,NP.newaxis]
    fourier   = SP.integrate.simps(integrand,geom.k_[0][0:midX],axis=0)
    fourier   = NP.swapaxes(fourier,0,1)*2.e0*NP.pi
    return fourier

def powerBitLength(x):
  ''' Next power of 2 after x
  '''
  return 2**(x-1).bit_length()

def padZeros(data,N=None,Axis=-1,nextPower2=False):
    ''' Add zeros for FFT operations.
        N is the final number of points in dimension Axis
        (must be greater than array size)
        if nextPower2, N will be set to the next power of 2
        to make FFT faster
    '''

    Nold = data.shape[Axis]
    if N is not None:
      if Nold > N:
        print((bColors.WARNING + 'Warning ' + bColors.ENDC +\
              ': temporalFFT, N must be greater than size of array in last dimension (Currently N: %d size: %d)' % (N,Nold)))
      if nextPower2 :
        N = powerBitLength(N)
    else:
      N = Nold
      if nextPower2 :
        N = powerBitLength(N)
    padNb = (N-Nold)/2

    pads = []
    if Axis == -1:
      Axis = data.ndim-1
    for dim in range(data.ndim):
      if dim == Axis:
        pads.append((padNb,padNb))
      else:
        pads.append((0,0))

    return NP.fft.ifftshift(NP.pad(NP.fft.fftshift(data,axes=Axis),pads,'constant',constant_values=(0,0)),axes=Axis)

def phiFFT(field,N=None,nextPower2=False,returnNewGeom=False,axis=-1,geom=None):

    if N is not None or nextPower2:
      data = padZeros(field,N,axis,nextPower2)
    else:
      data = field

    newField = NP.fft.fft(data,axis=axis)

    if returnNewGeom:
      if geom is None:
        print ("No geometry was given, won't return the new one")
        return newField
      newGeom = copy.deepcopy(geom)
      newGeom.setComponent(-1,data.shape(axis))
      return newField,newGeom
    else:
      return newField

def phiIFFT(field,N=None,nextPower2=False,returnNewGeom=False,axis=-1,geom=None):

    if N is not None or nextPower2:
      data = padZeros(field,N,axis,nextPower2)
    else:
      data = field

    newField = NP.fft.ifft(data,axis=axis)*data.shape[axis]

    if returnNewGeom:
      if geom is None:
        print ("No geometry was given, won't return the new one")
        return newField
      newGeom = copy.deepcopy(geom)
      newGeom.setComponent(-1,data.shape(axis))
      return newField,newGeom
    else:
      return newField



