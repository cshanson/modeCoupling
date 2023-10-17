import numpy as NP
import sys
import time
import os
try:
  from astropy.io import fits
except:
  pass

from .misc import *
#from ..Parameters import InitFileMJ

def writePowerSpectrum2FITS(self,FILEoutput,data,params,Lmin,Lmax,Lstep,Method):
  hdr = writeFITShdr(data,params)
  hdr.header.set('Lmin',Lmin  ,'Minimum harmonic degree')
  hdr.header.set('Lmax',Lmax  ,'Maximum harmonic degree')
  hdr.header.set('dL'  ,Lstep ,'Step size of harmonic Degree')
  hdr.header.set('Meth',Method,'Numerical Integration scheme used')
  fits.writeto(FILEoutput,data,hdr.header)

def writeCrossCovariance2FITS(self,FILEoutput,data,params):
  hdr = self.writeFITShdr(data,params)
  hdr.header.set('dDelta',NP.pi/params.geom_.Ntheta(),'Resolution in Delta')
  fits.writeto(FILEoutput,data,hdr.header)

def writeFITShdr(self,data,params):
  data  = NP.array(data)
  hdu   = fits.PrimaryHDU()
  naxis = len(data.shape)
  hdu.header.set('NAXIS',naxis)
  for i in range(1,naxis+1):
    hdu.header.set('NAXIS%i' %(i),data.shape[i-1],'size of dimension %i' %(i))
  hdu.header.set('DATE'   ,time.strftime("%d %m %Y"), ' Date created')
  hdu.header.set('Creator',os.getlogin(), 'Creator of FITS file')
  hdu.header.set('CADENCE',params.time_.ht_,'Cadence for Data file (seconds)')
  hdu.header.set('dFreq'  ,params.time_.homega_/(2.e0*pi),'Frequency Resolution')
  return hdu






