from scipy.interpolate import RectBivariateSpline
import numpy as NP
import itertools       as IT
from   multiprocessing import Pool
import scipy.interpolate as INTERP

from . import *
from ..Parameters import *

class Projections:

  def __init__(self,MontjoieOutput, Nw,repmat=100, coords=[512,512,100,100],ProjType='Orthographic',NX=1024,Radius=1,vinc=0.0,vazi=0.0,PROJMAT=None,minimize = None,fInd=0,PSF=False):
    self.MJoutput_ = MontjoieOutput
    # self.greenxw_  = loadGreenFreqInd(MontjoieOutput,fInd,MFFT = True)
    self.xc_ = coords[0];self.yc_ = coords[1];self.dx_ = coords[2];self.dy_ = coords[3];
    self.ProjType_ = ProjType
    self.NX_       = NX
    self.Radius_   = Radius
    self.vinc_     = -vinc 
    self.vazi_     = vazi
    self.repmat_   = repmat
    self.Nw_       = Nw
    self.PROJMAT_  = PROJMAT
    self.minimize_ = minimize
    self.PSF_      = PSF 
    if MontjoieOutput.typeOfOutput_ == TypeOfOutput.Surface2D:
      self.repmat_ = 1

  @staticmethod
  def MappedImage(self,greenxw,supress=None):
    if len(greenxw.shape) == 1 and self.MJoutput_.typeOfOutput_ == TypeOfOutput.Surface1D:
      input = NP.tile(greenxw,[self.repmat_,1]).T
    else:
      input   = greenxw
    ptype   = self.ProjType_
    ngrids  = self.NX_
    R       = self.Radius_
    vinc    = self.vinc_
    vazi    = self.vazi_
    PROJMAT = self.PROJMAT_
    xc = self.xc_; yc = self.yc_; dx = self.dx_; dy = self.dy_

    NP.seterr(invalid='ignore')
  
    #########################
    # Initialization commands
    
    inc1         = NP.linspace(-NP.pi/2.0,NP.pi/2.0,input.shape[0])
    azi1         = NP.linspace(0         ,2*NP.pi  ,input.shape[1])
  
    # Latitude and longitude of viewing angle
    phi0    = vinc;    
    lambda0 = NP.pi-vazi;
  
    # Output grid
    xn = ngrids;
    yn = ngrids;
    x  = NP.linspace(-R,R,xn);
    y  = NP.linspace(-R,R,yn);
     
    # User recap 
      
    if (phi0 == 0) and supress == None:
      print ('- Viewing at the Equator')
    elif (phi0 == -NP.pi/2) and supress == None:
      print ('- Viewing above the North Pole')
    elif (phi0 == NP.pi/2) and supress == None:
      print ('- Viewing above the South Pole')
    elif supress == None:
      ANG = phi0*180./NP.pi
      print ('- Viewing from angle phi = %3.4f degrees' % ANG)
      
    #########################
    # PROJECTIONS
    if PROJMAT == None:
      [x,y] = NP.meshgrid(x,y)
      if (ptype == 'Postel'):
        if supress == None:
          print ('- Postel Projection Used')
        [rho,l,phi] = Projections.InversePostel(x,y,R,phi0,lambda0)
        
      elif (ptype == 'Orthographic'):
        if supress == None:
          print ('- Orthographic Projection used')
        [rho,l,phi] = Projections.InverseOrthographic(x,y,R,phi0,lambda0)
      else:
        print ('- No valid projection : use either <Postel> or <Orthographic>')

    else:
      rho  = PROJMAT[0]
      l    = PROJMAT[1]
      phi  = PROJMAT[2]
    #########################
    # GENERATION OF IMAGE GIVEN PROJECTIONS
    azi  = l - NP.pi
    inc  = phi
    azit = azi + 2*NP.pi*(azi<0)

    if self.minimize_ != None:
      inc  = inc[xc-dx:xc+dx,yc-dy:yc+dy]
      azit = azit[xc-dx:xc+dx,yc-dy:yc+dy]

  
    spliner     = RectBivariateSpline(inc1,azi1,NP.real(input))
    splinei     = RectBivariateSpline(inc1,azi1,NP.imag(input))
    psir        = spliner.ev(inc,azit)
    psii        = splinei.ev(inc,azit)
    psi         = psir + psii*1.j
    psi[rho>1] = NP.NAN 
  
    return [psi,rho,l,phi]     
  
  ############################## 
  # ACTUAL PROJECTION ROUTINES #
  ############################## 
  
  ################### 
  # Postel projection
  
  @staticmethod
  def InversePostel(x,y,R,phi0,l0):
  
    rho    = NP.sqrt(x**2+y**2)
    c      = rho/R
    kd     = c/NP.sin(c)
  
    l      = NP.empty(x.shape)
    l[:]   = NP.NAN
    phi    = NP.empty(x.shape)
    phi[:] = NP.NAN
  
    # iterate over x
    it = NP.nditer(x,flags=['multi_index'])
    while not (it.finished):
  
      ij = it.multi_index
      if (rho[ij] > R):
        rho[ij] = NP.NAN
      else:
        # Equator
        if (phi0==0):
          phi[ij] = NP.arcsin(y[ij]/(R*kd[ij]));
          l  [ij] = l0 + NP.arctan2(x[ij],R*kd[ij]*NP.cos(c[ij]));
  
        # North pole
        elif (phi0 == NP.pi/2):
          phi[ij] = NP.arcsin(NP.cos(c[ij]));
          l[ij]   = l0 + NP.arctan2(x[ij],(-y[ij]));
  
        # South pole
        elif (phi0 == -NP.pi/2):
          phi[ij] = NP.arcsin(-NP.cos(c[ij]));
          l[ij]   = l0 + NP.arctan2(x[ij],y[ij]);
  
        # Other points
        else:
          phi[ij] = NP.arcsin(NP.cos(c[ij])*NP.sin(phi0)+(y[ij]*NP.sin(c[ij])*NP.cos(phi0))/c[ij]);
          l[ij]   = l0 + NP.arctan2((x[ij]*NP.sin(c[ij])),(c[ij]*NP.cos(phi0)*NP.cos(c[ij])-y[ij]*NP.sin(phi0)*NP.sin(c[ij])));
            
            
      it.iternext()

    return [rho,l,phi]
  
  #######################################
  # Orthographic projection (we are gods)
  
  @staticmethod
  def InverseOrthographic(x,y,R,phi0,l0):
                
    rho    = NP.sqrt(x**2+y**2)
    c      = NP.arcsin(rho/R)
  
    l      = NP.empty(x.shape)
    l[:]   = NP.NAN
    phi    = NP.empty(x.shape)
    phi[:] = NP.NAN
  
  
    # iterate over x
    it = NP.nditer(x,flags=['multi_index'])
    while not (it.finished):
  
      ij = it.multi_index
      if (rho[ij] > R):
        rho[ij] = NP.NAN
      else:
        # Equator
        if (phi0==0):
          phi[ij] = NP.arcsin(y[ij]/R)
          l[ij]   = l0 + NP.arctan2(x[ij],NP.cos(NP.real(c[ij]))*R)
  
        # North pole
        elif (phi0 == NP.pi/2):
          phi[ij] = NP.arcsin(NP.cos(c[ij]))
          l[ij]   = NP.arctan2(x[ij],-y[ij])
  
        # South pole
        elif (phi0 == -NP.pi/2):
          phi[ij] = NP.arcsin(-NP.cos(c[ij]))
          l[ij]   = l0 + NP.arctan2(x[ij],y[ij])
          l[ij]   = NP.pi-l[ij]-l0
  
        # Other points
        else:
          phi[ij] = NP.arcsin(NP.cos(c[ij])*NP.sin(phi0)+(y[ij]*NP.sin(c[ij])*NP.cos(phi0))/rho[ij]);
          l[ij]   = l0 + NP.arctan2(x[ij]*NP.sin(c[ij]),(rho[ij]*NP.cos(phi0)*NP.cos(c[ij])-y[ij]*NP.sin(phi0)*NP.sin(c[ij])));
            
      it.iternext()
  
    return [rho,l,phi]

###################################
# Parallelize
###################################

def ComputeSlice(self,nbproc=1):
  limit = self.Nw_
  OUTPUT = NP.zeros((self.dx_*2,self.dy_*2,limit),'complex')
  if nbproc > 1:
    global PG
    PG   = Progress_bar(limit,'parallel')
    pool = Pool(nbproc)
    OUTPUT = pool.map(ComputeSlice_Freq_Parallel, zip(IT.repeat(self),range(limit)))
    pool.close()
    pool.join()
  else:
    for i in range(limit):
      PG   = Progress_bar(limit,'serial')
      OUTPUT[:,:,i] = ComputeSlice_Freq(self,i)
      PG.update()

  OUTPUT = NP.transpose(NP.array(OUTPUT),[1,2,0])
  OUTPUTnegfreq = OUTPUT[...,1:]
  OUTPUTnegfreq = NP.conj(OUTPUTnegfreq[...,::-1])
  OUTPUT = NP.concatenate([OUTPUT,OUTPUTnegfreq],axis=2)
  del PG
  return OUTPUT

def ComputePowerSpectrumSlice_Freq(self,w_ind,FinalPictureSize = [384,384],GreenComputeOnFlyClass = None):
  if GreenComputeOnFlyClass is not None:
    GG = GreenComputeOnFlyClass.get(ifreq=w_ind,MFFT=self.MJoutput_.typeOfOutput_ in [TypeOfOutput.Surface2D])[0].squeeze()
  else:
    GG = loadGreenFreqInd(self.MJoutput_,w_ind,MFFT = self.MJoutput_.typeOfOutput_ in [TypeOfOutput.Surface2D]).squeeze()
  data = NP.imag(GG - NP.conj(GG)) 
  # if self.MJoutput_.typeOfOutput_ == TypeOfOutput.Surface2D:
  #   GG=GG[0:int(round(len(GG[:,1])/2.)),...]
  #   GG = NP.fft.ifft(GG,axis=1)
  # elif self.MJoutput_.typeOfOutput_ == TypeOfOutput.Surface1D:
  #   GG=GG[0:int(round(len(GG)/2.))]

  pixels,_,_,_ = self.MappedImage(self,data,supress = True)
  pixels = pixels[self.NX_/2-FinalPictureSize[0]/2:self.NX_/2+FinalPictureSize[0]/2,self.NX_/2-FinalPictureSize[1]/2:self.NX_/2+FinalPictureSize[1]/2]
  pixels = fft.fft2(NP.nan_to_num(pixels))
  pixels = fft.fftshift(NP.conj(pixels)*pixels)
  return NP.real(pixels)

def ComputeSlice_Freq_Parallel(args):
  '''to overpass the one argument limit in the case of the use of the parallel function getCurrentGreenFunction.'''

  res =  ComputeSlice_Freq(*args)
  if ('PG' in globals()):
    PG.update()

  return res






