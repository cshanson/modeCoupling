import numpy             as NP
import matplotlib.pyplot as PLOT

from ..Common     import *
from ..Parameters import *
from   ..Observations import *
# from crossCorrelation import *

# ============================================================
# Computes the l-w power spectrum in Parallel and Serial
# ============================================================

def computeLOmegaPowerSpectrum(params,G=None,data=None,XS=None,Ls=list(range(1000)),nProc=1,FULL='NO',normalized=False):
  ''' Computes the projection on Pl legendre polynomials of the cross correlation of function F. 
      Input data Green() should be shaped as (theta,omega).
      Routine is parallelized in omega.
  ''' 
  if XS is None:
    XS = toyModelCrossCorrelation(params)
  if G is None and data is None:
    G = Green(params)

  if data is not None:
    F = XS(data = data)
  else:
    F = XS(G)
  if F.ndim != 2:
    raise Exception("Input array not 2 Dimensions")
    
  # Getting the number of frequencies to compute
  frequencies = params.config_('Frequencies').split()
  
  if frequencies[0] == 'RANGE':
    Nw = params.time_.Nt_ - 1
  elif FULL != 'NO':
    Nw = F.shape[1]
  else:
    Nw = int(ceil(F.shape[1]/2.))

  # Calling projectOnLegendre for each frequency
  # cf Common/assocLegendre.py
  try :
    if params.config_('SaveEllCoeffs').upper() in ['ALL','RANGE'] and params.unidim_:
      Already_Ell = True
    else:
      Already_Ell = False
  except:
    Already_Ell = False

  if Already_Ell:
    POW = NP.squeeze(F[:,:Nw]) * NP.sqrt(4*NP.pi / (2*params.Ellgrid_[:,NP.newaxis] + 1))
  else:
    if nProc > 1:
      POW = reduce(projectOnLegendre,(F[:,:Nw],Ls,normalized,0),Nw,nProc,progressBar=True)
    else:
      POW = projectOnLegendre(F[:,:Nw],Ls,normalized,0,pgBar=True)

    if normalized:
      POW = POW * NP.sqrt(4*NP.pi / (2*NP.array(Ls)[:,NP.newaxis]+1)) # factor still left over if using normalized

  return POW

# ============================================================

def plotLOmegaPowerSpectrum(data,params,Ps=None,ucmap=None,\
                            scale=1,title='',lMax=1000,\
                            fileName=None,**plotArgs):
  ''' Plots a (l,w) power spectrum with parameters from the Framework paper.
      A source power function PS can be specified.
  '''

  # Select color map
  if isinstance(ucmap,str):
    print('Using Standard Python Colormap')
  elif type(ucmap).__module__ == 'numpy':
    print('Using Custom Colormap, numpy array')
    ucmap = colors.ListedColormap(ucmap/255)
  #elif ucmap is None:
    # Build cmap using reds...
    # 
  else:
    print('Using Custom Colormap, Unknown input type')

  Nf   = data.shape[1]
  freq = output.time_.omega_[:Nf]/(2.e0*pi)

  # Multiply by Ps if necessary
  if Ps is None:
    Ps = 1.e0
  else:
    if hasattr(Ps,"__len__"):
      Ps = NP.array([Ps])
      if Ps.shape[0] != Nf:
        raise Exception("Given Ps function does not match given power spectrum.")
  toPlot = data*Ps

  nuPoint = freq[NP.argmin(abs(freq-0.003))]
  scale   = scale/NP.amax(toPlot[nuPoint,:])
  if scale == 0:
    scale = 1.0

  PLOT.figure()
  PLOT.ion()
  PLOT.rc('text',usetex=True)
  PLOT.rc('font',family='serif')
  PLOT.imshow(toPlot*scale,origin='bottom',extent=[0,LMAX,0,freq[-1]*1000],\
              aspect='auto',vmin=0,vmax=NP.amax(toPlot*scale),cmap=ucmap,**plotArgs)
  PLOT.xlabel('Harmonic Degree $l$',fontsize=25)
  PLOT.ylabel('Frequency (mHz)'    ,fontsize=25)
  PLOT.xticks(fontsize=20)
  PLOT.yticks(fontsize=20)
  cbar = PLOT.colorbar()
  cbar.set_label(label='Arbritary Units',size=16)
  PLOT.tight_layout()

  try:
    PLOT.show()
  except:
    pass

  if isinstance(fileName,str):
    PLOT.savefig(fileName)

# ============================================================
# Produces a m-l-omega Power Spectrum
# ============================================================

def MLOmegaPowerSpectrum(params,F,Ls=list(range(1000)),Ms=list(range(-25,25)),FULL="NO",normalized=False,axisL=0,axisM=1,progressBar=True):
  ''' Computes the projection on spherical harmonics of the cross correlation of function F. 
      Last dimension of input data F must be omega
      Routine is parallelized in omega.
  ''' 

  # Getting the number of frequencies to compute
  frequencies = params.config_("Frequencies").split(" ")
  
  if frequencies[0] == "RANGE":
    Nw = params.time_.Nt_ - 1
  elif FULL != "NO":
    Nw = F.shape[1]
  else:
    Nw = int(ceil(F.shape[1]/2.))

  # Calling projectOnAssociatedLegendre for each frequency
  # cf Common/assocLegendre.py

  if params.nbProc_ > 1:
    POW = reduce(projectOnSphericalHarmonics,(F[:,:Nw],Ls,Ms,normalized,axisL,axisM),Nw,params.nbProc_,pgBar=progressBar)
  else:
    POW = projectOnSphericalHarmonics(F[:,:Nw],Ls,Ms,normalized,axisL,axisM,progressBar=pgBar)

  return POW

def MLOmegaPowerSpectrumFromMModes(params,F,Ls=list(range(1000)),Ms=list(range(-25,25)),FULL="NO",normalized=True,axisL=0,axisM=1,progressBar=True):
  ''' Same as previous, but data is already given as MJ output, ie, 
      input is already expanded into m modes.
  ''' 

  # Getting the number of frequencies to compute
  frequencies = params.config_("Frequencies").split(" ")
  
  if frequencies[0] == "RANGE":
    Nw = len(params.time_.omega_)
  elif FULL != "NO":
    Nw = F.shape[1]
  else:
    Nw = int(ceil(F.shape[1]/2.))

  # Calling projectOnAssociatedLegendre for each frequency
  # cf Common/assocLegendre.py

  if params.nbProc_ > 1:
    POW = reduce(projectOnAssociatedLegendre,(F[:,:Nw],Ls,Ms,normalized,axisL,axisM),Nw,params.nbProc_,progressBar=progressBar)
  else:
    POW = projectOnAssociatedLegendre(F[:,:Nw],Ls,Ms,normalized,axisL,axisM,pgBar=progressBar)

  return POW

# ============================================================
# Azimuthally average a 3D power spectrum
# ============================================================

def aziAvg(POW,kxGrid,kyGrid,fullKRrad=False):
  ''' XXX FILL COMMENTS XXX Routine from Jan XXX '''

  # Establish the kx, ky and kRad grids
  POW      = array(POW)
  kxGrid   = array(kxGrid)
  kyGrid   = array(kyGrid)
  kx,ky    = meshgrid(kxGrid,kyGrid)
  kGrid2D  = hypot(kx,ky)
  dK       = kxGrid[1] - kxGrid[0]
  kRad     = kxGrid[kxGrid>=0]
  kRadInd  = where(kxGrid>=(0))[0]
  kRadFull = absolute(kxGrid)
  kRadFirstNonnegInd = kRadInd[0]

  # Initialize P_azi_1D
  thirdCompLength = POW.shape[2]
  if fullKRad:
    PAzi1D     = zeros((len(kRadFull),thirdCompLength))
    fullUseInd = kRadFirstNonnegInd - 1
  else:
    PAzi1D     = zeros((len(kRad),thirdCompLength))
    fullUseInd = 0

  PB = progressBar(len(kRad),'serial')
  for i in range(len(kRad)):
    index   = logicalAnd(kGrid2D < kRad[i] + dK/2, kGrid2D >= kRad[i]-dK/2)
    index3D = index[:,:,newaxis]
    PAzi1D[i+fullUseInd,:] = sum(sum(index3D*POW,axis=0),axis=0)/sum(sum(index==0))
    PB.update()
  del PB

  # The following is for if we need all K (including negative)
  # NOT DEBUGGED
  if fullKRad:
    if sum(kxGrid==0)==1:
      if sum(kxGrid<0) == sum(kxGrid>0):
        PAzi1D[where(kxGrid<0),:] = fliplr(PAzi1D[where(kxGrid>0),:])
      elif sum(kxGrid<0)>sum(kxGrid>0):
        delta       = sum(kxGrid<0) - sum(kxGrid>0)
        ind         = kxGrid<0
        ind[:delta] = 0
        PAzi1D[where(ind==True),:] = flipud(PAzi1D[where(kxGrid>0),:])
      elif sum(kxGrid<0)<sum(kxGrid>0):
        delta           = sum(kxGrid>0) - sum(kxGrid<0)
        ind             = kxGrid>0
        ind[delta-1:-1] = 0
        PAzi1D[where[kxGrid<0],:] = flipud(PAzi1D[where(kxGrid>0),:])
    else:
      kRadNeg    = abs(kxGrid[where(kxGrid<0)])
      kRadNegInd = where(kxGrid<0)
      for i in arange(0,k_rad_neg):
        index   = kGrid2D < kRadNeg[i]+dK/2 and kGrid2D >= kRadNeg[i]-dK/2
        index3D = tile(index,[1,1,thirdCompLength])
        PAzi1D[i+kRadNegInd[i]-1] = sum(sum(POW*index3D))/sum(index==0)
    kRad = kRadFull

  return PAzi1D,kRad

# ============================================================
# Cartesian maps power spectrum
# ============================================================

class cartesianPowerSpectrum:
    ''' Class containing the methods to compute the power spectrum (kx, ky, w)
        or (k,w) from the filtered observations obtained 
        from the class dopplergram
    '''

    def __init__(self,doppler,fileNameP3D=None,fileNameP2D=None,checkPlots=False):
      ''' Doppler is an object of type dopplergram containing 
          all the informations about the data series and filtering. 
          filenamePower3D (resp. filenamePower2D) is the name of the 3D 
          (resp. 2D) power spectrum from reading and/or writing.
      '''

      self.doppler_ = doppler
      if not fileNameP3D is None:
        self.file3D_ = fileNameP3D
      if not fileNameP2D is None:
        self.file2D_ = fileNameP2D
      self.plot_ = checkPlots

    def setFileName3D(self,fileName3D):
      self.file3D_ = fileName3D

    def setFileName2D(self,fileName2D):
      self.file2D_ = fileName2D

    # =========================================================================

    def compute3D(self):
      ''' Computes the power spectrum from filtered observations. 
          Suppose that the filtered observations have already been created 
          using doppler.createFilteredObservations().
      '''

      # Shortcuts for the sake of readability
      d     = self.doppler_
      dDir  = d.directory_
      fName = '%s/%s'%(dDir,self.file3D_)

      # Load P_3D if already computed
      if hasattr(self,'file3D_') and os.path.isfile(fName):
        P_3D = NP.load('%s/%s'%(dDir,self.file3D_))
      else:
        P_3D = NP.zeros((d.params_.geom_.N_[0],d.params_.geom_.N_[1],d.params_.time_.Nt_))
        # Add all files contribution
        for sublist in d.names_:
          for name in sublist:
            dkw   = NP.load('%s/%s'%(dDir,name))
            P_3D += NP.abs(dkw)**2
        P_3D = P_3D/(d.nDays_*d.nDopplersPerDay_)
        if hasattr(self,'file3D_') and not self.file3D_ is None:
          NP.save('%s/%s'%(dDir,self.file3D_),P_3D)

      if self.plot_:
        self.plotPkwCut(d.params_.geom_.kx_,d.params_.time_.omega_,P_3D,0,'Pkw.png')
        self.plotPkkCut(d.params_.geom_.kx_,d.params_.time_.omega_,P_3D,3,'Pkk.png')

      return P_3D

    # =========================================================================

    def compute2D(self):
      '''  Computes the (||k||,omega) power spectrum 
           by integrating over the angle
      '''

      # Shortcuts
      d     = self.doppler_
      dDir  = d.directory_
      fName = '%s/%s'%(dDir,self.file3D_)
      time  = d.params_.time_
      geom  = d.params_.geom_

      # Load the file if already computed

      if hasattr(self,'file2D_') and os.path.isfile(fName):
        P_2D = NP.load(fName)
      else:
        P_3D = self.compute3DPowerSpectrum()
      
        # Perform a FFTshift to have the natural ordering 
        # in k in order to perform the interpolation 
        P_3D = NP.fft.fftshift(P_3D,axes=(0,1))
        kx   = NP.fft.fftshift(geom.k_[0])
        ky   = NP.fft.fftshift(geom.k_[1])
        k    = kx
        interp,Nr,Ntheta = initCart2DToRadial(k,kx,ky)


        # Reduce the interval for the projection if omegaMin or omegaMax 
        # is defined to avoid to project maps of 0
        if hasattr(d,'omegaMin_'):
          iMin = NP.argmin(NP.abs(time.omega_-d.omegaMin_))
        else:
          iMin = 0
        if hasattr(d, 'omegaMax_'):
          iMax = NP.argmin(NP.abs(time.omega_-d.omegaMax_))
        else:
          iMax = time.Nt_

        # Make the projection for all frequencies with non-zero power
        P_2D = NP.zeros((len(k),time.Nt_))
        for w in range(iMin,iMax+1):
          P_2D[:,w] = getCart2DToRadial(P_3D[:,:,w],interp,Nr,Ntheta)

        # Add negative frequencies
        P_2D = time.addSymmetricPart(P_2D) 

        # Put the 0 frequency first
        P_2D = NP.fft.ifftshift(P_2D, axes=0)

        # Save power spectrum 2D if required
        if hasattr(self,'file2D_') and not self.file2D_ is None:
          NP.save(fileName,P_2D)

      if self.plot_:
        self.plotP2D(geom_.k_[0],time.omega_,P_2D,'P2D.png')

      return P_2D

    # =========================================================================

    @staticmethod
    def plotPkwCut(self,k,omega,P_3D,ky,fileName,title=r'$P(k_x,k_y,\omega)$'):
      ''' Plots a (k_x,w) cut of the 3D power spectrum 
          and the spatial frequency k_y given by ky 
      '''
      iky = NP.argmin(NP.abs(k-ky))
      PLOT.rc('text',usetex=True)
      PLOT.rc('font',family='serif')
      PLOT.figure()
      im = PLOT.pcolormesh(NP.fft.ifftshift(k)*RSUN,\
                           NP.fft.ifftshift(omega)/(2*NP.pi)*1e3,\
                           NP.fft.ifftshift(NP.transpose(P_3D[:,iky,:])))

      PLOT.colorbar(im)
      PLOT.xlabel(r'$k_x R$')
      PLOT.ylabel(r'$\omega / 2 \pi$ (mHz)')
      PLOT.title (r'%s for $k_y R$ = %1.4g'%(title,ky*RSUN))
      axes = PLOT.gca()
      axes.set_xlim([-2000, 2000])
      axes.set_ylim([-5, 5])
      PLOT.savefig(fileName,dpi=72)
      PLOT.close()

    @staticmethod
    def plotPkkCut(k,omega,P_3D,f,fileName,title=r'$P(k_x,k_y,\omega$)'):
      ''' Plots a (k_x,k_y) cut of the 3D power spectrum 
          and the frequency given by f in mHz
      '''

      iFreq = NP.argmin(NP.abs(omega/(2*NP.pi)*1.e3-f))
      PLOT.rc('text',usetex=True)
      PLOT.rc('font',family='serif')
      PLOT.figure()
      im = PLOT.pcolormesh(NP.fft.ifftshift(k)*RSUN,\
                           NP.fft.ifftshift(k)*RSUN,\
                           NP.fft.ifftshift(P_3D[:,:,iFreq]))
      PLOT.colorbar(im)
      PLOT.xlabel(r'$k_x$ R')
      PLOT.ylabel(r'$k_y$ R')
      PLOT.title (r'%s for $\omega / 2 \pi$ = %1.4g mHz' % (title, f))
      axes = PLOT.gca()
      axes.set_xlim([-2000,2000])
      axes.set_ylim([-2000,2000])
      PLOT.savefig(fileName,dpi=72)
      PLOT.close()
  
    @staticmethod
    def plotP2D(k,omega,P_2D,fileName):
      ''' Plots the (k,w) 2D power spectrum '''
      PLOT.rc('text',usetex=True)
      PLOT.rc('font',family='serif')
      PLOT.figure()
      im = PLOT.pcolormesh(NP.fft.ifftshift(k)*RSUN,\
                           NP.fft.ifftshift(omega)/(2.e0*NP.pi)*1.e3,\
                           NP.fft.ifftshift(NP.transpose(P_2D)))
      PLOT.colorbar(im)
      PLOT.xlabel(r'$|k| R$')
      PLOT.ylabel(r'$\omega / 2 \pi$ (mHz)')
      PLOT.title (r'$P(|k|,\omega$)')
      axes = PLOT.gca()
      axes.set_xlim([-2000,2000])
      axes.set_ylim([-5,5])
      PLOT.savefig(fileName,dpi=72)
      PLOT.close()  

    # =========================================================================

    def generateRealisations(self):
      ''' Generates noise realisations 
          phi(k,w) = \sqrt{P} N(k, omega).
      '''
      P_3D    = NP.load('%s/%s'%(self.doppler_.directory_,self.file3D_)) 
      randomv = self.createNormalDistribution(P_3D.shape)
      return NP.sqrt(P_3D)*randomv

    @staticmethod
    def createNormalDistribution(sizes):
      nbx = int(NP.round(sizes[0]/2))
      nby = int(NP.round(sizes[1]/2))
      nbo = int(NP.round(sizes[2]/2))

      # Normal distribution : N 
      N = (NP.random.randn(*sizes)+1j*NP.random.randn(*sizes))/NP.sqrt(2.e0)
 
      # 3d symmetries
      revN = N[::-1,::-1,::-1]
      N[nbx+1::,nby+1::,nbo+1::] = NP.conj(revN[nbx:-1,nby:-1,nbo:-1])
      N[1:nbx+1,1:nby+1,nbo+1::] = NP.conj(revN[:nbx  ,:nby  ,nbo:-1])
      N[1:nbx+1,nby+1::,nbo+1::] = NP.conj(revN[:nbx  ,nby:-1,nbo:-1])
      N[nbx+1::,1:nby+1,nbo+1::] = NP.conj(revN[nbx:-1,:nby  ,nbo:-1])
            
      # 2d symmetries - cut at 0 and first point
      for i in [0,nbo]:
        N[nbx+1::,nby+1::,i] = NP.conj(revN[nbx:-1,nby:-1,i])
        N[1:nbx+1,nby+1::,i] = NP.conj(revN[:nbx  ,nby:-1,i])

      for i in [0,nby]:
        N[nbx+1::,i,nbo+1::] = NP.conj(revN[nbx:-1,i,nbo:-1])
        N[1:nbx+1,i,nbo+1::] = NP.conj(revN[:nbx  ,i,nbo:-1])

      for i in [0,nbx]:
        N[i,nby+1::,nbo+1::] = NP.conj(revN[i,nby:-1,nbo:-1])
        N[i,1:nby+1,nbo+1::] = NP.conj(revN[i,:nby  ,nbo:-1])
            
      # 1d symmetries
      for i in [0,nbx]:
        for j in [0,nby]:
          N[i,j,nbo+1::] = NP.conj(revN[i,j,nbo:-1])

      for i in [0, nbx]:
        for j in [0, nbo]:
          N[i,nby+1::,j] = NP.conj(revN[i,nby:-1,j])

      for i in [0, nby]:
        for j in [0, nbo]:
          N[nbx+1::,i,j] = NP.conj(revN[nbx:-1,i,j])

      # 0d symmetries
      for i in [0,nbx]:
        for j in [0,nby]:
          for k in [0,nbo]:
            N[i,j,k] = NP.real(N[i,j,k])

      return NP.fft.fftshift(N)
   
