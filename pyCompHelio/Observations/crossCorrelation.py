import numpy             as NP
import matplotlib.pyplot as PLOT
from   .powerSpect import *
from   .psFunction import *
from   ..Observations import *
import os

# class anotherCrossCorrelation:

class toyModelCrossCorrelation:

    ''' Class to compute C(r1,r2,w) = Ps(w)/(4iw) * (G(r2,r1,w,u)-G*(r2,r1,w,-u)). Can also compute the azimuthally averaged cross correlation C_m(r1,r2,w) anc C_{-m}(r1,r2,w) using 
C_m(r1,r2,w) =  Ps(w)/(4iw) * (G_m(r2,r1,w,u)-G_{-m}*(r2,r1,w,-u))
C_{-m}(r1,r2,w) =  Ps(w)/(4iw) * (G_{-m}(r2,r1,w,u)-G_m*(r2,r1,w,-u))
    '''

    def __init__(self,params,Ps=None,Ps0=0.e0):

      ''' Ps  : psFunction instance, not an array of values. 
          Ps0 : value to replace Ps(w)/w at w=0.e0 by.
      '''

      self.params_  = params
      self.Flow_    = params.config_('Flow',False)
      if Ps is None:
        self.Ps_  = psFunction(params,'one')
      else:
        self.Ps_ = Ps
      self.Ps0_  = Ps0 

    #===============================================================

    def __call__(self,GPuPm=None,GMuPm=None,GPuMm=None,GMuMm=None,\
                     data=None,dataMuPm=None,dataPuMm=None,dataMuMm=None,\
                     freq=None,ifreq=None,omega=None,singleM=None):
      return self.compute(GPuPm,GMuPm,GPuMm,GMuMm,\
                          data,dataMuPm,dataPuMm,dataMuMm,\
                          freq,ifreq,omega,singleM)

    def getOmega(self, freq=None,ifreq=None,omega=None):
      if omega is None:
        if freq is not None:
          Omega = freq*2.e0*NP.pi
        elif ifreq is not None:
          Omega = self.params_.time_.omega_[ifreq]
        else:
          Omega = self.params_.time_.omega_
      else:
          Omega = omega
      return Omega

    def getPs(self,Omega):
      # Load Ps values
      Ps = self.Ps_(omega=Omega)

      if DEBUG:
        PLOT.ion()
        PLOT.figure()
        PLOT.plot(real(Ps))

      if Ps.ndim==0:
        Ps = NP.complex128(Ps)
      elif len(Ps)==1:
        Ps = NP.complex128(Ps[0])
      else:
        Ps = Ps.astype(NP.complex128)

      # Divide by 4i*omega, check 0 frequency
      if type(Ps) == NP.complex128 or type(Ps) == float:
        if Omega != 0.e0:
          Ps /= 4.e0j*Omega
        else:
          Ps = self.Ps0_/4.e0j
      else:
        if Omega is None:
          Ps[1:] /= 4.e0j*self.params_.time_.omega_[1:]
          Ps[0]   = self.Ps0_/4.e0j
        else:
          mask = abs(Omega)<1.e-20
          Ps[  mask]  = self.Ps0_/4.e0j
          Ps[(1-mask).astype(bool)] /= 4.e0j*self.params_.time_.omega_[(1-mask).astype(bool)]

        if DEBUG:
          print(NP.amax(Ps))
          PLOT.plot(imag(Ps))

      return Ps

    def compute(self,GPuPm=None,GMuPm=None,GPuMm=None,GMuMm=None,\
                     data=None,dataMuPm=None,dataPuMm=None,dataMuMm=None,\
                     freq=None,ifreq=None,omega=None,singleM=None):

      ''' Green's functions can be given with data arrays
          or via GP's, which are instances of the class Green.
          If frequencies or pulsations are provided
          (directly or by their indices), computes restriction
          to those frequencies.
          GPuPm: G_m(+u), GMuPm: G_m(-u),GPuMm: G_-m(+u),GMuMm: G_-m(-u).
          If only GPuPm is given, the other Green's functions are built automatically by changing the signs of u and m.
          If there is a flow, returns [XS_m, XS_-m], otherwise returns XS_m
      '''

      # Load Ps values
      Omega = self.getOmega(freq,ifreq,omega)
      Ps = self.getPs(Omega)

      # Load Green's function(s)     
      Gm = None
      if data is not None:
        Gp = data
        if dataMuPm is not None:
          GmPm = dataMuPm
        if dataPuMm is not None and dataMuMm is not None:
          GpMm = dataPuMm
          GmMm = dataMuMm
      else:
        # Compute or load Green's function on the fly
        if freq is None and omega is not None:
          freq = omega/(2.e0*NP.pi)

        if hasattr(self.params_.bgm_, 'flow'):
          # With a flow, we also need the Green's function with -u
          if GMuPm is None:
            # Compute on the fly
            Gp,GmPm,GpMm,GmMm = GPuPm(freq,ifreq,reverseFlow=True,singleM=singleM)[::2]
          else:
            # Load using the user defined Green's function
            Gp = GPuPm(freq,ifreq)[0]
            GmPm = GMuPm(freq,ifreq)[0]
            if GPuMm is not None and GMuMm is not None:
              GmMm = GMuMm(freq,ifreq)[0]
              GpMm = GPuMm(freq,ifreq)[0]
            else: # Guess G_{-m} from G_m
              m      = getModes(GPuPm.params_.config_)[0]
              GtmpMm = GPuPm.getGReverseMFromG(Gp,GmPm,singleM=m)
              GpMm   = GtmpMm[0]
              GmMm   = GtmpMm[1]

        else:
          Gp = GPuPm(freq,ifreq)[0]


      if hasattr(self.params_.bgm_, 'flow'):
        # Returns XS_m and XS_{-m} if there is a flow
        if singleM is None or singleM == 0:
          return (Gp-NP.conj(GmMm))*Ps
        else:
          return (Gp-NP.conj(GmMm))*Ps, (GpMm-NP.conj(GmPm))*Ps 
      else:
        if self.params_.config_('GammaLaplacian',0) == 'YES' and self.params_.config_('TypeEquation',0) == 'HELMHOLTZ_GAMMA_LAPLACE':
          return 2.e0j*NP.imag(Gp)*Ps + 1.j*params.bgm_.kappa.getKappa(1) * Ps * NP.real(Gp)
        else:
          return 2.e0j*NP.imag(Gp)*Ps
# ======================================================================
class crossCovarianceRHS:
    ''' Computes the cross-covariance using the RHS of eq. 47 of gizon et al. 2017
    '''
    def __init__(self,configFile,typeOfObservable = TypeOfObservable.rhoc2DivXi,Ps = None,Ps0 = 0.,\
                 perturbation = None):
      ''' params is the parameters class from which to compute the cross covariance
          Ps is the psfunction of the cross-covariance
      '''
      params = parameters(configFile,TypeOfOutput.Polar2D,nbProc=8)
      if typeOfObservable == TypeOfObservable.rhoc2DivXi:
        raise Exception('only cDivXi implemented at the moment')
      self.configFiles_ = [params.configFile_]
      if perturbation is not None:
        params.config_.set('OutDir',params.config_('OutDir') + '/PERT/')
        params.config_.set(perturbation[0],perturbation[1])
        configNEW = os.getcwd() + '/perturb.init'
        params.config_.save(configNEW)
        self.configFiles_ = self.configFiles_ + [configNEW]

      self.TOObs_    = typeOfObservable

      if Ps is None:
        self.Ps_  = psFunction(params,'one')
      else:
        self.Ps_ = Ps
      self.Ps0_  = Ps0

    def compute(self,ifreq = None,nProc = 300,Memory=8.):
      ''' Compute the crosscovariance from the initFile
          ifreq is the ind on which to compute the crossCov
          nProc = number of processes, Memory in Gb
      '''
      params        = parameters(self.configFiles_[0],TypeOfOutput.Polar2D,nbProc=8)
      omega         = params.time_.omega_
      limit         = params.time_.limit_

      # Perform some checks for the parallelization
      if len(omega) < 5:
        parallelRegroup_nbproc = limit
      else:
        parallelRegroup_nbproc = 5

      if nProc > limit:
        nProc = limit

      # Perform the calculations

      if ifreq is None and nProc > 1:
        DATA       = reduceOnCluster(RHSeq47,(self.configFiles_,NP.arange(limit),self.Ps_,self.TOObs_),\
                                        limit,nProc,Memory*1024*1024,parallelRegroup=parallelRegroup_nbproc,deleteFiles=True)
      elif ifreq is None and nProc == 1:
        DATA       = RHSeq47(self.configFiles_,NP.arange(limit),self.Ps_,self.TOObs_)
      else:
        DATA       = RHSeq47(self.configFiles_,ifreq,self.Ps_,self.TOObs_)

      # Use causality if required

      if params.time_.useCausality() and ifreq is None:
        array             = NP.zeros((len(self.configFiles_),params.time_.Nt_),complex)
        array[:,:limit]   = DATA
        array[:,limit:]   = NP.conj(DATA[:,limit:0:-1])
        DATA              = array

      # Store the data
      self.dataRef_     = DATA[0]
      if len(self.configFiles_) > 1:
        self.dataPert_  = DATA[1]
      if len(self.configFiles_) > 2:
        self.dataOther_ = DATA[2:]


def RHSeq47(initFiles,ifreq,PsFunc,TOObs = TypeOfObservable.rhoc2DivXi):
  DATAS = []
  if isinstance(initFiles,str):
    initFiles = [initFiles]
  for i in range(len(initFiles)):
    initFile = initFiles[i]

    params   = parameters(initFile   , TypeOfOutput.Polar2D, nbProc=10)
    G1 = Green(params,onTheFly=initFile,onTheFlyDel = False, nSrc= 0,observable=TOObs)
    G2 = Green(params,onTheFly=initFile,onTheFlyPrev = G1, nSrc =1,observable=TOObs)     

    G_p1 = G1.get(ifreq = ifreq)
    G_p2 = G2.get(ifreq = ifreq)

    NA    = NP.newaxis
    omega = params.time_.omega_[ifreq]
    gamma = params.bgm_.damping.getDamping(freq = omega/(2*NP.pi),rads = True)
    rho   = params.bgm_.rho(geom=params.geom_)
    c     = params.bgm_.c(geom=params.geom_)
    R     = params.geom_.r()*RSUN
    theta = params.geom_.theta()
    if omega != 0:
      CrossCovCoeff = PsFunc(omega=omega)/(4.j*omega)
    else:
      CrossCovCoeff = 0

    Jac   = R[:,NA]**2*NP.sin(theta[NA,:])

    # Compute integrands of eqn. 47
    if TOObs == TypeOfObservable.rhoc2DivXi:
      raise Exception('Only rhoc2DivXi implemented')
    elif TOObs == TypeOfObservable.cDivXi:
      T1 = simps(NP.conj(G_p1)*G_p2*rho*Jac,x=R,axis=0)
      T2 = c[-1]*NP.conj(G_p1[-1])*G_p2[-1]*rho[-1]*Jac[-1]

    DATA = 2*NP.pi*2.j*omega*(simps(2*gamma*T1 - T2,x=theta,axis=0))
    print(2*NP.pi*2.j*omega*(simps(2*gamma*T1,x=theta,axis=0)))
    print(2*NP.pi*2.j*omega*(simps(T2,x=theta,axis=0)))

    # scale by rhoc(r=1) for the conversion to cDivXi
    if TOObs == TypeOfObservable.cDivXi:
      DATA = DATA/(params.bgm_.getc0()*params.bgm_.getrho0())

    DATAS.append(DATA*CrossCovCoeff)

  # Compute integrals and return
  return NP.array(DATAS)


# ======================================================================
class cartesianCrossCovarianceMaps:

    ''' Class containing the methods to compute the cross covariance 
        C(x, x+delta, w) from the filtered observations 
        obtained from the class dopplergram
    '''

    def __init__(self,doppler,delta=None,averageType=None,checkPlot=False):
      ''' doppler : object of type dopplergram containing all 
                    the informations about the data series and filtering. 
            delta : distance between the points so that C(x,x+delta,w) 
                    is computed for a averaging method given by averageType 
                    (point to point, annulus, ...)
      '''

      self.doppler_ = doppler
      if not delta is None:
        self.delta_ = delta
      if not averageType is None:
        self.averageType_ = averageType
      self.plot_ = checkPlot

    def initFileNames(self,prefix):
      ''' Initializes cross-covariance filenames
      '''
      prefix = '%s_%iMm_%s' % (prefix,round(self.delta_*1e-6),TypeOfTravelTimeAveraging.toString(self.averageType_))
      self.fileNames_ = self.doppler_.generateFileNames(prefix)

    def __call__(self,noise=False,fileNameP3D=None):
      self.compute(noise,fileNameP3D)

    def compute(self,noise=False,fileNameP3D=None):
      ''' Computes C(x-delta/2,x+delta/2,omega)= h_omega phi(x-delta/2,omega)*phi(x+delta/2, omega) 
          from the filtered phi obtained by createFilteredObservations. 
          If noise then create noise cubes from the power spectrum
      '''

      # Shortcuts
      d    = self.doopler_
      geom = d.params_.geom_
      time = d.params_.time_

      deltaShift = int(NP.round(self.delta_/geom.h_[0]))
      for i in range(d.nDays_):
        for j in range(d.nDopplersPerDay_):
          
          # Load phi(k,w)
          if noise:
            power = cartesianPowerSpectrum(d,filenameP3D)
            phikw = power.generateRealisations()
          else:
            phikw = NP.load('%s%s'%(d.directory_,d.names_[i][j]))
          phixw = NP.fft.fftshift(solarFFT.spatialIFFT(phikw,geom),axes=(0,1))
  
          # Point to point East-West
          if (self.averageType_ == TypeOfTravelTimeAveraging.PtP_EW):
            phi_plus  = NP.roll(phixw         , deltaShift,axis=1)
            phi_minus = NP.roll(NP.conj(phixw),-deltaShift,axis=1)
            C0 = time.homega_ *phi_plus *phi_minus

          # Point to point South-North
          elif (self.averageType_ == TypeOfTravelTimeAveraging.PtP_SN):
            phi_plus  = NP.roll(        phixw , deltaShift,axis=0)
            phi_minus = NP.roll(np.conj(phixw),-deltaShift,axis=0)
            C0 = time.homega_ *phi_plus *phi_minus

          # Annulus, East-West arcs, South-North arcs
          elif self.averageType_ in [TypeOfTravelTimeAveraging.AN,\
                                         TypeOfTravelTimeAveraging.EW,\
                                         TypeOfTravelTimeAveraging.SN]:

            # Get the list of points 
            inds,weights = geom.getPointAndWeightingQuadrant(deltaShift*geom.h_[0],self.averageType_)
            C0 = NP.zeros((geom.N_[1],geom.N_[0],time.Nt_),dtype=complex)
  
            # Reduce frequency interval if possible 
            # (0 after a given frequency due to filtering of the dopplergrams)
            if hasattr(d,'omegaMin_') and hasattr(d,'omegaMax_'):
              iwMin    = NP.argmin(NP.abs(d.omegaMin_ - time.omega_))
              iwMax    = NP.argmin(NP.abs(d.omegaMax_ - time.omega_))
              iwMinNeg = NP.argmin(NP.abs(d.omegaMin_ + time.omega_))
              iwMaxNeg = NP.argmin(NP.abs(d.omegaMax_ + time.omega_))
            else:
              iwMin = 0
              iwMax = time.Nt_
  
            phixw = phixw[...,iwMin:iwMax]
            if (params.nbProc_ > 1):    
              C0[...,iwMin:iwMax] = reduce(getC0ByRingPoint,(phixw,inds,weights),\
                                    inds.shape[1],params.nbProc_,'SUM',progressBar=True)
            else:
              PB   = progressBar(inds.shape[1],'serial')
              for l in range(inds.shape[1]):
                C0[...,iwMin:iwMax] += getC0ByRingPoint(phixw,inds[:,l],weights[l])
                PB.update()
              del PB

            C0 *= time.homega_ /inds.shape[1]
            # multiply by the angular step
            #if self.averageType_ == TypeOfTravelTimeAveraging.ANN:
            #  C0 *= 2 * NP.pi / inds.shape[1]
            #else: # quadrant
            #  C0 *= NP.pi / inds.shape[1]

            # Add negative frequencies 
            if iwMin != 0:
              C0[...,iwMaxNeg:iwMinNeg] = NP.conj(C0[...,-iwMaxNeg:-iwMinNeg:-1])
  
          else:
            raise Exception('type of travel time averaging not implemented yet')
  
          NP.save('%s%s' % (d.directory_,self.fileNames_[i][j]),C0)

    @staticmethod
    def getC0ByRingPoint(phixw,ind,weight):
      phi_plusDelta = NP.roll(NP.roll(phixw,ind[0],axis=0),ind[1],axis=1)
      return  weight * phi_plusDelta * NP.conj(phixw)

    def initRefFileName(self,prefix):
      ''' Initialize cross-covariance reference 
          filename and return 1 if the file already exists
      '''

      self.fileNameRef_ = '%s_%iMm.npy' %(prefix,round(self.delta_*1.e-6))
      return  os.path.isfile('%s%s' %(self.doppler_.directory_,self.fileNameRef_))

    def compute3DCrefxw(self,fileNameP3D=None,\
                        fileNameP2D=None,fileNameC3D=None):

      ''' Computes the 3D reference cross covariance from the 
          power spectrum (3D or 2D, depending on the one given).
          Stores it if filenameC3D is given.
      '''

      # Macros
      geom = self.doppler_.params_.geom_
      time = self.doppler_.params_.time_

      # Compute or load 3D power spectrum 
      if fileNameP2D is None:
        if filenamePower3D is None:
          power = cartesianPowerSpectrum(self.doppler_)
          P_3D  = power.compute3D()
        else:
          P_3D = NP.load(self.doppler_.directory_+fileNameP3D)

        C  = solarFFT.spatialIFFT(P_3D,geom)
        C  = NP.fft.fftshift(C,axes=(0,1))
        C *= time.homega_*geom.hk_[0]*geom.hk_[1]

      # Computation from 2D power spectrum
      else:
        C_2D = self.compute2DCrefxw(fileNameP2D=fileNameP2D)

        x     = geom.coords_[0] 
        y     = geom.coords_[1] 
        midX  = int(NP.round(len(x)/2))
        omega = time.omega_
        iwMin = NP.argmin(NP.abs(omega-self.doppler_.omegaMin_))
        iwMax = NP.argmin(NP.abs(omega-self.doppler_.omegaMax_))
  
        if self.plot_:
          PLOT.figure()
          im = PLOT.pcolormesh(x*1e-6,NP.fft.ifftshift(omega)/(2.e0*NP.pi)*1e3,\
                                      NP.fft.ifftshift(NP.transpose(C_2D),axes=0))
          PLOT.colorbar(im)
          PLOT.savefig('C_2D.png',dpi=72)
          PLOT.close()
  
        # Parallelize in omega
        C      = NP.zeros((len(y),len(x),len(omega)))
        nbProc = self.doppler_.params_.nbProc_
        if (nbProc > 1):
          C[...,iwMin:iwMax] = reduce(radialToCart2D,(C_2D[midX:,:],x[midX:],x,y),\
                                      iwMax-iwMin,nbProc,progressBar=True)
        else:
          PB = progressBar(iwMax-iwMin,'serial')
          for w in range(iwMin, iwMax):
            C[:,:,w] = radialToCart2D(C_2D[midX:,w],x[midX,:],x,y)
            PB.update()
          del PB
 
        # Add negative frequencies
        if self.doppler_.omegaMin_ != 0:
          iwMinNeg = NP.argmin(NP.abs(omega +self.doppler_.omegaMin_))
          iwMaxNeg = NP.argmin(NP.abs(omega +self.doppler_.omegaMax_))
          C[...,iwMaxNeg:iwMinNeg] = NP.conj(C[...,-iwMaxNeg:-iwMinNeg:-1])
  
        if self.plot_:
          iy = NP.argmin(NP.abs(x))
          PLOT.figure()
          im = PLOT.pcolormesh(x*1.e-6,NP.fft.ifftshift(omega)/(2.e0*NP.pi)*1.e3,\
                                       NP.fft.ifftshift(NP.transpose(C[:,iy,:]),axes=0))
          PLOT.colorbar(im)
          PLOT.savefig('C_3D.png',dpi=72)
          PLOT.close()
  
          iw = NP.argmin(NP.abs(omega/(2.e0*NP.pi)*1.e3-3.e0))
          PLOT.figure()
          PLOT.plot(x[midX:]*1.e-6,C[midX:,iy,iw],label='Cartesian XS')
          PLOT.plot(x[midX:]*1.e-6,C_2D[midX:,indw],'r',label='Radial XS')
          PLOT.legend()
          PLOT.savefig('C_cut.png', dpi=72)
          PLOT.close()

      if not filenameC3D is None:
        filenameC = '%s/%s' % (self.doppler_.directory_,fileNameC3D)
        NP.save(filenameC, C)
      return C

    def compute2DCrefxw(self,fileNameC2D=None,fileNameC3D=None,\
                             fileNameP2D=None,fileNameP3D=None):
      ''' Computes the (||x||,omega) reference cross covariance 
          by integrating over the angle
      '''

      # Macros
      ddir = self.doppler_.directory_
      geom = self.doppler_.params_.geom_
      time = self.doppler_.params_.time_

      # Load the result if it already exists...
      fileName = '%s/%s'% (ddir,fileNameC2D)
      if os.path.isfile(fileName):
        C_2D = NP.load(fileName)

      else: 
        # ... or compute from the 2D power spectrum ... 
        fileName = '%s/%s'%(ddir,fileNameP2D)
        if os.path.isfile(fileName):
          P_2D  = NP.load(fileName)
          C_2D  = solarFFT.polarIFFT(P_2D,geom)
          C_2D *= time.homega_ *geom_.hk_[0] *geom_.hk_[1]

        # ... otherwise from the 3D cross covariance
        else:
          fileName = '%s/%s'%(ddir,fileNameC3D)
          if os.path.isfile(fileName):
            C_3D = NP.load(fileName)
          else:
            C_3D = self.compute3DCrefxw(fileNameP3D,fileNameP2D,fileNameC3D)

          # Grid in x, y
          x = geom.coords_[0]
          y = geom.coords_[1]
          r = x
          interp,Nr,Ntheta = initCart2DToRadial(r,x,y)

          # Reduce the interval for the projection 
          # if omegaMin or omegaMax is defined 
          # to avoid to project maps of 0
          if hasattr(self.doppler_, 'omegaMin_'):
            iwMin = NP.argmin(np.abs(time.omega_ -self.doppler_.omegaMin_))
          else:
            iwMin = 0
          if hasattr(self.doppler_, 'omegaMax_'):
            iwMax = NP.argmin(np.abs(time.omega_ - self.doppler_.omegaMax_))
          else:
            iwMax = time.Nt_

          # Make the projection for all frequencies with non-zero power
          C_2D = NP.zeros((len(r),time_.Nt_),dtype=complex)
          for w in range(iwMin,iwMax+1):
            C_2D[:,w] = getCartToRadial2D(C_3D[:,:,w],interp,Nr,Ntheta)

          # Add negative frequencies
          C_2D = time.addSymmetricPart(C_2D) 

        if filenameC2D is not None:
          NP.save('%s/%s' % (self.doppler_.directory_,fileNameC2D),C_2D)

      if self.doppler_.params_.initFile_.plot_:
        self.plotC2DTime (C_2D,'C2Dxt.png')
        self.plotC2DOmega(C_2D,'C2Dxw.png')
        self.plotC2DTimeFixedDistance(C_2D,10,'C_10Mm.png')

      return C_2D

    def compute1DCrefxw(self,fileNameC2D=None,fileNameC3D=None,\
                             fileNameP2D=None,fileNameP3D=None,prefix='Crefxw'):

      ''' Computes Cref(Delta, omega) from the power spectrum 
          for two points separated by delta. 
          Suppose that it depends only on ||Delta|| and omega. 
          Only one of the filenames is required to be able to make the computation.
      '''

      # Initialize cross covariance filenames
      if not hasattr(self,'fileNameRef_'):
        self.initFilenames(prefix)
      fullname = self.doppler_.directory_ + self.fileNameRef_

      C = []
      # Load if already computed
      if os.path.isfile(fullname):
        C = NP.load(fullname)
      else:
        C_2D   = self.compute2DCrefxw(fileNameC2D,fileNameC3D,\
                                        fileNameP2D,fileNameP3D)
        iDelta = NP.argmin(NP.abs(self.doppler_.params_.geom_.coords_[0]-self.delta_))
        C      = C_2D[iDelta,:]
        NP.save(fullname,C)

      return C

    def plotC2DTime(self,C_2D,fileName):
      ''' Plots a cross covariance 
          as a function of time and distance
      '''

      # Fourier transform in time
      C_2Dt = solarFFT.temporalIFFT(C_2D,self.doppler_.params_.time_)
      C_2Dt = solarFFT.testRealFFT(C_2Dt)

      PLOT.figure()
      PLOT.rc('text',usetex=True)
      PLOT.rc('font',family='serif')
      im = PLOT.pcolormesh(self.doppler_.params_.geom_.coords_[0]*1.e-6,\
                           NP.fft.ifftshift(self.doppler_.params_.time_.t_)/60.e0,\
                           NP.fft.ifftshift(NP.transpose(C_2Dt),axes=0))
      PLOT.colorbar(im)
      PLOT.xlabel(r'$x$ (Mm)')
      PLOT.ylabel(r'$t$ (min)')
      PLOT.title (r'Cross covariance $C(x,t)$')
      PLOT.xlim([-20,20])
      PLOT.ylim([-50,50])
      PLOT.savefig(fileName,dpi=72)
      PLOT.close()

    def plotC2DOmega(self,C_2D,fileName):
      ''' Plots the reference cross covariance 
          as a function of frequency and distance
      '''

      C_2D = NP.real(C_2D)
      PLOT.figure()
      PLOT.rc('text',usetex=True)
      PLOT.rc('font',family='serif')
      im = PLOT.pcolormesh(self.doppler_.params_.geom_.coords_[0]*1.e-6,\
                           NP.fft.ifftshift(self.doppler_.params_.time_.omega_)/(2.e0*NP.pi)*1.e3,\
                           NP.fft.ifftshift(NP.transpose(C_2D),axes=0))
      PLOT.colorbar(im)
      PLOT.xlabel(r'$x$ (Mm)')
      PLOT.ylabel(r'$\omega / (2 \pi)$ (mHz)')
      PLOT.title (r'Cross covariance $C(x,\omega)$')
      PLOT.xlim([-20,20])
      PLOT.ylim([-5,5])
      PLOT.savefig(fileName,dpi=72)
      PLOT.close()

    def plotC2DTimeFixedDistance(self,C_2D,delta,fileName):
      ''' Plots a cross covariance at a given distance 
          delta in Mm as a function of time
      '''

      # Get time signal
      indx  = NP.argmin(NP.abs(self.doppler_.params_.geom_.coords_[0]*1.e-6-delta))
      C_2Dt = solarFFT.temporalIFFT(C_2D,self.doppler_.params_.time_)
      C_2Dt = solarFFT.testRealFFT(C_2Dt)

      PLOT.figure()
      PLOT.rc('text',usetex=True)
      PLOT.rc('font',family='serif')
      im = PLOT.plot(NP.fft.ifftshift(self.doppler_.params_.time_.t_)/60,NP.fft.ifftshift(C_2Dt[indx,:]))
      PLOT.xlabel(r'$t$ (min)')
      PLOT.title (r'Cross covariance $C(\Delta,t)$ for $\Delta =$ %1.4g Mm' % delta)
      PLOT.xlim([-100,100])
      PLOT.savefig(fileName,dpi=72)
      PLOT.close()

