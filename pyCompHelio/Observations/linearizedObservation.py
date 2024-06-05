import numpy             as NP
import matplotlib.pyplot as PLOT 
import copy
from ..Common        import *
from ..Parameters    import *
from .               import *
from scipy.integrate import simps as SIMPS

# Classes that allows for computation of travel time or amplitude 
# (implicit 'differences') by computing
#
# dtau (or da) = \int W(omega)(C-Cref) domega
# 
# where the definition of the weighting function differs if we
# consider travel times or amplitudes. 

# Elements in common are regrouped under the
# linearizedObservation class

class linearizedObservation:

  def __init__(self,params,XS,windowFunction,travelTimeType,checkPlot=False):
    self.params_         = params
    self.XS_             = XS
    self.window_         = windowFunction
    self.travelTimeType_ = travelTimeType
    self.plot_           = checkPlot

  #============================================================================

  def computeGB04(self,Gxw,Grefxw,GxwNf = None,newTimeStep=None,\
                  typeOfMeas='tau',iterNb=1):
    '''Compute travel time or amplitude difference between a reference cross-covariance Cref and C
    according to the formula in Gizon, Birch 2004 for the travel time (default)
    http://adsabs.harvard.edu/abs/2004ApJ...614..472G
    or for the amplitude (Nagashima et al. 2017).
    newTimeStep is used to increase the resolution of Cref and C in the time
    domain by doing zero padding. If not specified no padding is done.'''

    Cxw    = self.XS_(data=Gxw   ,dataMuPm=GxwNf)
    Crefxw = self.XS_(data=Grefxw,dataMuPm=GxwNf)

    # pad the cross covariances if necessary
    Crefxt,Cxt,newTime = self.getTimeCrossCovariance(Cxw,Crefxw,newTimeStep)
    if newTimeStep is None:
      newTimeStep = self.params_.time_.ht_

    CrefxtShift = Crefxt
    tau = 0
    amp = 0

    for i in range(iterNb):
      # First compute the travel time and shift the cross covariance
      tauCrt = self.computeIntegral(CrefxtShift,Cxt,newTime,'tau')

      if tauCrt.ndim > 0: # maps of travel times
        tau = tauCrt
        break

      tauShift = int(NP.round(tauCrt/newTimeStep))
      if tauShift != 0:
        tau += tauCrt
        CrefxtShift = NP.roll(CrefxtShift,tauShift)
      else:
        if typeOfMeas == 'tau':
          tau = tauCrt
          break

      if typeOfMeas == 'amplitude':
        # Then compute the amplitude perturbation if required
        ampCrt  = self.computeIntegral(CrefxtShift,Cxt,newTime,'amplitude')
        amp    += ampCrt
        CrefxtShift *= (1+ampCrt)
        if NP.abs(ampCrt/amp) < 1e-5:
          break

    if self.plot_ and tauCrt.ndim == 1:
      fig,(ax,) = PLOT.subplots(1,1)
      ax.plot(newTime.t_/60,Crefxt     ,'r',label='Crefxt'        )
      ax.plot(newTime.t_/60,Cxt        ,'b',label='Cxt'           )
      ax.plot(newTime.t_/60,CrefxtShift,'k',label='Crefxt shifted')
      ax.set_xlim([35,85])
      ax.legend(loc='upper left')
      PLOT.savefig('amplitude.png')
      PLOT.close()

    if typeOfMeas == 'tau':
      return tau
    else: # return amplitude and travel time as they were both computed
      return amp,tau

  #============================================================================

  def computeIntegral(self,Crefxt,Cxt,newTime,typeOfMeas):
    ''' Returns the observation (travel time or amplitude) 
        computed such that the difference between C and Cref 
        is as small as possible.
    '''

    # Create window function
    newParams       = copy.copy(self.params_)
    newParams.time_ = newTime
    window          = self.window_.createWithNewTimeParameters(newParams)

    # Compute weight function
    Crefxw = solarFFT.temporalFFT(Crefxt, newTime)
    weight = weightFunction(window,typeOfMeas)
    WxtCrt = weight.Wt([TypeOfTravelTime.PLUS,TypeOfTravelTime.MINUS],Cw=Crefxw)

    # Compute the observation (travel time or amplitude)
    if Cxt.ndim == 3: # if we have a map of XS
      WxtCrt[0] = WxtCrt[0][NP.newaxis,NP.newaxis,:]
      WxtCrt[1] = WxtCrt[1][NP.newaxis,NP.newaxis,:]
      Crefxt    = Crefxt   [NP.newaxis,NP.newaxis,:]

    valPlus  = newTime.ht_*SIMPS(WxtCrt[0]*(Cxt-Crefxt))# / (2*NP.pi) # due to the transform of Cxw (gizon & birch 2004 Eq A2)
    valMinus = newTime.ht_*SIMPS(WxtCrt[1]*(Cxt-Crefxt))#/ (2*NP.pi) # 2PI is already taken care for the W in Wt call

    # return the appropriate quantity
    if   (self.travelTimeType_ == TypeOfTravelTime.PLUS):
      return valPlus
    elif (self.travelTimeType_ == TypeOfTravelTime.MINUS):
      return valMinus
    elif (self.travelTimeType_ == TypeOfTravelTime.MEAN):
      return (valPlus+valMinus)/2.e0
    elif (self.travelTimeType_ == TypeOfTravelTime.DIFF):
      return valPlus-valMinus
    else:
      raise Exception('unknown type of travel time')

  #============================================================================

  def getTimeCrossCovariance(self,Cxw,Crefxw,newTimeStep=None):
    ''' Returns Cref and C padded according to the new time step. 
        Also returns the new time parameters.
    '''

    if hasattr(self.params_,'time_'):
      time = self.params_.time_
    else:
      time = self.params_.createTimeParameters()

    if newTimeStep is None:
      newTime = self.params_.time_
      padNb   = None
    else:
      newTime,padNb = self.params_.time_.changeTimeStep(newTimeStep)
      padNb         = newTime.Nt_

    # Compute C and Cref in the time domain
    Cxt    = solarFFT.temporalIFFT(Cxw   ,newTime,padNb)
    Crefxt = solarFFT.temporalIFFT(Crefxw,newTime,padNb)
    Cxt    = solarFFT.testRealFFT (Cxt   )#/ (2*NP.pi) # due to the transform of Cxw (gizon & birch 2004 Eq A2)
    Crefxt = solarFFT.testRealFFT (Crefxt)# / (2*NP.pi) # 2PI is already taken care for the W in Wt call

    return Crefxt,Cxt,newTime

#=======================================================================

class amplitude(linearizedObservation):
    ''' Computes the amplitude difference between a reference cross covariance
        and a measured one using the linearization proposed 
        in (Nagashima et al. 2015). 
        Minimize \int f(t) (C(t) - a Cref(t-\tau)) dt 
        where f is a window function, \tau the travel time 
        and a the amplitude
     ''' 

    def __init__(self,params,XS,window,travelTimeType,checkPlot=False):
      linearizedObservation.__init__(self,params,XS,window,travelTimeType,checkPlot)

    def compute(self,Gxw,Grefxw,GxwNf,newTimeStep=None,iterNb=1):
      ''' Returns \delta a such that a = 1 + \delta a
      '''
      return self.computeGB04(Gxw,Grefxw,GxwNf,newTimeStep,'amplitude',iterNb)

#=======================================================================

class travelTimeCompute(linearizedObservation):

  def __init__(self,params,XS,window,travelTimeType,checkPlot=False):
    linearizedObservation.__init__(self,params,XS,window,travelTimeType,checkPlot)

  def __call__(self,G,Gref,GNf = None,newTimeStep=None,method='GB04',iterNb=1):
    if method == 'GB04':
      return self.computeGB04(G,Gref,GNf,newTimeStep,'tau',iterNb)
    else:
      return self.computeGB02(G,Gref,GNf,newTimeStep,iterNb)


  def computeGB02(self,Gxw,Grefxw,GxwNf=None,newTimeStep=None,iterNb=1):
    ''' Computes travel time between a reference cross-covariance Cref and C
        according to the formula in Gizon, Birch 2002 (if method == 'GB02')
        http://adsabs.harvard.edu/abs/2002ApJ...571..966G

        newTimeStep is used to increase the resolution of Cref and C 
        in the time domain by doing zero padding. 
        If not specified no padding is done.
    '''

    Cxw    = self.XS_(data=Gxw   ,dataMuPm=GxwNf)
    Crefxw = self.XS_(data=Grefxw,dataMuPm=GxwNf)

    # Pad the cross covariances if necessary
    Crefxt,Cxt,newTime = self.getTimeCrossCovariance(Cxw,Crefxw,newTimeStep)
    newParams           = copy.copy(self.params_)
    newParams.time_     = newTime


    # Create window function
    window = self.window_.createWithNewTimeParameters(newParams)
    window() # create windows
    if newTimeStep is None:
      newTimeStep = self.params_.time_.ht_

    # To fasten the computation we suppose that the perturbation 
    # does not exceed 10 minutes
    maxPerturb = 600
    nbPts      = int(maxPerturb/newTimeStep)
    searchInterval     = arange(-nbPts,nbPts+1)
    searchIntervalTime = searchInterval*newTimeStep

    # Xplus and Xminus correspond to Eq. 4 in [GB02]. 
    # It is the quantity we want to minimize
    Xplus  = NP.zeros((2*nbPts+1))
    Xminus = NP.zeros((2*nbPts+1))

    for i in range(len(searchInterval)):
      Cshift    = NP.roll(Crefxt,-searchInterval[i])
      Xplus[i]  = newTime.ht_*SIMPS(window.FPlus_ *(Cxt-Cshift)**2)
      Cshift    = NP.roll(Crefxt, searchInterval[i])
      Xminus[i] = newTime.ht_*SIMPS(window.FMinus_*(Cxt-Cshift)**2)

    # number of interpolation points used to build the parabola around tmin 
    # total number of points is 2*nIP+1
    nIP = 2
 
    #==========
    # tauPlus

    # P = c0 + c1*t + c2*t^2 around t=t_{min}
    # tauPlus = c1/(2c2)
    iMin = NP.argmin(Xplus) 
    P    = NP.polyfit(searchIntervalTime[iMin-nIP:iMin+nIP+1],\
                                   Xplus[iMin-nIP:iMin+nIP+1],2)
    tauPlus = P[1]/(2.e0*P[0])

    #===========
    # tauMinus

    # P = c0 + c1*t + c2*t^2 around t=t_{min}
    # tauMinus = c1/(2c2)
    iMin = NP.argmin(Xminus)
    P    = NP.polyfit(searchIntervalTime[iMin-nIP:iMin+nIP+1],\
                                  Xminus[iMin-nIP:iMin+nIP+1], 2)
    tauMinus = P[1]/(2.e0*P[0]);

    if   (self.travelTimeType_ == TypeOfTravelTime.PLUS):
      return tauPlus
    elif (self.travelTimeType_ == TypeOfTravelTime.MINUS):
      return tauMinus
    elif (self.travelTimeType_ == TypeOfTravelTime.MEAN):
      return (tauPlus+tauMinus)/2.e0
    elif (self.travelTimeType_ == TypeOfTravelTime.DIFF):
      return tauPlus-tauMinus
    else:
      raise Exception('unknown type of travel time')



#=======================================================================

class cartesianTravelTimeMaps:
    ''' Class containing the methods to compute the travel times 
        tau(x,x+delta) from the cross covariance maps
    '''

    def __init__(self,XSmap,windowFunctionType,travelTimeType,checkPlot=False):
      ''' Take all the required information from 
          the cross covariance maps 
      '''

      self.XSmap_      = XSmap
      self.windowType_ = windowFunctionType
      self.type_       = travelTimeType
      self.plot_       = checkPlot

    def initFilenames(self,prefix='tau'):
      ''' Initializes output names '''
      prefix = '%s_%iMm_%s' % (prefix,round(self.XSmap_.delta_*1e-6),\
               TypeOfTravelTimeAveraging.toString(self.XSmap_.averageType_))
      self.fileNames_ = self.XSmap_.doppler_.generateFileNames(prefix)

    def __call__(self):
      self.compute()

    def compute(self):
      ''' Computes all travel time maps for the total observation time 
          and all type of averaging and save the result in the filenames 
          defined during initFilenames.
      '''
      d    = self.XSmap_.doppler_
      geom = d.params_.geom_

      # Load reference cross covariance
      Crefxw = NP.load('%s/%s' % (d.directory_,self.XSmap_.fileNameRef_))

      # Loop through all computed cross covariances maps
      for i in range(d.nDays_):
        for j in range(d.nDopplersPerDay_):
          Cxw = NP.load('%s/%s'%(d.directory_,self.XSmap_.fileNames_[i][j]))
          tau = travelTime(d.params_,self.windowType_,self.type_,Crefxw,Cxw)
          NP.save('%s/%s'%(d.directory_,self.fileNames_[i][j]),tau())

    def computeNoise(self,self2,filenameP2D=None,filenameC3D=None):
      ''' ========
          COMMENTS 
          ========
      '''

      d = self.XSmap_.doppler_
      Crefxw1 = NP.load('%s/%s'%(d.directory_,self .XSmap_.fileNameRef_))
      Crefxw2 = NP.load('%s/%s'%(d.directory_,self2.XSmap_.fileNameRef_))

      #if  self .XSmap_.averageType_ == TypeOfTravelTimeAveraging.ANN\
      #and self2.XSmap_.averageType_ == TypeOfTravelTimeAveraging.ANN:
      if 0:
        # For two annulii the noise covariance matrix does not depend 
        # on the angle so we can compute the covariance in 1D 
        # and then expand it to Cartesian 2D
        cov1D = self.computeNoiseAnnulus(Crefxw1,self2,Crefxw2,filenameP2D)
        midX  = int(NP.round(geom.N_[0]/2))
        return RadialToCart2D(cov1D[midX:],geom.x()[midX:],geom.x())

      else:
        fileName = '%s/%s' % (d.directory_,filenameC3D)
        if os.path.isfile(fileName):
          Crefxw3D = NP.load(fileName)
        else:
          Crefxw3D = self.XSmap_.compute3DCrefxw(fileNameP2D=fileNameP2D,\
                                                 fileNameC3D=fileNameC3D)

        return self.computeNoiseQuadrant(self2,Crefxw3D,Crefxw1,Crefxw2)

    def computeNoiseAnnulus(self,Crefxw1,self2,Crefxw2,filenamePower2D):
      ''' Computes the noise between two annulii as a function 
          of the distance d between both annulus. 
          The radii of the annulii are given by self.XSmap_.delta_ 
          and self2.XSmap_.delta_.
      '''

      # Macros
      d      = self.XSmap_.doppler_ 
      params = d.params_
      geom   = params.geom_
      time   = params.time_
      delta1 = self .XSmap_.delta_
      delta2 = self2.XSmap_.delta_

      # Load the 2D power spectrum
      power  = cartesianPowerSpectrum(d,fileNameP2D=fileNameP2D)
      P_2D   = power.compute2D()

      # Compute the window functions associated to both measurements
      window1 = windowFunction(self.windowType_,params)
      weight1 = weightFunction(window1)   
      Wxw1    = weight1(Crefxw1,self.type_)

      window2 = windowFunction(self2.windowType_,params)
      weight2 = weightFunction(window2)   
      Wxw2    = weight2(Crefxw2,self2.type_)

      # compute the terms given by Eq. (D7, D8) in (Gizon & Birch 2004)
      k      = params.geom_.k_[0]
      coef   = geom.hk_[0]**2 *time.homega_
      term1  = SP.jv(0,k*delta1) * SP.jv(0,k*delta2)
      term1  = term1[:,NP.newaxis] * P_2D
      term1  = solarFFT.polarIFFT(term1,geom) *coef
      term1 *= solarFFT.polarIFFT(P_2D ,geom) *coef

      term2delta = SP.jv(0,k*delta1)
      term2delta = term2delta[:,NP.newaxis] * P_2D
      term2delta = solarFFT.polarIFFT(term2delta,geom) *coef

      term2deltap = SP.jv(0,k*delta2)
      term2deltap = term2deltap[:,NP.newaxis] * P_2D
      term2deltap = solarFFT.polarIFFT(term2deltap,geom) *coef

      # Eq. D6 in (Gizon & Birch 2004)
      integrand = NP.conj(Wxw1[NP.newaxis,:]) * (Wxw2[NP.newaxis,:]*term1\
                  + NP.conj(Wxw2[NP.newaxis,:]) *term2delta*term2deltap)
      cov  = SIMPS(integrand,axis=1) *params.time_.homega_
      cov *= (2*NP.pi)**3/params.time_.T_
      return cov

    def computeNoiseMonteCarlo(self, self2):
      ''' ========
          Comments
          ========
      '''

      # Macros
      d      = self.XSmap_.doppler_ 
      params = d.params_
      geom   = params.geom_
      time   = params.time_

      cov = NP.zeros(geom_.N_)
      for i in range(d.nDays_):
        for j in range(d.nDopplersPerDay_):   
          tau1    = NP.load('%s/%s' %(directory_,self .fileNames_[i][j]))
          tau2    = NP.load('%s/%s' %(directory_,self2.fileNames_[i][j]))
          tau1k   = NP.conj(solarFFT.spatialFFT(NP.fft.fftshift(tau1),geom))
          tau2k   =         solarFFT.spatialFFT(NP.fft.fftshift(tau2),geom)
          covk    = tau1k*tau2k
          covCrt  = NP.fft.ifftshift(solarFFT.spatialIFFT(covk,geom))
          covCrt /= (geom.N_[0]*geom.N_[1])
          covCrt *= (2.e0*NP.pi)**2 / (geom.h_[0] *geom.h_[1])
          covCrt  = solarFFT.testRealFFT(covCrt)
          cov    += covCrt
      cov /= (d.nDays_*d.nDopplersPerDay_)
      return cov

    def computeNoiseQuadrant(self,self2,Crefxw3D,Crefxw1,Crefxw2):
      ''' ========
          Comments
          ========
      '''

      # Macros
      d      = self.XSmap_.doppler_ 
      params = d.params_
      geom   = params.geom_
      time   = params.time_
      delta1 = self .XSmap_.delta_
      delta2 = self2.XSmap_.delta_
      at1    = self .XSmap_.averageType
      at2    = self2.XSmap_.averageType

      # Point indices belonging to the annulus or quadrants
      inds1,w1 = geom.getPointAndWeightingQuadrant\
                      (delta1,self .XSmap_.averageType_)
      inds2,w2 = geom.getPointAndWeightingQuadrant\
                      (delta2,self2.XSmap_.averageType_)

      # Compute the window functions associated to both measurements
      window1 = windowFunction(self.windowType_,params)
      weight1 = weightFunction(window1)   
      Wxw1    = weight1(Crefxw1,self.type_)

      window2 = windowFunction(self2.windowType_,params)
      weight2 = weightFunction(window2)   
      Wxw2    = weight2(Crefxw2,self2.type_) 

      # Reduce frequency interval if possible 
      # (0 after a given frequency due to filtering of the dopplergrams)
      if hasattr(d,'omegaMin_'):
        iwMin = NP.argmin(NP.abs(d.omegaMin_-time.omega_))
      else:
        iwMin = 0
      if hasattr(d,'omegaMax_'):
        iwMax = NP.argmin(NP.abs(d.omegaMax_-time.omega_))
      else:
        iwMax = time.Nt_
      Crefxw3D = Crefxw3D[...,iwMin:iwMax]

      # Parallelize in y direction
      Nx = geom.N_[0]
      Ny = geom.N_[1]
      if (params.nbProc_ > 1):
        term1 = reduce(getTerm1NoiseQuadrant,\
                      (NP.arange(Ny),Nx,inds1,inds2,w1,w2,Crefxw3D),\
                       Ny,params.nbProc,progressBar=True)
      else:
        term1 = NP.zeros((Ny,Nx,iwMax-iwMin),dtype=complex)
        PB    = Progress_bar(Ny,'serial')
        for dy in range(Ny):
          term1[dy,...] = getTerm1NoiseQuadrant(dy,Nx,inds1,inds2,w1,w2,Crefxw3D)
          PB.update()
        del PB

      term1 /=  (inds1.shape[1]*inds2.shape[1])
      term1 *= Crefxw3D *NP.conj(Wxw1[NP.newaxis,NP.newaxis,iwMin:iwMax])\
                                *Wxw2[NP.newaxis,NP.newaxis,iwMin:iwMax]

      term2_delta1 = NP.zeros(term1.shape,dtype='complex')
      term2_delta2 = NP.zeros(term1.shape,dtype='complex')
      for dx in range(Ny):
        for dy in range(Nx):
          for delta1 in range(inds1.shape[1]):
            ix = (inds1[1][delta1]-dx) % Nx
            iy = (inds1[0][delta1]-dy) % Ny
            term2_delta1[dy,dx,:] += w1[delta1]*Crefxw3D[iy,ix,:]

          for delta2 in range(inds2.shape[1]):
            ix = (inds2[1][delta2]+dx) % Nx
            iy = (inds2[0][delta2]+dy) % Ny
            term2_delta2[dy,dx,:] += w2[delta2]*Crefxw3D[iy,ix,:]

      term2_delta1 /= inds1.shape[1]
      term2_delta2 /= inds2.shape[1]

      term2  = term2_delta1 * term2_delta2 
      term2 *= NP.conj(WxwCrt1[NP.newaxis,NP.newaxis,iwMin:iwMax]) 
      term2 *= NP.conj(WxwCrt2[NP.newaxis,NP.newaxis,iwMin:iwMax])

      # Multiply by the geometry sampling depending on the averaging
      if at1 == TypeOfTravelTimeAveraging.ANN:
        term1 *= 2.e0*NP.pi
        term2 *= 2.e0*NP.pi
      else: # quadrant (quarter of a circle)
        term1 *= NP.pi 
        term2 *= NP.pi 
      if at2 == TypeOfTravelTimeAveraging.ANN:
        term1 *= 2.e0*NP.pi
        term2 *= 2.e0*NP.pi
      else:
        term1 *= NP.pi 
        term2 *= NP.pi 

      if self.plot_:
        PLOT.figure()
        PLOT.rc('text',usetex=True)
        PLOT.rc('font',family='serif')
        im = PLOT.pcolormesh(geom.x()*1.e-6,geom.x()*1.e-6,NP.sum(NP.real(term1),axis=2))
        PLOT.xlim([-20, 20])
        PLOT.ylim([-20, 20])
        PLOT.colorbar(im)
        PLOT.title(r'Term1')
        PLOT.savefig('term1.png')
        PLOT.close()

        PLOT.figure()
        im = PLOT.pcolormesh(geom.x()*1.e-6,geom.x()*1.e-6,NP.sum(NP.real(term2),axis=2))
        PLOT.xlim([-20,20])
        PLOT.ylim([-20,20])
        PLOT.colorbar(im)
        PLOT.title(r'Term2')
        PLOT.savefig('term2.png')
        PLOT.close()

      # Add negative frequencies and integrate over frequency
      cov  = 2.e0*SIMPS(NP.real(term1+term2),axis=2)*time.homega_
      cov *= (2.e0*NP.pi)**3/time.T_

      # Ensure symmetries depending on the observations
      midX   = int(NP.round(Nx/2))
      covSym = NP.zeros(cov.shape)

      covs     = []
      covinvxy = cov[::-1,::-1] 
      covinvxy = covinvxy[midX-1:-1,midX-1:-1]
      covs.append(covinvxy)

      covinvy = cov[::-1,:]
      covinvy = covinvy[midX-1:-1,midX:]
      covs.append(covinvy)

      covinvx = cov[:,::-1]
      covinvx = covinvx[midX:,midX-1:-1]
      covs.append(covinvx)

      covs.append(cov[midX:,midX:])
      sign = [[], [], [1, 1, 1, 1], [-1, 1, -1, 1], [-1, -1, 1, 1]] # sign by type of geometry {ANN, EW, SN}, then by quadrant {SE, SW, NE, NW}

      res = NP.zeros((midX, midX))
      for i in range(4):
        res +=  (sign[at1][i]*sign[at2][i]*covs[i])/4.e0

      covSym[1:midX+1,1:midX+1] = sign[at1][0] * sign[at2][0] * res[::-1,::-1]
      covSym[1:midX+1,midX:   ] = sign[at1][1] * sign[at2][1] * res[::-1,:]
      covSym[midX:   ,1:midX+1] = sign[at1][2] * sign[at2][2] * res[:,::-1]
      covSym[midX:   ,midX:   ] = sign[at1][3] * sign[at2][3] * res

      cov = covSym

      if self.plot_:
        PLOT.figure()
        im = PLOT.plot(geom.x()*1.e-6,cov[:,midX])
        PLOT.xlim([0,10])
        PLOT.title(r'Noise')
        PLOT.savefig('noise.png')
        PLOT.close()

      return cov

    def plotTravelTimeMaps(self, dayNb, dopplergramNb, fileName):

      ''' Plots the travel time maps number dopplergramNb 
          (between 0 and number of dopplergrams per day) 
          of the day dayNb
      '''

      tauVal = NP.load('%s/%s'%(self.XSmap_.doppler_.directory_, self.fileNames_[dayNb][dopplergramNb]))
      PLOT.figure()
      PLOT.rc('text',usetex=True)
      PLOT.rc('font',family='serif')
      im = PLOT.pcolormesh(tauVal)
      PLOT.colorbar(im)
      PLOT.savefig(fileName, dpi=72)
      PLOT.close()


def getTerm1NoiseQuadrant(dy,Nx,inds1,inds2,w1,w2,Crefxw3D):

  term1 = NP.zeros((Nx,Crefxw3D.shape[2]),dtype='complex')
  for dx in range(Nx):
    for delta1 in range(inds1.shape[1]):
      for delta2 in range(inds2.shape[1]):
        iy = (inds1[0][delta1]-inds2[0][delta2]-dy) % Nx
        ix = (inds1[1][delta1]-inds2[1][delta2]-dx) % Nx
        term1[dx,:] += w1[delta1]*w2[delta2]*Crefxw3D[iy,ix,:]
  return term1



