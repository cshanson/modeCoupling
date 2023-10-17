import numpy as NP
import scipy.interpolate as ITP
import scipy.optimize    as OPT
from ..Common import *
from  .bgcoef import *

class Damping:
  ''' 
      Damping class for backgroundModel.
      Damping is splitted into its spatial dependency, which is represented by a bgcoeff instance
      and its frequency dependency, treated in this file.

      Definitions of damping as function of frequency  are to be written
      in ComputeFrequencyTerm
  '''

  def __init__(self,config):
    ''' 
        Reads the configfile to setup type and parameters
    '''

    self.config = config
    fOptions    = config('Damping','CONSTANT 0').split()
    sOptions    = config('DampingSpatial','UNIFORM 1').split()

    if fOptions[0] == 'SOLAR_FWHM':
      config.set('gamma_L_DEP','TRUE')
    else:
      config.set('gamma_L_DEP','FALSE')

    if config('DampingRW',0):
      fOptions = ['CONSTANT','0']
      sOptions = ['UNIFORM', '1']
      

    self.spatialDamping_ = SpatialDamping(config) 

    self.setFreqOptions (fOptions)

    self.mult = evalFloat(self.config('DampingFactor','1.e0'))

    if config('DampingRW',0):
      self.dependsUponFrequency = True
      self.DampingRW = True
    else:
      self.DampingRW = False

  ######################################################################
  def setFreqOptions(self,options):

    self.dependsUponFrequency = False

    if len(options) == 0:
      raise ValueError('Damping: no parameters found!')

    ####################################################################
    if options[0].upper() == 'CONSTANT':
      self.typeFreq = DampingTypes.CONSTANT
      try:
        self.value = evalFloat(options[1])
      except:
        raise ValueError('Damping: could not read value for CONSTANT damping')

    ####################################################################
    if options[0].upper() == 'POLYNOMIAL':
      self.dependsUponFrequency = True
      self.typeFreq             = DampingTypes.POLYNOMIAL
      self.PolyDegree           = len(options[1:]) - 1
      self.coeffs_              = []
      try:
        for i in range(len(options[1:])):
          self.coeffs_.append(evalFloat(options[i+1]))
      except:
        raise ValueErr('Damping: could not read value for POLYNOMIAL damping')


    ####################################################################
    elif options[0].upper() == 'PROPORTIONAL':
      self.dependsUponFrequency = True
      self.typeFreq             = DampingTypes.PROPORTIONAL
      try:
        self.factorFreq = evalFloat(options[1])
      except:
        raise ValueError('Damping: No given multiplier after PROPORTIONAL.')

    ####################################################################
    elif options[0].upper() == 'EXP':
      self.dependsUponFrequency = True
      self.typeFreq             = DampingTypes.EXP
      try:
        self.f0 = evalFloat(options[1])
        self.A  = evalFloat(options[2])
        self.B  = evalFloat(options[3])
        self.Dw = evalFloat(options[4])
      except:
        raise ValueError('Impossible to read parameters for EXP damping type.')

    ####################################################################
    elif options[0].upper() == 'WW0':
      self.dependsUponFrequency = True
      self.typeFreq             = DampingTypes.WW0
      try:
        self.f0     = evalFloat(options[1])
        self.gamPow = evalFloat(options[2])
      except:
        raise ValueError('Impossible to read parameters for WW0 damping type.')
      
    ####################################################################
    elif options[0].upper() == 'FREQFILE':
      self.dependsUponFrequency = True
      self.typeFreq             = DampingTypes.FREQFILE
      try:
        self.freqfilename = options[1]
      except:
        raise ValueError('Impossible to read parameters for FREQFILE damping type.')
      if len(options) > 2:
        self.univariateSmooth = int(options[2])
      else:
        self.univariateSmooth = None
    ####################################################################
    elif options[0].upper() == "LRANGE_SPLINE":
      self.dependsUponFrequency = True
      self.typeFreq             = DampingTypes.LRANGE_SPLINE
      try:
        self.lmin     = evalFloat(options[1])
        self.lmax     = evalFloat(options[2])
        self.FWHMType = options[3].upper()
        try:
          self.binwidth = evalFloat(damping[4])
        except:
          self.binwidth = 25
      except:
        raise ValueError('Impossible to read parameters for LRANGE_SPLINE damping type.')

    ####################################################################
    elif options[0].upper() == "LRANGE_PLAW":
      self.dependsUponFrequency = True
      self.typeFreq             = DampingTypes.LRANGE_PLAW
      try:
        self.lmin   = evalFloat(options[1])
        self.lmax   = evalFloat(options[2])
        self.w0     = evalFloat(options[3]) * 1e6
        self.pmax   = 22
        self.obsmin = 5.e-7
        self.ampl   = 1
        try: 
          self.pmax   = float(options[4])
          self.obsmin = float(options[5])
          self.ampl   = float(options[6])
        except:
          pass
        self.binwidth = 25
      except:
        raise ValueError('Impossible to read parameters for LRANGE_PLAW damping type.')

    ####################################################################
    elif options[0].upper() == "PHASESPEED_SPLINE":

      self.dependsUponFrequency = True
      self.typeFreq             = DampingTypes.PHASESPEED_SPLINE
      try:
        try: 
          self.nosigma = evalFloat(options[3])
        except:
          self.nosigma = 3.e0
        self.v0 = evalFloat(options[1])
        self.dv = evalFloat(options[2])*self.nosigma
        try:
          self.binwidth = evalFloat(damping[4])
        except:
          self.binwidth = 25

      except:
        raise ValueError('Impossible to read parameters for PHASESPEED_SPLINE damping type.')

    ####################################################################
    elif options[0].upper() == "PHASESPEED_PLAW":

      self.dependsUponFrequency = True
      self.typeFreq             = DampingTypes.PHASESPEED_PLAW
      try:
        self.nosigma = 3.e0 
        self.v0      = evalFloat(options[1])
        self.dv      = evalFloat(options[2])*self.nosigma
        self.w0      = evalFloat(options[3])*1.e6
        try:
          self.continuous = (options[4]=='True')
        except:
          self.continuous = False
        self.binwidth = 25
      except:
        raise ValueError('Impossible to read parameters for PHASESPEED_PLAW damping type.')

    ####################################################################
    elif options[0].upper() == 'SOLAR_FWHM':
      self.dependsUponFrequency = True
      try:
        self.fwhm_min = evalFloat(options[1])
      except:
        self.fwhm_min = 0
      if self.config('TypeElement').upper() == 'EDGE_LOBATTO':
        self.L_Dep    = True
        self.typeFreq = DampingTypes.L_DEP
      else:
        raise Exception('Only 1-D Simulations can have ell dependant damping')

  ######################################################################

  def spatialType(self):
    ''' 
        Returns the type of spatial output (UNIFORM,RADIAL,INTERP2D,NODAL)
    '''
    return self.spatialDamping_.spatialType()

  # =====================================================================
  # Returns the value of the damping on given points at given frequency

  def __call__(self,points=None,freq=None,nodalPoints=None,rads=False,geom=None):
    return self.getDamping(points,freq,nodalPoints,rads,geom)

  def getDamping(self,points=None,freq=None,nodalPoints=None,rads=False,geom=None):
    ''' Damping is written as f(w)g(x), which are computed separatly '''

    if not self.config('DampingRW',0):

      f = self.computeFrequencyTerm(freq)
      try:
        if len(f) == 1:
          f = f[0]
      except:
        f = f  
      #g = self.computeSpatialTerm(points,nodalPoints)
      g = self.spatialDamping_(points,nodalPoints,geom)

      if rads:
        f *= 2*NP.pi

      # relate FWHM to gamma
      if self.typeFreq in {DampingTypes.EXP,DampingTypes.LRANGE_SPLINE,
                           DampingTypes.LRANGE_PLAW,DampingTypes.PHASESPEED_SPLINE,
                           DampingTypes.PHASESPEED_PLAW,DampingTypes.L_DEP}:
        f = f/2.

      return f*g*self.mult
    else:
      opts = self.config('DampingRW').split()
      if opts[0].upper() == 'FILE':
        omega,r,gamma = NP.load(opts[1],allow_pickle=True)
      elif opts[0].upper() == 'BASIS':
        basis2D = NP.load(opts[1])
        basis_coeff = NP.load(opts[2])

      # Create interpoltion function
      # ff,rr = NP.meshgrid(omega/(2*NP.pi),r,indexing='ij')
      # coefITP  = ITP.interp2d(ff,rr,gamma,bounds_error=False,fill_value=0.e0)


      if points is None:
        if geom is not None:
          points = geom.getCartesianCoordsMeshGrid()
        else:
          if nodalPoints is not None:
            x,z    = nodalPoints.getCartesianCoords()
            points = NP.array([x,NP.zeros(x.shape),z])

      
      if points is None:
        radii = NP.genfromtxt(self.config('BackgroundFile'))[:,0]
      else:
        points = NP.array(points)
        radii = NP.sqrt(NP.sum(points**2,axis=0))

      if not hasattr(freq,'__len__'):
        freq = NP.array([freq])

      # coefITP = ITP.interp2d(r,omega/(2*NP.pi),gamma,kind='linear',bounds_error=False,fill_value=0.e0)
      coefITP = ITP.interp2d(r,omega/(2*NP.pi),NP.log10(gamma + 1.e-20),kind='linear',bounds_error=False,fill_value=[-20,NP.log10(NP.amin(gamma)+1e-20)][int(NP.amin(gamma) != 0.)])

      values  = 10**coefITP(radii,freq)

      # values = []

      # for i in freq:
      #   ind = NP.argmin(abs(omega - abs(i)*2*NP.pi))
      #   coefITP = ITP.interp1d(r,gamma[ind],bounds_error=False,fill_value=0.e0)
      #   values.append(coefITP(radii))
      # values = NP.array(values)
      # # if radii.ndim >1:
      # values = NP.squeeze(NP.rollaxis(values,0,len(radii.shape)+1))

      # if radii.ndim > 1:
      #   values = []
      #   for i in range(radii.shape[1]):
      #     values.append(coefITP(freq,radii[:,i]))
      #   values = NP.squeeze(NP.array(values)).T
      # else:
      #   values = coefITP(freq,radii)

      if rads:
        values *= 2*NP.pi
      
      return values*self.mult
        

  #======================================================================

  def computeSpatialTerm(self,points=None,nodalPoints=None):
    ''' computes the part of gamma which varies in space'''

    if nodalPoints is not None:
      points = nodalPoints.points

    #================
    # Uniform damping
    if self.typeSpace == SpatialTypes.UNIFORM:
      if points is None:
        ans = self.uniformValue
      else:
        ans = self.uniformValue*NP.ones(points.shape[0])
      
    #==============================================
    # Radially varying damping : loaded from a file
    elif self.typeSpace == SpatialTypes.RADIAL:

      try:
        r_file = NP.loadtxt(self.filename)[:,0]
        g_file = NP.loadtxt(self.filename)[:,1]
      except:
        raise IOError("Could not open file to load radial damping.")
      
      if points is None:
        ans = g_file
      else:
        g = ITP.interp1d(r_file,g_file)
        if points.ndim == 1: 
          ans = g(points)
        else:
          radiuses = NP.sqrt(NP.sum(points**2,axis=1))
          ans = g(radiuses)

    #==============================
    # Damping given on nodal points
    elif self.typeSpace == SpatialTypes.NODAL:
      if points is not None:
        raise ValueError("Interpolation on given points impossible from data on nodal points.") 
      try:
        ans = NP.loadtxt(self.filename)
      except:
        raise IOError("Could not open file to load damping on nodal points.")

    return ans

  #======================================================================

  def computeFrequencyTerm(self,freq):
    ''' computes the frequency dependant part of damping'''

    #####################################################################
    # CONSTANT
    if self.typeFreq == DampingTypes.CONSTANT:
      return self.value

    #####################################################################
    # POLYNOMIAL
    if self.typeFreq == DampingTypes.POLYNOMIAL:
      self.value = 0
      for i in range(self.PolyDegree+1):
        self.value += self.coeffs_[i] * abs(freq*2*NP.pi)**i
      self.value = self.value/(2*NP.pi)
      return self.value

    #####################################################################
    # PROPORTIONAL
    elif self.typeFreq == DampingTypes.PROPORTIONAL:
      return freq*self.factorFreq

    #####################################################################
    # EXP
    elif self.typeFreq == DampingTypes.EXP:
      freqLimit = 0.005231929050467
      minFreq   = NP.where(abs(freq) < freqLimit,freq,freqLimit)
      return self.A * NP.exp(self.B *(abs(minFreq/self.f0)-1)) + self.Dw/2.e0

    #####################################################################
    # WW0: (w/w0)^gamma
    elif self.typeFreq == DampingTypes.WW0:
      return pow(abs(freq/self.f0),self.gamPow)

    #####################################################################
    # FILE:
    elif self.typeFreq == DampingTypes.FREQFILE:
      try:
        freq_f,gamma_f = NP.load(self.freqfilename)
      except:
        raise IOError("Damping FREQFILE: could not open "+self.freqfilename)

      g = ITP.UnivariateSpline(freq_f,gamma_f,s=self.univariateSmooth)
      return g(abs(freq))

    #####################################################################
    elif self.typeFreq == DampingTypes.L_DEP:
      Lmax = int(self.config('MaximumDegree').split()[0])+1
      ls = NP.arange(0,Lmax)
      Freq = freq * 1.e6
      # Interpolator = NP.load(pathToMPS() + '/data/Observations/SOLAR_FWHM_log.npy').item()
      ell_data,n_data,nu_data,fwhm_data,dfwhm_data = NP.genfromtxt(pathToMPS() + '/data/Observations/FWHM_OBS/FWHM_artificial.dat').T
      MAT = (fwhm_data > 0)
      ell_data,n_data,nu_data,fwhm_data,dfwhm_data = NP.genfromtxt(pathToMPS() + '/data/Observations/FWHM_OBS/FWHM_artificial.dat').T[:,MAT]
      # FWHM = NP.reshape(NP.exp(Interpolator(ls,Freq)),len(ls))/1.e6
      FWHM = 10**(ITP.griddata((ell_data, nu_data), NP.log10(fwhm_data), (ls, Freq), method='linear',fill_value=20))*1.e-6
      FWHM[FWHM<self.fwhm_min] = self.fwhm_min

      return FWHM
    #####################################################################
    # Fit from observations data
    elif self.typeFreq == DampingTypes.LRANGE_SPLINE:
      ''' fits an exponential of a polynomial of degree 2 to the 
          FWHM data of Korzennik for a given lmin and lmax range '''

      # Check if the fit was already done
      fitFile = self.config('OutDir')+'/TMPdampingParams.npy'
      try:
        x,y = NP.load(fitFile)
      except:

        # --------------------------------------------------------
        # Recover data to fit
        if self.FWHMType == 'RIDGE':

          if self.lmin<100:
            raise Exception('Error: RIDGE fitting cannot be done for l < 100, please use MODE option')
 
          MAT = NP.genfromtxt(pathToMPS() + '/data/Observations/KORZENNIK_DATA/multiplets-mdi-2001.dat')
          MAT = MAT[NP.sign(MAT[:,8])!=-1.]
          output = MAT   [MAT   [:,1]<self.lmax]
          output = output[output[:,1]>self.lmin]
          # remove f-mode from fit
          output = output[output[:,0]!=0]
          # Sort by ascending mode Frequency (for the fit)
          output = output[output[:,2].argsort()]

          freqK = output[:,2]*1e-6
          FWHM  = output[:,8]*1e-6

        if self.FWHMType == 'MODE':

          MAT = NP.genfromtxt(pathToMPS() + '/data/Observations/FWHM_DATA.dat')
          MAT = MAT[NP.sign(MAT[:,3])!=-1]
          # Select ls between lmin and lmax then remove f-mode
          output = MAT   [MAT   [:,0]<self.lmax]
          output = output[output[:,0]>self.lmin]
          # remove f-mode from fit
          output = output[output[:,1]!=0]
          # Sort by ascending mode Frequency (for the fit)
          output = output[output[:,2].argsort()]

          freqK     = output[:,2]*1e-6
          FWHM      = output[:,3]*1e-6
        # --------------------------------------------------------

        x,y   = self.binData(freqK,FWHM)

        x = NP.concatenate(([NP.amin(freqK)],x,[NP.amax(freqK)]))
        y = NP.concatenate(([NP.amin(FWHM )],y,[NP.amax(FWHM )]))

        NP.save(fitFile,[x,y])
      # ---------------------------------------------------------

      if   abs(freq) < NP.amin(abs(x)):
        return y[0]
      elif abs(freq) > NP.amax(abs(x)):
        return y[-1]
      else:
        f     = ITP.interp1d(x,y,kind='cubic')
        return f(abs(freq))

    #####################################################################
    # FITS part 2
    elif self.typeFreq == DampingTypes.LRANGE_PLAW:
      ''' fits a power law to the FWHM data of Korzennik 
          for a given lmin and lmax range
          gamma = gamma0(w/w0)**m '''

      freqOpts = self.config('Frequencies','CONSTANT 1.e0').split()
      
      if   freqOpts[0].upper() == 'POSITIVE_ONLY':
        homega = 1.e0/evalFloat(freqOpts[1])*1.e6
      elif freqOpts[0] == 'RANGE':
        homega = evalFloat(freqOpts[3])*1.e6

      # Check if fit already performed
      fitFile = self.config('OutDir')+'/TMPdampingParams.npy'
      try:
        [qout,gamma0,freqK] = NP.load(fitFile)
      except:
        MAT    = NP.genfromtxt(pathToMPS()+'/data/Observations/FWHM_DATA.dat')
        MAT    = MAT[NP.sign(MAT[:,3])!=-1.]
        output = MAT   [MAT   [:,0]<self.lmax]
        output = output[output[:,0]>self.lmin]
        # remove f-mode from fit
        output = output[output[:,1]!=0]
        output = output[output[:,1] <= self.pmax]
        # Sort by ascending mode Frequency (for the fit)
        output = output[output[:,2].argsort()]

        length = int(output.shape[0]/25)
        x      = output[:,2][:(output[:,2].size//length)*length].reshape(-1,length).mean(axis=1)
        y      = output[:,3][:(output[:,3].size//length)*length].reshape(-1,length).mean(axis=1)

        freqK  = output[:,2]
        FWHM   = output[:,3]
        gamma0 = y[NP.argmin(abs(x-self.w0))]

        def fitfunc(p,x):
          return x**p[0]
        def errfunc(p,x,y):
          return y-fitfunc(p,x)

        xdata = freqK/self.w0
        ydata = FWHM /gamma0

        qout,success = OPT.leastsq(errfunc,5,args=(xdata,ydata),maxfev=3000)
        if not os.path.isdir(self.config('OutDir')):
          mkdir_p(self.config('OutDir'))
        NP.save(fitFile,[qout,gamma0,NP.amax(freqK)])

      minF = NP.minimum(abs(freq*1.e6),NP.amax(freqK))
      return gamma0*self.ampl*((minF/self.w0)**qout)*1e-6 + self.obsmin

    #####################################################################
    # FITS part 3
    elif self.typeFreq == DampingTypes.PHASESPEED_SPLINE:
      fitFile = self.config('OutDir')+'/TMPdampingParams.npy'
      try:
        x,y = NP.load(fitFile)
      except:
        ## Load in Data and Parameters
        MAT = NP.genfromtxt(pathToMPS()+'/data/Observations/FWHM_DATA.dat')
        MAT = MAT[NP.sign(MAT[:,3])!=-1.]
        # Remove the f-mode
        MAT = MAT[MAT[:,1]!=0]

        # obtain the angular frequencies of the modes in Hz
        omega = 2*NP.pi * MAT[:,2] * 1e-6
        # Now the Wave numbers (rad/m)
        k = MAT[:,0] / (696*1.e6)
        # Now filter out the modes that wont be in the Phase speed filter
        NP.seterr(divide='ignore')
        MAT = MAT[(omega/k < (self.v0+self.dv/2.)) * (omega/k > (self.v0-self.dv/2.))]
        NP.seterr(divide=None)

        # Sort the remaining data by increasing mode frequency (for the interpolation to work)
        MAT = MAT[MAT[:,2].argsort()]

        freqK = MAT[:,2]*1.e-6
        FWHM  = MAT[:,3]*1.e-6
        x,y   = self.binData(freqK,FWHM)

        NP.save(fitFile,[x,y])

      if   abs(freq) < NP.amin(abs(x)):
        return y[0]
      elif abs(freq) > NP.amax(abs(x)):
        return y[-1]
      else:
        f = ITP.interp1d(x,y,kind='cubic')
        return f(abs(freq))

    #####################################################################
    # FITS part 4
    elif self.typeFreq == DampingTypes.PHASESPEED_PLAW:
      print(' NEED FIXES TO BE LIKE LRANGE_PLAW')

      fitFile = self.config('OutDir')+'/TMPdampingParams.npy'
      try:
        gamma0,qout,wac = NP.load(fitFile)
      except:
        ## Load in Data and Parameters
        MAT = NP.genfromtxt(MPS + '/data/Observations/FWHM_DATA.dat')
        MAT = MAT[NP.sign(MAT[:,3])!=-1.]

        # Remove the f-mode
        MAT = MAT[MAT[:,1]!=0]

        # obtain the angular frequencies of the modes in Hz
        omega = 2*NP.pi * MAT[:,2] * 1e-6
        # Now the Wave numbers (rad/m)
        k = MAT[:,0] / (696*1.e6)
        # Now filter out the modes that wont be in the Phase speed filter
        NP.seterr(divide='ignore')
        MAT = MAT[(omega/k < (self.v0+self.dv/2.)) * (omega/k > (self.v0-self.dv/2.))]
        NP.seterr(divide=None)
        # Sort the remaining data by increasing mode frequency (for the interpolation to work)
        MAT = MAT[MAT[:,2].argsort()]

        freqK = MAT[:,2]
        FWHM  = MAT[:,3]
        x,y   = self.binData(freqK,FWHM)        

        gamma0 = y[NP.argmin(abs(x -self.w0))]

        def fitfunc(p,x):
          return x**p[0]
        def errfunc(p,x,y):
          return y - fitfunc(p, x)
      
        xdata = freqK/self.w0
        ydata = FWHM/gamma0
        wac   = NP.amax(abs(freqK))
        
        qout,success = OPT.leastsq(errfunc,5,args=(xdata,ydata),maxfev=3000)
        NP.save(fitFile,[gamma0,qout,wac])

      minF = NP.minimum(abs(freq*1.e6),NP.amax(freqK))
      return gamma0*((minF/self.w0)**qout)*1e-6 + self.obsmin

  def binData(self,freqK,FWHM):
     ''' Bin data in order to better fit interpolation '''
     
     length = int(len(freqK)/self.binwidth)
     x = freqK[:(freqK.size//length)*length].reshape(-1,length).mean(axis=1)
     y = FWHM [:(FWHM.size //length)*length].reshape(-1,length).mean(axis=1)

     x = NP.concatenate(([NP.amin(freqK)],x,[NP.amax(freqK)]))
     y = NP.concatenate(([NP.amin(FWHM )],y,[NP.amax(FWHM )]))
     return x,y


class DampingTypes:

  CONSTANT           = 0
  EXP                = 1
  WW0                = 2
  PROPORTIONAL       = 3
  KORZENNIK          = 4
  FREQFILE           = 5
  LRANGE_PLAW        = 6
  LRANGE_SPLINE      = 7
  PHASESPEED_SPLINE  = 8
  PHASESPEED_PLAW    = 9
  L_DEP              = 10
  POLYNOMIAL         = 11


# =======================================================================
# =======================================================================

class Kappa:
  ''' Coefficients of tensorial term kappa in div((mu+kappa*gamma)grad psi)
      in Helmholtz Gamma Laplace equation
  '''

  def __init__(self,config):
    ''' reads the configfile to setup type and parameters'''

    self.kappa = [None]*3
    self.nodalPointsFile = ''

    options = config('GammaLaplacian','ISOTROPE 0').split()

    if len(options) == 0:
      raise ValueError('GammaLaplacian: no parameters found!')

    if options[0].upper() == 'ISOTROPE':

     self.type      = KappaTypes.ISOTROPE
     self.typeSpace = [SpatialTypes.UNIFORM]*3
     try:
       self.kappa   = [evalFloat(options[1])]*3
     except:
       raise ValueError('Unable to read parameters for ISOTROPE gamma laplacian')

    elif options[0].upper() == 'SURFACE':

     self.type      = KappaTypes.SURFACE
     self.typeSpace = [SpatialTypes.UNIFORM]*3
     try:
       self.kappa   = [0.,evalFloat(options[1]),evalFloat(options[1])]
     except:
       raise ValueError('Unable to read parameters for SURFACIC gamma laplacian')

    elif options[0].upper() == 'ORTHOTROPE':

     self.type      = KappaTypes.ORTHOTROPE
     self.typeSpace = [SpatialTypes.UNIFORM]*3
     try:
       self.kappa   = [evalFloat(options[1]),evalFloat(options[2]),evalFloat(options[3])]
     except:
       raise ValueError('Unable to read parameters for ORTHOTROPE gamma laplacian')
    else:
      raise ValueError('Unable to read option for GammaLaplacian = ISOTROPE/SURFACE/ORTHOTROPE')

  # =========================================================================
  # Returns the value of kappa
     
  def __call__(self,component,points=None,nodalPoints=None):
    return self.getKappa(component,points,nodalPoints)
    
  def getKappa(self,component,points=None,nodalPoints=None):
 
    # ======================================
    # CONSTANT kappa : return array with same
    # shape as points if asked
    if (self.typeSpace[component] == SpatialTypes.UNIFORM):

      if nodalPoints is not None:
        points = nodalPoints.points
      value = self.kappa[component]
      if points is None:
        return value 
      else:
        return value*NP.ones((points.shape[0],))
 
    else:
      if points is not None:
        raise NotImplementedError('Interpolation from nodal points not implemented.')
      else:
        if (self.kappa[0] is None) or (self.kappa[1] is None) or (self.kappa[2] is None)\
        or (SpatialTypes.NODAL in self.typeSpace and self.nodalPointsFile != nodalPoints.fileName):   
          self.computeKappa(nodalPoints) 
        return self.kappa[component]

  # =========================================================================
  # Computation routines 

  def computeKappa(self,nodalPoints=None):
      ''' Empty for now, bu there we can setup more options...
          As for the flows
      '''

      if nodalPoints is None:
        self.kappa = [1,1,1]
      else:
        self.kappa = [NP.ones(nodalPoints.N)]*3
        self.nodalPointsFile = nodalPoints.fileName

class KappaTypes:

  ISOTROPE = 0
  SURFACE = 1
  ORTHOTROPE = 2

