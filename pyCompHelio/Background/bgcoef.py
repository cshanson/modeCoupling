import numpy as NP
import scipy.interpolate as ITP
import os
from ..Common import *


class BGCoef:
    ''' 
        Background coefficient class for backgroundModel: density, sound speed and pressure
        inherit from this class.
        Data from the background file is stored upon initialization.
        Evaluations of the coefficients are done with the __call__ routine

        RADIAL and BACKGROUND types are evaluated using interpolation of background data file
        INTERP2D uses interp2d (wow!) with data from given file
        NODAL uses data already computed on nodal points (2D mesh)

        In order to use add a perturbation to the background coefficient,
        the entry
        Perturbation = coef_name1 <pert_options>, coef_name2 <pert_options>
        must be present in the configuration file (.init)
    '''

    def __init__(self,config):
        ''' 
            Reads the configfile to setup type and parameters
        '''

        # Get config keywords
        if hasattr(self,'default_'):
          opt = self.default_
        else:
          opt = 'BACKGROUND %d' % self.numCol_
    
        options              = config(self.keyword_,opt).split()
        self.bg_file_         = config('BackgroundFile',pathToMPS()+'/data/background/modelS_SI_reversed.txt')
        self.nodalPointsFile = ''   

        if len(options) == 0:
          raise ValueError(self.UIName_+': no parameters found!')
    
        # -------------------------------------------------------------------------------------------------
        if options[0].upper() == 'CONSTANT':
          self.typeBase = SpatialTypes.UNIFORM
          try:
            self.coefBase = evalFloat(options[1])
          except:
            raise ValueError(self.UIName_+':Could not read constant value for uniform %s' %self.keyword_)

        # -------------------------------------------------------------------------------------------------
        elif options[0].upper() == 'RADIAL':
          self.typeBase = SpatialTypes.RADIAL
          try:
            self.filename = options[1]
          except:
            raise ValueError('No radial file specified for %s.'%self.keyword_)
          try:
            self.radius = NP.loadtxt(self.filename,comments='#')[:,0]
            self.coefBase = NP.loadtxt(self.filename,comments='#')[:,1]
          except:
            raise IOError('Could not load file for '+self.keyword_+': '+ self.filename)
    
        # -------------------------------------------------------------------------------------------------
        elif options[0].upper() == 'BACKGROUND':
          self.typeBase = SpatialTypes.RADIAL
          try:
            column = int(options[1])
          except:
            raise ValueError('Could not determine column of background file for %s'%self.keyword_)
          try:
            self.radius   = NP.loadtxt(self.bg_file_,comments='#')[:,0]
            self.coefBase = NP.loadtxt(self.bg_file_,comments='#')[:,column-1]
          except:
            raise IOError('Could not load background file for %s: '%self.keyword_+self.bg_file_)
    
        # -------------------------------------------------------------------------------------------------
        elif options[0].upper() == 'INTERP2D':
          self.typeBase = SpatialTypes.INTERP2D
          try:
            self.fileName_ = options[1]
          except:
            raise ValueError('No interp2D file specified for %s.'%self.keyword_)
          try:
            iStrR      = options.index('R')
            iStrTH     = options.index('THETA')
            self.r_    = readCoordinatesOptions(options[iStrR :iStrTH],self.bg_file_)
            self.th_   = readCoordinatesOptions(options[iStrTH:      ],self.bg_file_)
          except:
            raise IOError('Problem reading options ')

          try:
            self.coefBase = NP.loadtxt(self.fileName_,comments='#')
          except:
            raise IOError('Could not load file for '+self.keyword_+': '+ self.fileName_)

          if len(self.r_ ) != self.coefBase.shape[0]\
          or len(self.th_) != self.coefBase.shape[1]:
            exstring = "Shape not compatible between loaded data " + self.coefBase.shape +\
                       " and given options Nr: %d, Ntheta: %d " %(len(self.r_),len(self.th_))
            raise Exception(exstring)

        # -------------------------------------------------------------------------------------------------
        elif options[0].upper() == 'NODAL':
          raise NotImplementedError('No NODAL points options for base %s (perturbations ok though)'%self.keyword_) 
        else:
          raise ValueError('Unrecognized type '+options[0]+' for '+self.keyword_)
    
        # ====================================================================
        # Check perturbation

        self.perturbed = False
        self.pertType  = SpatialTypes.UNIFORM
        Options = config('Perturbation','NO').split(',')
        for options in Options:
          options = options.split()
          if options[0] == self.keyword_:
            self.perturbed = True
            try:
              if options[1] == 'CONSTANT':
                self.pert = evalFloat(options[2])
              elif options[1] == 'RADIAL':
                print(options)
                try:
                  self.pert     = NP.loadtxt(options[2])[:,1]
                  self.pertType = SpatialTypes.RADIAL
                except:
                  raise IOError('Creation perturbation for %s: unable to open perturbation file %s ' % (self.keyword_, options[2]))
              elif options[1] == 'RING':
                Ampl = evalFloat(options[2])
                Cntr = evalFloat(options[3])
                Wdth = evalFloat(options[4])
                try:
                  self.radius = NP.loadtxt(self.bg_file_,comments='#')[:,0]
                except:
                  raise IOError('Creation perturbation for %s: unable to open background model file to get radiuses'%self.keyword_)
                self.pert     = Ampl*NP.exp(-(self.radius-Cntr)**2/(2*Wdth**2))
                self.pertType = SpatialTypes.RADIAL
              elif options[1] == 'LOBE':
                self.pertAmpl = evalFloat(options[2])
                self.pertCtrx = evalFloat(options[3])
                self.pertCtry = evalFloat(options[4])
                self.pertWdth = evalFloat(options[5])
                self.pertType = SpatialTypes.NODAL

            except:
              raise ValueError('Trouble reading options of %s perturbation'%self.keyword_)
    
        # ===================================================================

    def spatialType(self):
        ''' 
            Returns the type of spatial output (UNIFORM,RADIAL,INTERP2D,NODAL)
        '''
        return max(self.typeBase,self.pertType)

    # ==================================================================
    # Returns the value of the coefficient on given points 
    # or the stored array if no points are given

    def __call__(self,points=None,nodalPoints=None,geom=None):
        ''' 
            Returns original or perturbed coefficient.
            - If given, points must be in cartesian coordinates: (3) or (3,N).
            - If a nodal points file is given: the routine will check.
              if the coefficient has been computed for this file, otherwise, recompute.
            - If a Geometry instance is given, the values of the coefficients will be returned
              in a array of shape geom.N_ corresponding to geometry points.
        '''     
        return self.getCoef(points,nodalPoints,geom)

    def getCoef(self,points=None,nodalPoints=None,geom=None):

        # Check if the coeff is asked on nodal Points
        nodalOutput = (points is None and geom is None)

        # ------------------------------------------------------
        # Check if coefficient needs to be computed a first time
        if not hasattr(self,'coef'):
          self._BGCoef__computeCoef(nodalPoints)

        if nodalPoints is not None:
          if nodalPoints.fileName != self.nodalPointsFile and self.spatialType()==SpatialTypes.NODAL:
            self._BGCoef__computeCoef(nodalPoints)

        # -------------------------------------------------
        # Determine points where to compute the coefficient

        if points is None:
          if geom is not None:
            points = geom.getCartesianCoordsMeshGrid()
          else:
            if nodalPoints is not None:
              x,z    = nodalPoints.getCartesianCoords()
              points = NP.array([x,NP.zeros(x.shape),z])

        if points is None:
          return self.coef

        points = NP.array(points)
        if points.shape[0] != 3:
          print("\nGiven points not compatible with flow computation routines:")
          print("Coordinates must be cartesian and with a dimensions (3) or (3,...).\n")
          raise Exception()

        # -------------------------------------------------
        # Compute coeff

        if self.spatialType() == SpatialTypes.UNIFORM:
          return self.coef*NP.ones(points.shape[1:])
        elif self.spatialType() == SpatialTypes.RADIAL:
          coefITP  = ITP.interp1d(self.radius,self.coef,bounds_error=False,fill_value=0.e0)
          radiuses = NP.sqrt(NP.sum(points**2,axis=0))
          return coefITP(radiuses)
        elif self.spatialType() == SpatialTypes.INTERP2D:
          ptsS    = cartesianToSpherical(points)
          coefITP = ITP.RectBivariateSpline(self.r_,self.th_,self.coef)
          res     = coefITP(ptsS[0],ptsS[1],grid=False)
          # Replace by zeros outside the interpolation domain
          return NP.where( (ptsS[0] >= NP.amin(self.r_ ))*(ptsS[0]<=NP.amax(self.r_ ))\
                          *(ptsS[1] >= NP.amin(self.th_))*(ptsS[1]<=NP.amax(self.th_)),res,0.e0)
             
        else:
          points = NP.array(points)
          if self.spatialType() == SpatialTypes.UNIFORM:
            return self.coef*NP.ones(points.shape[0])
          elif self.spatialType() == SpatialTypes.RADIAL:
            coefITP = ITP.interp1d(self.radius,self.coef)
            if points.ndim == 1:
              return coefITP(points)
            else:
              radiuses = NP.sqrt(NP.sum(points**2,axis=1))
              return coefITP(radiuses)
          else:
            if nodalOutput:
              return self.coef
            else:
              if not hasattr(nodalPoints, 'points'):
                nodalPts.computePoints()
              radiuses = NP.sqrt(NP.sum(nodalPoints.points**2,axis=1))
              indMax = NP.argmax(radiuses)
              valMax = self.coef[indMax] # value for outside points
              val = ITP.griddata(nodalPoints.points, self.coef, (points[0], points[2]), method='cubic',fill_value=valMax)
              return val
            #  raise NotImplementedError("Interpolation from nodal points not implemented")

    # ==================================================================

    def __computeCoef(self,nodalPoints=None):
    
      if not self.perturbed:
        self.coef = self.coefBase
      else:  
        if self.pertType in [SpatialTypes.UNIFORM,SpatialTypes.RADIAL]:
          self.coef = self.coefBase + self.pert        
        else:
          if nodalPoints is None:
            raise ValueError('Please provide a nodal points filename for perturbation computation')
          else:
            self.nodalPointsFile = nodalPoints.fileName
            if not hasattr(nodalPoints, 'points'):
              nodalPoints.computePoints()
            distances    = NP.sqrt((nodalPoints.points[:,0]-self.pertCtrx)**2 + (nodalPoints.points[:,1]-self.pertCtry)**2)
            perturbation = self.pertAmpl*NP.exp(-distances**2/(2*self.pertWdth**2))

            if self.typeBase == SpatialTypes.UNIFORM:
              self.coef = self.coefBase + perturbation
            elif self.typeBase == SpatialTypes.RADIAL:
              radiuses  = NP.sqrt(NP.sum(nodalPoints.points**2,axis=-1))
              coefITP   = ITP.interp1d(self.radius,self.coefBase,bounds_error=False, fill_value="extrapolate")
              self.coef = coefITP(radiuses) + perturbation

    def getGradientLog(self,points=None):

      if   self.spatialType() == SpatialTypes.NODAL:
        raise NotImplementedError("Gradient of NODAL coefficient not implemented")
      elif self.spatialType() == SpatialTypes.UNIFORM:
        if points is None:
          return 0
        else:
          return NP.zeros(points.shape)
      else: # RADIAL
        dr = FDM_Compact(self.radius*RSUN)
        if points is None:
          return dr(NP.log(self.coef))
        else:
          radiuses = NP.sqrt(NP.sum(points**2,axis=0))
          dlcoef   = dr(NP.log(self.coef))
          dlcoef   = ITP.interp1d(self.radius,dlcoef)
          return dlcoef(radiuses)

    def getGradient(self,points=None,geom=None):

      if points is None:
        if geom is not None:
          points = geom.getCartesianCoordsMeshGrid()

      if points is not None:
        points = NP.array(points)

      if self.spatialType() == SpatialTypes.UNIFORM:
        if points is None:
          return 0
        else:
          return NP.zeros((points.shape))
      elif self.spatialType() == SpatialTypes.RADIAL:
        # Load dcoef/dRSUN from background file
        dcoef = NP.loadtxt(self.bg_file_,comments='#')[:,self.numColDeriv_-1]
        if points is not None:
          radiuses = NP.sqrt(NP.sum(points**2,axis=0))
          dcoefItp = ITP.interp1d(self.radius,dcoef)
          dcoef    = dcoefItp(radiuses)
          DCOEF    = NP.zeros(points.shape)
          DCOEF[0] = dcoef
        else:
          DCOEF    = NP.zeros((3,len(dcoef)))
          DCOEF[0] = dcoef
        return DCOEF
          
      elif self.spatialType() == SpatialTypes.INTERP2D:
        if geom:
          points = geom.GetCartesianCoordsMeshGrid()
          coef   = self.getCoef(points)
          return geom.sphericalGradient(coef)
        else:
          # Compute gradient of coefBase and interpolate
          dims = list(self.coefBase.shape)
          dims.insert(0,2)
          DCoefBase = NP.zeros(dims)
          dr = FDM_Compact(self.r_)
          for i in range(len(self.th_)):
            DCoefBase[0,:,i] = dr(self.coefBase[:,i])
          dth = FDM_Compact(self.th_)
          for i in range(len(self.r_)):
            DCoefBase[1,i,:] = dth(self.coefBase[i,:])/self.r_[i]
          
          DCOEF = NP.zeros(points.shape)
          ptsS  = cartesianToSpherical(points)
          for i in [0,1]:
            dcoefITP = ITP.RectBivariateSpline(self.r_,self.th_,DCoefBase[0])
            DCOEF[i] = dcoefITP(ptsS[0],ptsS[1],grid=False)
            DCOEF[i] = NP.where( (ptsS[0] >= NP.amin(self.r_ ))*(ptsS[0]<=NP.amax(self.r_ ))\
                                *(ptsS[1] >= NP.amin(self.th_))*(ptsS[1]<=NP.amax(self.th_)),DCOEF[i],0.e0)
          return DCOEF

      else:
        if geom:
          points = geom.GetCartesianCoordsMeshGrid()
          coef   = self.getCoef(points)
          return geom.sphericalGradient(coef)
        else:
          # pass
          raise Exception('Cannot compute gradient of %s without geometry'%self.keyword_)

class Density(BGCoef):
    def __init__(self,config):
      self.keyword_     = 'rho'
      self.numCol_      = 3
      self.numColDeriv_ = 5
      self.UIName_      = 'Density'
      self.default_     = 'BACKGROUND 3'
      BGCoef.__init__(self,config)

class Pressure(BGCoef):
    def __init__(self,config):
      self.keyword_     = 'p'
      self.numCol_      = 6
      self.numColDeriv_ = 7
      self.UIName_      = 'Pressure'
      self.default_     = 'BACKGROUND 6'
      BGCoef.__init__(self,config)

class SoundSpeed(BGCoef):
    def __init__(self,config):
      self.keyword_     = 'c'
      self.numCol_      = 2
      self.numColDeriv_ = 4
      self.UIName_      = 'SoundSpeed'
      self.default_     = 'BACKGROUND 2'
      BGCoef.__init__(self,config)

class SpatialDamping(BGCoef):
    def __init__(self,config):
      self.keyword_     = 'DampingSpatial'
      self.numCol_      = -1
      self.numColDeriv_ = -1
      self.UIName_      = 'damping spatial dependancy'
      self.default_     = 'CONSTANT 1.'
      BGCoef.__init__(self,config)


