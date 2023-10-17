import os
from numpy        import *
from ..Common     import *
from .bgcoef      import *
from .damping     import *
from .flow        import *
from ..Common.nodalPoints import *

class backgroundModel:
    ''' 
        Class that regroups data about physical properties of the medium.
        Members:
          - rho (instance of Density (<bgcoeff) class )
          - c   (instance of SoundSpeed(<bgcoeff) class )
          - damping (instante of Damping class (bgcoeff for spatial dependancy + frequency treatment)
          - flow (instance of the flow class (close to a bgcoeff)
          - kappa (tensor for the div(1/rho+kappa grad( )) term
        Contains links to
          - the configuration file
          - nodalpoints structure
    '''

    def __init__(self,cfg):

        self.cfg         = cfg
        self.nodalPoints = nodalPoints(cfg)
        self.fileName    = cfg('BackgroundFile',pathToMPS()+'/data/background/modelS_SI_reversed.txt')

        self.rho = Density   (self.cfg)
        self.c   = SoundSpeed(self.cfg)
        if self.cfg('TypeEquation').upper() == 'GALBRUN':
          self.p = Pressure(self.cfg)

        if not self.cfg('Damping',0):
          self.cfg.set('Damping','CONSTANT 0.')
        self.damping = Damping(self.cfg)

        if self.cfg('Flow',0):
          self.flow  = Flow(self.cfg)

        if self.cfg('GammaLaplacian',0):
          self.kappa = Kappa(self.cfg)

    # =====================================================================

    def getTypeForCoeffs(self,liste,*coeffList):
        ''' 
            Returns the type of the combination of spatial coefficients
            For example: 
              if rho is of type RADIAL, c of type NODAL, gamma of type UNIFORM
              getTypeForCoeffs('rho','c','damping') will return NODAL. 
            Order is UNIFORM < RADIAL < INTERP2D < NODAL
        '''
      
        Types = []
        if 'rho' in coeffList:
          Types.append(self.rho.spatialType())
        if 'c' in coeffList:
          Types.append(self.c.spatialType())
        if 'p' in coeffList:
          Types.append(self.p.spatialType())
        if 'damping' in coeffList:
          Types.append(self.damping.spatialType())
        if 'M' in coeffList:
          Types.append(self.flow.typeSpace)
        if 'kappaR' in coeffList:
          Types.append(self.kappa.typeSpace[0])
        if 'kappaTheta' in coeffList:
          Types.append(self.kappa.typeSpace[1])
        if 'kappaPhi' in coeffList:
          Types.append(self.kappa.typeSpace[2])
        Types = NP.array(Types)

        if liste:
          return NP.amax(Types),Types
        else:
          return NP.amax(Types)

    # =====================================================================

    def getPointsForCoeffs(self,*coeffList):
        ''' 
            Returns the nodal points structure or None depending on the type
            of the combination of coeffs in coeffList 
            See getTypeForCoeffs above.
        '''

        maxTypes,Types = self.getTypeForCoeffs(True,*coeffList)

        # Uniform and/or radial
        if maxTypes<=2:
          return None
        else:
          return self.nodalPoints

    # =====================================================================

    def getRadius(self):
        ''' 
            Returns radius list from background file 
        '''

        if not hasattr(self,'radius'):
           self.radius = NP.loadtxt(self.fileName,comments='#')[:,0]
        return self.radius

    # =====================================================================

    def getRhoc(self,params,scaleDirac=False,nSrc=0,Grad = False):
      '''
          Returns the product rho*c or rho*c*c(source)
          Used for scaling Green's functions depending on the observable.
 
      '''
      points = params.geom_.getCartesianCoordsMeshGrid()
      # Compute the nodal points if it was not already done for 2D rho or c
      nodalPts = self.nodalPoints
      # if not hasattr(nodalPts, 'points') and not params.unidim_:
        # nodalPts.computePoints()

      rho    = self.rho(points, nodalPoints=nodalPts)
      c      = self.c  (points, nodalPoints=nodalPts)

      if scaleDirac:
        scale = self.getc0(nSrc)
      else:
        scale = 1.e0

      if Grad:
        drho = self.rho.getGradient(points, geom=params.geom_)
        dc   = self.c.getGradient(points, geom=params.geom_)
        return (drho*c + rho*dc)*scale
      else:
        return rho*c*scale

    # =====================================================================

    def getc0(self,nSrc=0):
      '''
          Returns the value of the sound speed at the source location.
      '''

      srcLoc = getSource(self.cfg)[0][nSrc]
      return self.c(points=srcLoc)

    # =====================================================================

    def getrho0(self,nSrc=0):
      '''
          Returns the value of the density at the source location.
      '''

      srcLoc = getSource(self.cfg)[0][nSrc]
      return self.rho(points=srcLoc)

    # =====================================================================

    def getSourceHeight(self,nSrc=0):
      ''' 
          Returns the spherical radius of the source location
      '''

      srcLoc = getSource(self.cfg)[0][nSrc]
      return NP.sqrt(NP.sum(NP.asarray(srcLoc)**2))

    # =====================================================================

    def getAcousticCutOffFrequency(self):

      data = NP.loadtxt(self.fileName,comments='#')
      r    = data[:,0]*RSUN
      dr   = FDM_Compact(r)
      rho  = data[:,2]
      c    = data[:,1]
      drho = data[:,4]

      return sqrt(c*c*sqrt(rho) *dr(-0.5*r*r*drho/sqrt(rho)**3)/(r*r))
      #r         = self.getRadius()
      #dr        = FDM_Compact(r*RSUN)
      #points    = NP.zeros((3,len(r)))
      #points[0] = r

      #rho = self.rho(points)
      #c   = self.c(points)

      #return sqrt(c*c*sqrt(rho) *dr(r*r*dr(1./sqrt(rho)))/(r*r))




