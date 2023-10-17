import inspect, os
import numpy as NP
from scipy.integrate   import simps, cumtrapz
from scipy.interpolate import interp1d as itp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special     import jv
from scipy.optimize import brentq, newton
from astropy.io import fits as pyfits

from ..Common import *
from ..Common import interpGrid as ITG
from ..Parameters import Legendre
class Flow:
    ''' Flow class for backgroundModel '''

    ''' BE CAREFUL : flow is given in CYLINDRICAL coordinates by default,
        but can specify a coordinate system in which you would like 
        the flow vector to be expressed
    '''

    def __init__(self,config=None,nodalPoints=None,flowString=None):
        ''' reads the configfile to setup type and parameters'''
    
        if config is None and flowString is None:
          raise Exception('Flow class needs some input!')

        if config:
          options = config('Flow','CONSTANT 0 0 0').split()
        else:
          options = flowString.split()
        self.flow = [None,None,None]
        self.nodalPointsFile = '' 

        if len(options) == 0:
          raise ValueError('Flow: no parameters found!')

        self.constantSpherical = False    
        if options[0].upper() == 'CONSTANT':
          self.type      = FlowTypes.CONSTANT
          self.typeSpace = SpatialTypes.UNIFORM
          try:
            self.flow = [evalFloat(options[1]),evalFloat(options[2]),evalFloat(options[3])]
          except:
            raise ValueError('Unable to read parameters for CONSTANT Flow.')
          try:
            self.constantSpherical = (options[4].upper() == 'SPHERICAL')
          except:
            pass
    
        elif options[0].upper() == 'DIFFERENTIAL_ROTATION':
          self.type      = FlowTypes.DIFFERENTIAL_ROTATION
          self.typeSpace = SpatialTypes.NODAL
          self.flow[0]   = 0
          self.flow[2]   = 0
          try:
            self.omega0  = evalFloat(options[1])
            self.omega1  = evalFloat(options[2])
            self.omega2  = evalFloat(options[3])
            self.omegas0 = evalFloat(options[4])
          except:
            raise ValueError('Cannot read constant for DIFFERENTIAL_ROTATION flow.')
          try:
            self.rTacho  = evalFloat(options[5])
          except:
            self.rTacho  = 0.7
 
        elif options[0].upper() == 'NODAL_FULL':
          self.type      = FlowTypes.NODAL_FULL
          self.typeSpace = SpatialTypes.NODAL
          try:
            self.filename = [options[1],options[2],options[3]]
          except:
            raise ValueError('Could not read file names for NODAL_FULL Flow')
          try:
            self.factor = evalFloat(options[4])
          except:
            self.factor = 1.e0
    
        elif options[0].upper() == 'NODAL_MERID':
          self.type      = FlowTypes.NODAL_MERID
          self.typeSpace = SpatialTypes.NODAL
          self.flow[1]   = 0
          try:
            self.filename = [options[1],'',options[2]]
          except:
            raise ValueError('Could not read file names for NODAL_MERID Flow')
          try:
            self.factor = evalFloat(options[3])
          except:
            self.factor = 1.e0
          
        elif options[0].upper() == 'NODAL_LONGI':
          self.type      = FlowTypes.NODAL_LONGI
          self.typeSpace = SpatialTypes.NODAL
          self.flow[0]   = 0
          self.flow[2]   = 0
          try:
            self.filename = ['',options[1],'']
          except:
            raise ValueError('Could not read file name for NODAL_LONGI Flow')
          try:
            self.factor = evalFloat(options[2])
          except:
            self.factor = 1.e0
          
        elif options[0].upper() == 'MERIDIONAL_CIRCULATION':
          self.type      = FlowTypes.MERIDIONAL_CIRCULATION
          self.typeSpace = SpatialTypes.NODAL
          self.flow[1]   = 0
          try:
            self.meridOptions = options[1:]
          except:
            raise ValueError('Could not read parameters for MERIDIONAL_CIRCULATION Flow')
   
        elif options[0].upper() == 'SUPERGRANULE':
          self.type      = FlowTypes.SUPERGRANULE
          self.typeSpace = SpatialTypes.NODAL
          self.flow[1]   = 0
          try:
            self.sgOptions = options[1:]
          except:
            raise ValueError('Could not read parameters for SUPERGRANULE Flow')

 
    # ===============================================================
    # Returns the value of the flow
    
    def __call__(self,rho=None,points=None,nodalPoints=None,geom=None,coordsSystem='cylindrical',ReturnStream=False, rotateTheta = 0.):
        return self.getFlow(rho,points,nodalPoints,geom,coordsSystem,ReturnStream,rotateTheta)
    
    def getFlow(self,rho=None,points=None,nodalPoints=None,geom=None,coordsSystem='cylindrical',ReturnStream=False, rotateTheta = 0.):

        # ====================================================
        # Determine points on which to compute the coefficient
        if points is None:
          if nodalPoints is None:
            if geom is not None:
              points = geom.getCartesianCoordsMeshGrid()
          else:
            x,z    = nodalPoints.getCartesianCoords()
            points = NP.array([x,NP.zeros(x.shape),z])

        # Several checks if no points/geom or nodal points were given
        # Return what is stored in the class instance
        if points is None:

          if (self.flow[0] is None) or (self.flow[1] is None) or (self.flow[2] is None):
            raise Exception('\nNo nodalPoints file given for first computation of flow.\n')

          if coordsSystem is not 'cylindrical':
            print("\nImpossible to perform the conversion from cylindrical")
            print("to %s coordinates as no points were given.\n" % systemCoords)
            raise Exception()
          
          return self.flow

        # Check given coordinates

        points = NP.asarray(points)
        if points.shape[0] != 3:
          print("\nGiven points not compatible with flow computation routines:")
          print("Coordinates must be cartesian and with a dimensions (3) or (3,...).\n")
          raise Exception()

        # Obtain flow in cylindrical coordinates
        if nodalPoints is None:
          npfn = None
        else:
          npfn = nodalPoints.fileName

        flow = self.computeFlow(points,npfn,rho,ReturnStream,rotateTheta)
        if ReturnStream:
          return flow
 
        # Coordinates system conversion
        if coordsSystem is 'spherical':
          pointsSph = cartesianToSpherical(points)
          flow      = cylindricalToSphericalVector(flow,pointsSph[1])
        elif coordsSystem is 'cartesian':
          pointsCyl = cartesianToCylindrical(points)
          flow      = cylindricalToCartesianVector(flow,pointsCyl[1])

        return flow
 
    # =============================================================================================================
    # Computation routines 
    # =============================================================================================================
    # Top routine

    def computeFlow(self,points,nodalPointsFile=None,rho=None,ReturnStream=False,rotateTheta=0.):

        ''' 
            Flow computation:
            - If nodalPointsFile is given (as in preparing MJ simulations) the values
              at nodal points are stored in the structure to avoid unecessary further computations
            - Dimensionality of points is (3) or (3,...) 
        '''

        if nodalPointsFile is not None:
          # Check if the flow has already been computed on nodal points
          if hasattr(self,nodalPointsFile) and nodalPointsFile == self.nodalPointsFile:
            return self.flow
          else:
            # We set the new nodal points file and proceed with computation
            self.nodalPointsFile = nodalPointsFile

        if self.type == FlowTypes.CONSTANT:
          res = []
          for i in range(len(self.flow)):
            res.append(self.flow[i] * NP.ones(points[i].shape))
          res = NP.array(res) 
          # res = NP.asarray(self.flow)[:,NP.newaxis,NP.newaxis]*NP.ones(points.shape)
          if self.constantSpherical:
            pointsCyl = cartesianToCylindrical(points)
            res       = sphericalToCylindricalVector(res,pointsCyl[1])
          #res = NP.asarray(self.flow)[:,NP.newaxis,NP.newaxis]*NP.ones(NP.insert(points.shape,0,3))

        # =========================================================================================================
        # FILES

        if self.type == FlowTypes.NODAL_FULL:
          if nodalPointsFile is None:
            raise ('Interpolation from nodal points not implemented')
          for i in range(3):
            self.flow[i] = self.factor*NP.loadtxt(self.filename[i])
          res = self.flow

        if self.type == FlowTypes.NODAL_MERID:
          if nodalPointsFile is None:
            raise ('Interpolation from nodal points not implemented')
          self.flow[0] = self.factor*NP.loadtxt(self.filename[0])
          self.flow[1] = NP.zeros(self.flow[0].shape)
          self.flow[2] = self.factor*NP.loadtxt(self.filename[2])
          res = self.flow

        if self.type == FlowTypes.NODAL_LONGI:
          if nodalPointsFile is None:
            raise ('Interpolation from nodal points not implemented')
          self.flow[1] = self.factor*NP.loadtxt(self.filename[1])
          self.flow[0] = NP.zeros(self.flow[1].shape)
          self.flow[2] = NP.zeros(self.flow[1].shape)
          res = self.flow

        # =========================================================================================================

        if self.type == FlowTypes.DIFFERENTIAL_ROTATION:

          res = self.computeDifferentialRotationGS(points)

        # ==========================================================================================================

        if self.type == FlowTypes.MERIDIONAL_CIRCULATION:

          # Predefined models without options
          if self.meridOptions[0][:2] == 'MC':
            model = self.meridOptions[0].split('+')
            if model[0] == 'MC1':
              res = self.computeMeridionalLiang(points,rho,0.548632,0.7,1.,1,0,180,1, 3, 0.3, ReturnStream=ReturnStream)
            elif model[0] == 'MC2':
              res = self.computeMeridionalLiang(points,rho,0.304565,0.7,1.,2,0,180,1, 1.3, 0.5, ReturnStream=ReturnStream)
            elif model[0] == 'MC3':
              res = self.computeMeridionalLiang(points,rho,0.232803,0.7,1.,3,0,180,1, 1.4, 0.6, ReturnStream=ReturnStream)
            else: 
              raise Exception('Only MC1, MC2 and MC3 meridional flow models are predefined')

            if len(model) > 1:
              # extra inflow cell to take into account the activity bell in the north and south hemisphere
              if model[1] == 'LC1':
                res += self.computeMeridionalLiang(points,rho,0.293332,0.7,1.,1,33,113,2, 3, 0.3, ReturnStream=ReturnStream)
                res += self.computeMeridionalLiang(points,rho,0.293332,0.7,1.,1,67,147,2, 3, 0.3, ReturnStream=ReturnStream)
              elif model[1] == 'LC2':
                res += self.computeMeridionalLiang(points,rho,0.203096,0.8,1.,1,33,113,2, 1.4, 0.6, ReturnStream=ReturnStream)
                res += self.computeMeridionalLiang(points,rho,0.203096,0.8,1.,1,67,147,2, 1.4, 0.6, ReturnStream=ReturnStream)
              elif model[1] == 'LC3':
                res += self.computeMeridionalLiang(points,rho,0.293396,0.7,1.,1,38,108,2, 3., 0.3, ReturnStream=ReturnStream)
                res += self.computeMeridionalLiang(points,rho,0.293396,0.7,1.,1,72,142,2, 3., 0.3, ReturnStream=ReturnStream)
              else: 
                raise Exception('Only LC1, LC2 and LC3 extra meridional flow cells are predefined')

          else:

            # Options that are common to all models
            try:
              rb = evalFloat(self.meridOptions[2])
            except:
              raise Exception('Cannot evaluate lower radius')
            try:
              ru = evalFloat(self.meridOptions[3])
            except:
              raise Exception('Error: Cannot evaluate upper radius')
            try:
              amplitude = evalFloat(self.meridOptions[1])
            except:
              raise Exception('Error, cannot evaluate amplitude of flow')


            # Different flow cells
            if self.meridOptions[0] == 'JOUVE_2008':
              res = self.computeMeridionalJouve(points,rb,ru,amplitude)

            elif self.meridOptions[0] == 'CUSTOM':
              res = self.computeMeridionalCustom(points,rb,ru,amplitude,rho,ReturnStream,rotateTheta)

            elif self.meridOptions[0] == 'ROTH_2008':
              try:
                lMax = evalFloat(self.meridOptions[4])
              except:
                raise Exception('Error, cannot evaluate the maximum degree for the Legendre decomposition of the flow')
              try:
                nbCells = evalFloat(self.meridOptions[5])
              except:
                nbCells = 1
              res = self.computeMeridionalRoth(points,rb,ru,amplitude,lMax,nbCells,rho,ReturnStream)

            elif self.meridOptions[0] == 'LIANG_2018_old':
              try:
                nbCells = evalFloat(self.meridOptions[4])
              except:
                nbCells = 1
              try:
                thetan = evalFloat(self.meridOptions[5])
              except:
                thetan = 0.
              try:
                thetas = evalFloat(self.meridOptions[6])
              except:
                thetas = 180.
              try:
                alpha = evalFloat(self.meridOptions[7])
              except:
                alpha = 1
              try:
                beta1 = evalFloat(self.meridOptions[8])
              except:
                beta1 = 0
              try:
                beta2 = evalFloat(self.meridOptions[9])
              except:
                beta2 = 0
              res = self.computeMeridionalLiangOld(points,rho,amplitude,rb,ru,nbCells,thetan, thetas, alpha, beta1, beta2, ReturnStream)

            elif self.meridOptions[0] == 'LIANG_2018':
              if 'LC' in self.meridOptions:
                indLC =  self.meridOptions.index('LC')
                nbKnots = (indLC-1)/2
              else:
                nbKnots = (len(self.meridOptions)-1)//2
              knots = [] # knot   locations for the spline
              amps  = [] # amplitude of the flow at the knots
              for i in range(nbKnots):
                knots.append(evalFloat(self.meridOptions[1+2*i]))
                amps.append(evalFloat(self.meridOptions[2+2*i]))
              res = self.computeMeridionalLiang(points,rho,knots,amps,0.7,1.,0., 180., 1, ReturnStream=ReturnStream)

              if 'LC' in self.meridOptions:
                # Add the local cells
                fromFile = False                
                try:
                  evalFloat(self.meridOptions[indLC+1])
                except:
                  fromFile = True

                if not fromFile:  
                  # Analytic expression of the LC 
                  thN1 = evalFloat(self.meridOptions[indLC+1])
                  thS1 = evalFloat(self.meridOptions[indLC+2])
                  thN2 = evalFloat(self.meridOptions[indLC+3])
                  thS2 = evalFloat(self.meridOptions[indLC+4])
                  knots = [] # knot   locations for the spline
                  amps  = [] # amplitude of the flow at the knots
                  nbKnots = (len(self.meridOptions)-indLC-5)/2
                  for i in range(nbKnots):
                    knots.append(evalFloat(self.meridOptions[indLC+5+2*i]))
                    amps.append(evalFloat(self.meridOptions[indLC+6+2*i]))

                  res += self.computeMeridionalLiang(points,rho,knots,amps,0.7,1.,thN1, thS1, 2, ReturnStream=ReturnStream)
                  for i in range(nbKnots):
                    amps[i] = -amps[i]
                  res += self.computeMeridionalLiang(points,rho,knots,amps,0.7,1.,thN2, thS2, 2, ReturnStream=ReturnStream)
                else:
                  # Read from file
                  knots = [] # knot   locations for the spline
                  amps  = [] # amplitude of the flow at the knots
                  nbKnots = (len(self.meridOptions)-indLC-2)/2
                  for i in range(nbKnots):
                    knots.append(evalFloat(self.meridOptions[indLC+2+2*i]))
                    amps.append(evalFloat(self.meridOptions[indLC+3+2*i]))
 
                  res += self.computeMeridionalLiang(points,rho,knots,amps,0.7,1.,fileTheta=self.meridOptions[indLC+1],  ReturnStream=ReturnStream)

            else: 
              raise Exception('No other meridional flow types other than JOUVE_2008, ROTH_2008, LIANG_2018, LIANG_2018_old and CUSTOM for now.')

        # ==========================================================================================================

        if self.type == FlowTypes.SUPERGRANULE:

          # Different supergranule models
          if self.sgOptions[0] in ['ROTH','ROTHMODIF','SPLINES','DUVALLLEG','FERRET']:
            res = self.computeSGLegendre(points,rho)
          elif self.sgOptions[0] == 'BASIS':
            res = self.computeSGBasis(points,rho)
          elif self.sgOptions[0] == 'DUVALL': 
            res = self.computeSGDuvall(points,rho,ReturnStream)




        # Store the result if nodalPoints were given
        if nodalPointsFile:
          self.flow = res
        return res


    # =============================================================================================================
    # Specific models
    # =============================================================================================================

    # ==================================================================
    # Rotation profiles
    # ==================================================================

    def computeDifferentialRotationGS(self,points):

        ''' 
            Returns the simple differential rotation profile from Gizon and Solanki 2003.
        '''

        r     = NP.sqrt(NP.sum(points**2,axis=0))  # distance to origin
        varpi = NP.sqrt(points[0]**2+points[1]**2) # distance to z-axis
        with NP.errstate(all='ignore'):
          th  = NP.arccos(points[2]/r)

        u    = NP.zeros(points.shape)
        u[1] = NP.where(r>self.rTacho,self.omega0 + self.omega1*NP.cos(th)**2 + self.omega2*NP.cos(th)**4,\
                                      self.omegas0)

        return u*varpi*RSUN

    # ==================================================================
    # Meridional flow models
    # ==================================================================

    def computeMeridionalJouve(self,points,rb,ru,amplitude):
           
        ''' 
            Meridional flow cell from the model of Jouve et al 2008 
            WARNING : this model is not mass conservative
            as it was derived with a specific density profile
        '''   

        r,th,phi = cartesianToSpherical(points)

        with NP.errstate(all='ignore'):
          fr = NP.where((r>rb)*(r<ru),\
               -2.e0*(1.e0-rb)/(NP.pi*r) * ((r-rb)/(1.e0-rb))**2 *NP.sin(NP.pi*(r-rb)/(1.e0-rb)) *(3.e0*NP.cos(th)**2-1.e0),0.e0)
          ft = NP.where((r>rb)*(r<ru),\
               ((3.e0*r-rb)/(1-rb)* NP.sin(NP.pi*(r-rb)/(1.e0-rb)) + r*NP.pi*(r-rb)/(1.e0-rb)**2*NP.cos(NP.pi*(r-rb)/(1-rb)))\
               * (r-rb)/(NP.pi*r)*NP.sin(2.e0*th),0.e0)

        u = NP.zeros(points.shape)

        u[0] = amplitude*(fr*NP.sin(th) + ft*NP.cos(th))
        u[2] = amplitude*(fr*NP.cos(th) - ft*NP.sin(th))

        return u

    # ==================================================================

    def computeMeridionalCustom(self,points,rb,ru,amplitude,Rho,ReturnStream = False,rotateTheta=0.):
       
        ''' 
            Meridional flow cell described in the paper 
        '''
        if Rho is None:
          raise Exception('Please provide a valid density class instance for computation or meridional flow cell')

        r,th,phi = cartesianToSpherical(points)
        th       = th - rotateTheta

        # Stream function psi = f(r)*g(th) where g(th) = sin(2th)
        # uth is defined such as h(r) = uth(r,pi/4)/g(th)
        # f(r) = 1/r int(h(r)rho*r)
        # h(r) depends upon a parameter which satisfies f(RSUN) = 0


        def h(r,rh):
          vtop = 1.e0/((ru-rh)*(ru-rb))
          a    = vtop/(rb-ru)+ru+rb
          b    = vtop*rb/(rb-ru)+ru*rb
          return (r-rb)*(r-rh)*(r**2-a*r+b)
        
        #--------------------- 
        # Get the parameter RH

        if Rho.spatialType() not in [SpatialTypes.RADIAL,SpatialTypes.UNIFORM]:
          raise Exception('Density must be spherically symmetric to be able to compute the meridional flow cell')

        if hasattr(Rho,'radius'):
          rS   = Rho.radius
          rhoS = Rho.coefBase
        else:
          rS   = NP.loadtxt(Rho.bg_file,comments='#')[:,0]
          rhoS = Rho.coefBase*NP.ones(rS.shape)

        ind_rb = NP.argmin(abs(rS-rb))
        ind_rt = NP.argmin(abs(rS-ru))
        Rb = rS[ind_rb]
        Rt = rS[ind_rt]
        rH   = rS[ind_rb+1:ind_rt]
        # Restrict r and rho
        rS   = rS[ind_rb:ind_rt+1]
        rhoS = rhoS[ind_rb:ind_rt+1]
        # Get RH by dichotomia
        rMin = rS[1]
        rMax = rS[-2] 

        signMin = NP.sign(simps(h(rS,rMin)*rS*rhoS,rS))
        Niter   = 0

        while Niter<200:
          RH       = (rMax+rMin)/2.
          signHalf = NP.sign(simps(h(rS,RH)*rS*rhoS,rS))
          if signHalf == signMin:
            rMin = RH
          else:
            rMax = RH
          Niter += 1

        #--------------------
        # Compute F for ur
        H = itp1d(rS,h(rS,RH),bounds_error=False,fill_value=0.e0)
        F = NP.zeros(len(rS))
        for i in range(len(rS)):
          F[i] = 1/rS[i] * simps(h(rS[:i+1],RH)*rS[:i+1]*rhoS[:i+1],rS[:i+1])
        #plt.figure()
        #plt.plot(rS,F*rS)
        #plt.xlim((0.7,1))
        #plt.figure()
        #plt.plot(rS, -20 * F / (rS*rhoS))
        #plt.xlim((0.7,1))
        #plt.figure()
        #plt.plot(rS, 1. / (rhoS))
        F = itp1d(rS,F,bounds_error=False,fill_value=0.e0)


        if ReturnStream:
          res = amplitude*F(r)*NP.sin(2.e0*th)*RSUN
          return NP.where((r>rb)*(r<ru),res,0.)

        #--------------------
        # Now build the flow cell
        with NP.errstate(all='ignore'):
          rho = Rho(points)        
          fr  = NP.where((r>rb)*(r<ru),-F(r)/(r*rho)*(4.e0-6.e0*NP.sin(th)**2),0.e0)
          ft  = NP.where((r>rb)*(r<ru),H(r)*NP.sin(2.e0*th),0.e0)
        fr = NP.nan_to_num(fr)
        ft = NP.nan_to_num(ft)

        u    = NP.zeros(points.shape)
        u[0] = -amplitude*(fr*NP.sin(th) + ft*NP.cos(th))
        u[2] = -amplitude*(fr*NP.cos(th) - ft*NP.sin(th))

        return u


    def computeMeridionalRoth(self,points,rb,ru,amplitude,lMax,nbCells,Rho,ReturnStream=False):
      '''Compute the meridional flow profile from Roth & Stix 2008. 
      !!! Not optimized for large values of l. It should be rewritten using iter_legendre if one wants to go to high values of l.'''

      # Geometry and density profile
      r,th,phi = cartesianToSpherical(points)
      r1d = NP.linspace(NP.amin(r), NP.amax(r), 1000)
      points1d = NP.zeros((3,len(r1d)))
      points1d[0,:] = r1d
      rho = Rho(points1d)

      # Radial profile of u_s (Eq. 13 in Roth & Stix 2008)
      us   = (-1)**(nbCells)*NP.sin(nbCells * NP.pi * (r1d-rb) / (ru-rb))
      us[r1d>ru] = 0.
      us[r1d<rb] = 0.

      # v_s is proportional to the derivative of r**2*rho*f
      dr = FDM_Compact(r1d)
      #vs = NP.real(dr(r1d**2*rho*us)) / (rho*r1d) 
      vs = NP.real( (2+r1d*dr(NP.log(rho)))*us + r1d*dr(us))
      #vs[r1d==0] = 0.
      stream = r1d*RSUN*rho*us

      # Normalize such that vs(r=1) = amplitude
      #ind1 = NP.argmin(NP.abs(r1d-1.))
      #normalization = vs[ind1] / amplitude
      #us = us / normalization
      #vs = vs / normalization
      #stream = stream / normalization

      # Interpolate us and vs on grid points
      us = NP.interp(r,r1d,us)
      vs = NP.interp(r,r1d,vs)
      stream = NP.interp(r,r1d,stream)


      # Theta part 
      ls  = NP.arange(2,lMax+1)
      leg = Legendre(NP.cos(th),ls,normalized=False)
      ur = NP.zeros(points.shape[1:])
      uth = NP.zeros(points.shape[1:])
      streamPart = NP.zeros(points.shape[1:])
      for s in range(len(ls)):
        Ps = leg(ls[s],NP.cos(th))
        ur  += NP.sqrt((2*ls[s]+1) / (4.*NP.pi)) * Ps # Ps(cos(th))
        if ls[s] != 0:
          dPs = leg(ls[s],NP.cos(th),derivative=1)
          uth += -NP.sqrt((2*ls[s]+1) / (4.*NP.pi)) * NP.sin(th) * dPs / (ls[s]*(ls[s]+1)) # Ps'(cos(th)) + normalization of vs

      if ReturnStream:
        return stream*uth / NP.amax(uth) * amplitude
      else:
        ur  = ur  * us
        uth = uth * vs
        ur  = ur  / NP.amax(uth) * amplitude
        uth = uth / NP.amax(uth) * amplitude

        ur  = ur / NP.amax(uth) * amplitude
        uth = uth / NP.amax(uth) * amplitude

        # Flow in Cartesian coordinates
        u    = NP.zeros(points.shape)
        u[0] = (ur*NP.sin(th) + uth*NP.cos(th))
        u[2] = (ur*NP.cos(th) - uth*NP.sin(th))

        return u

    def computeMeridionalLiangOld(self,points,Rho,amplitude,rb,rt,nbCells=1,thetan = 0, thetas = 180., alpha = 1, beta1 = 0, beta2 = 0, ReturnStream=False):
      '''Compute the meridional flow profile from Liang et al. 2018'''

      # Geometry and density profile
      r,th,phi = cartesianToSpherical(points)
      r1d = NP.linspace(NP.amin(r), NP.amax(r), 2000)
      points1d = NP.zeros((3,len(r1d)))
      points1d[0,:] = r1d
      rho = Rho(points1d)

      # Radial part (Eq. 4 in Liang et al 2018)
      fr   = (-1)**(nbCells+1)*NP.sin(nbCells * NP.pi * (r1d-rb) / (rt-rb))/2.*(1. + NP.tanh(NP.pi * (beta1 * (r1d-rb)/(rt-rb) - beta2)))
      fr[r1d>rt] = 0.
      fr[r1d<rb] = 0.

      # Fr is proportional to the derivative of r**2*rho*f
      dr = FDM_Compact(r1d)
      Fr = NP.real( (2+r1d*dr(NP.log(rho)))*fr + r1d*dr(fr))
      stream = r1d*RSUN*rho*Fr

      # Horizontal part
      th1d = NP.linspace(0., NP.pi, 1000)
      thdeg  = th1d * 180. / NP.pi
      Gtheta = NP.sin(2.*NP.pi*(thdeg-thetan)/(thetas-thetan))*NP.sin(NP.pi*(thdeg-thetan)/(thetas-thetan))**alpha
      Gtheta[thdeg>thetas] = 0.
      Gtheta[thdeg<thetan] = 0.

      dth = FDM_Compact(th1d)
      gtheta = NP.real(dth(NP.sin(th1d)*Gtheta)) / NP.sin(th1d)
      gtheta[th1d==0] = 0.; gtheta[th1d==NP.pi] = 0. # gtheta = 0 for theta = 0 or pi

      # # now we normalize to utilize the Abar in liang 2018
      # uth = Fr[:,NP.newaxis] * Gtheta[NP.newaxis,:]
      # ind1 = NP.argmin(abs(r1d - 0.963))
      # ind2 = NP.argmin(abs(r1d - 0.999))
      # uthmax = NP.max(NP.abs(uth),axis=1)
      # normalization = simps(uthmax[ind1:ind2],x=r1d[ind1:ind2])/simps(NP.ones(ind2-ind1),x=r1d[ind1:ind2])
      # amplitude = amplitude/normalization

      # Interpolate fr and Fr on grid points
      fr = NP.interp(r,r1d,fr)
      Fr = NP.interp(r,r1d,Fr)
      stream = NP.interp(r,r1d,stream)

      # Interpolate gtheta and Gtheta on grid points
      gtheta = NP.interp(th,th1d,gtheta)
      Gtheta = NP.interp(th,th1d,Gtheta)

      ur  = -fr * gtheta
      uth = Fr * Gtheta


      if ReturnStream:
        return stream*Fr * amplitude
      else:
        ur  = ur * amplitude
        uth = uth * amplitude

        # Flow in Cartesian coordinates
        u    = NP.zeros(points.shape)
        u[0] = (ur*NP.sin(th) + uth*NP.cos(th))
        u[2] = (ur*NP.cos(th) - uth*NP.sin(th))

        return u

    def get_F(self, rlist, Flist, r, rb, rt):
      inds = NP.argsort(NP.array(rlist))
      s = InterpolatedUnivariateSpline(NP.array(rlist)[inds],NP.array(Flist)[inds],k=3)
      F = s(r)
      F[(r < rb) | (r > rt)] = 0
      return F


    def computeMeridionalLiang(self,points,Rho,knots,amps,rb, rt,thetan=None,thetas=None,alpha=None, fileTheta = None, ReturnStream=False):
      '''Compute the meridional flow profile from Liang et al. 2018'''

      # Geometry and density profile
      r,th,phi = cartesianToSpherical(points)
      r1d = NP.linspace(NP.amin(r), NP.amax(r), 2000)
      points1d = NP.zeros((3,len(r1d)))
      points1d[0,:] = r1d
      rho = Rho(points1d)

      # Radial part (see Appendix A in Liang et al 2018) 
      if alpha == 1:
        # Main cell 
        F = lambda Fb: self.get_F(knots+[rb], amps+[Fb], r1d, rb, rt)
        find_root = lambda a: simps((r1d*rho*F(a)), x=r1d, even='first')
        Fb_init = 0
        Fb = newton(find_root, Fb_init)
        Fr = self.get_F([rb] + knots,  [Fb] + amps, r1d, rb, rt)
      else:
        # local cell
        F = lambda rb: self.get_F(knots+[rb], amps+[0], r1d, rb, rt)
        find_root = lambda a: simps((r1d*rho*F(a)), x=r1d, even='first')
        rb_init = 0.7
        rb = newton(find_root, rb_init)
        if rb < 0.7 or rb > 1:
          print('Enable to find rb in the convection zone (rb = %s)' % rb)
          abort()
        Fr = self.get_F([rb] + knots,  [0] + amps, r1d, rb, rt)

      #Fr = Fr / rho**(1./8.) / r1d
      fr = cumsimps((r1d*RSUN)*rho*Fr, r1d*RSUN)/((r1d*RSUN)**2 * rho)
      stream = fr*rho*RSUN*r1d#Fr*r1d*RSUN*rho

      # Horizontal part
      if fileTheta is None:
        # Analytic expression for G(theta) from Liang et al. 2018 
        th1d = NP.linspace(0., NP.pi, 1000)
        thdeg  = th1d * 180. / NP.pi
        Gtheta = NP.sin(2.*NP.pi*(thdeg-thetan)/(thetas-thetan))*NP.sin(NP.pi*(thdeg-thetan)/(thetas-thetan))**alpha
        Gtheta[thdeg>thetas] = 0.
        Gtheta[thdeg<thetan] = 0.
      else:
        # Read file for Gth
        fileG = NP.loadtxt(fileTheta)
        th1d  = fileG[:,0]
        Gtheta = fileG[:,1]

      dth = FDM_Compact(th1d)
      gtheta = NP.real(dth(NP.sin(th1d)*Gtheta)) / NP.sin(th1d)
      gtheta[th1d==0] = 0.; gtheta[th1d==NP.pi] = 0. # gtheta = 0 for theta = 0 or pi

      # Interpolate fr and Fr on grid points
      fr = NP.interp(r,r1d,fr)
      Fr = NP.interp(r,r1d,Fr)
      stream = NP.interp(r,r1d,stream)

      # Interpolate gtheta and Gtheta on grid points
      gtheta = NP.interp(th,th1d,gtheta)
      Gtheta = NP.interp(th,th1d,Gtheta)

      ur  = fr * gtheta
      uth = -Fr * Gtheta

      if fileTheta is None:
        amplitude = amps[-1] / NP.amax(NP.abs(uth))
      else:
        amplitude = 1. / NP.amax(NP.abs(Fr)) # The amplitude is contained in the file

      if ReturnStream:
        return stream*Gtheta*amplitude
      else:
        ur  = ur * amplitude
        uth = uth * amplitude
        # Flow in Cartesian coordinates
        u    = NP.zeros(points.shape)
        u[0] = (ur*NP.sin(th) + uth*NP.cos(th))
        u[2] = (ur*NP.cos(th) - uth*NP.sin(th))

        return u


    # ==================================================================
    # Supergranules 
    # ==================================================================

    def computeSGDuvall(self,points,rho,ReturnStream=False):
      ''' Supegranule model from Tom Duvall
          r0     : depth location of vertical flow peak (in Mm)
          sig    : Width of Gaussian (in Mm)
          U      : Amplitude of normalized flow (m/s)
          size   : Size of SG in Mm
          theta0 : Angle of dissipation (in Mm)
      '''
      if rho is None:
        raise Exception('Please provide a valid density class instance for computation or meridional flow cell')
      if hasattr(rho,'radius'):
        rS   = rho.radius
      else:
        rS   = NP.loadtxt(rho.bg_file,comments='#')[:,0]

      # Read parameters, NOTE SCALING BY RSUN IS REMOVED HERE
      try:
        r0     = 1.e0 - evalFloat(self.sgOptions[1])*1.e6/RSUN
        sig    = evalFloat(self.sgOptions[2])*1.e6/RSUN
        U      = evalFloat(self.sgOptions[3])
        size   = evalFloat(self.sgOptions[4])
        theta0 = NP.arcsin(evalFloat(self.sgOptions[5])*1e6/RSUN)
      except:
        raise Exception('5 arguments are required for SUPERGRANULE DUVALL. r0,sigma,U,size,theta0.')

      try:
        normalize = (self.sgOptions[6].upper() == 'YES')
      except:
        normalize = False

      k = 2.e0*NP.pi/(size*1.e6)*RSUN # wave number of SG
      if not normalize:
        U /= k

      # Determine the appropriate mesh size (i.e when the supergranule is 1e-6 of it's maximum)
      ValueMin = 1.e-6
      # r component
      rG = NP.linspace(0.9e0,rS[-1],2000)
      u_r = NP.exp(-(rG-r0)**2/(2*sig**2)) - ValueMin
      rG_indz = NP.where(NP.sign(u_r[:-1])!=NP.sign(u_r[1:]))[0]
      rG1 = rG[rG_indz[0]]
      if len(rG_indz) > 1:
        rG2 = rG[rG_indz[1]]
      else:
        rG2 = rS[-1]
      # theta component
      thG  = NP.linspace(1.e-10  ,0.25   ,1000)
      g_th = NP.exp(-thG**2/theta0**2)
      thmax = thG[NP.argmin(NP.abs(g_th-ValueMin))]

      # Mesh the supergranule region 
      rG   = NP.linspace(rG1,rG2,1000)
      thG  = NP.linspace(0  ,thmax   ,1000)
      iR1  = NP.argmin(abs(rG-1.))
      rhoG = NP.interp(rG,rS,rho())
      drhoG = NP.interp(rG,rS,rho.getGradient()[0]*RSUN)
      
      # Velocity components of the single granule (cf /Dropbox/Computational Helioseismology/RESULTS/Inversion/SG_INVERSION_KERNEL/DH_SG.pdf)
      # Define the FDM class
      dth = FDM_Compact(thG)
      dr   = FDM_Compact(rG)

      # Define u(r) and g(theta) from duvall and hanasoge (note changes see zelia notes)
      u_r  = NP.exp(-(rG-r0)**2/(2*sig**2))/rG
      g_th = jv(1,k*NP.sin(thG)) * NP.exp(-thG**2/theta0**2)

      # Compute derivative terms 1/(rho*r)*dr(rho*r*u(r)), 1/rsinth*dth(g(th)*sin(theta))
      # and asymptotic terms for theta --> 0
      f_r  = NP.real((rhoG*dr(rG**2*u_r)+drhoG*rG**2*u_r)/(rG*rhoG))
      j_th     = NP.zeros(thG.shape)
      j_th[1:] = NP.real(1./(NP.sin(thG[1:]))*dth(g_th*NP.sin(thG))[1:])
      j_th[0]  = NP.real(dth(g_th)[0]) + k/2


      if normalize:
        norm = 1.e0/f_r[iR1]
      else:
        norm = 1.
      u_r   = u_r * norm
      f_r   = f_r * norm

      ur  =  U*u_r[:,NP.newaxis]*j_th[NP.newaxis,:]
      uth = -U*f_r[:,NP.newaxis]*g_th[NP.newaxis,:]

      if 'VERTICAL' in self.sgOptions:
        # Keep only the vertical part of the flow
        uth = NP.zeros(uth.shape)
      elif 'HORIZONTAL' in self.sgOptions:
        # Keep only the horizontal part of the flow
        ur = NP.zeros(ur.shape)

      if ReturnStream:
        Npts     = NP.product(points.shape[1:])
        points2  = points.reshape((3,Npts))

        rN  = NP.sqrt(NP.sum(points2**2,axis=0))
        with NP.errstate(all='ignore'):
          thN = NP.arccos(points2[2]/rN)
        thN = NP.nan_to_num(thN)


        # Interpolation on given points
        itp = ITG.interpGrid((rG,thG),method='linear',\
                       fillOutside=True,fillWithZeros=True)
        itp.setNewCoords((rN,thN))
        rG = rG*RSUN
        Aphi = -(1/(rG[:,NP.newaxis]))*cumtrapz(rhoG[:,NP.newaxis]*uth*rG[:,NP.newaxis],x=rG,axis=0,initial=0)

        # Aphi = 1/(NP.sin(thG[NP.newaxis,:]))*cumtrapz(rhoG[:,NP.newaxis]*ur*rG[:,NP.newaxis]*NP.sin(thG[NP.newaxis,:]),x=thG,axis=1,initial=0)
        Aphi = itp(Aphi )
        
        return Aphi.reshape(points[0].shape)

      else:
        # Now compute the flow on requested points using interpolation
        return self.interpSGOnPoints((rG,thG),points,ur,uth)

    # ==================================================================
    def computeSGBasis(self,points,Rho):
      ''' Supergranules deduced from fitted surface data approximated onto a Gaussian basis'''
      try:
        if self.sgOptions[1] == 'CUSTOM':
          amp      = float(self.sgOptions[2])
          rLCT     = 1.+float(self.sgOptions[3])/696000.
          siLCT    = float(self.sgOptions[4])/(696000*2*NP.sqrt(2*NP.log(2)))
          rD       = 1.+float(self.sgOptions[5])/696000.
          siD      = float(self.sgOptions[6])/(696000*2*NP.sqrt(2*NP.log(2)))
          alpha    = float(self.sgOptions[7])
          lmax     = int(self.sgOptions[8])
          lda      = float(self.sgOptions[9])
          r0       = 1.+float(self.sgOptions[10])/696.
          sig      = float(self.sgOptions[11])/696.
          steprk   = float(self.sgOptions[12])
        elif self.sgOptions[1] == 'CLASSIC':
          amp      = float(self.sgOptions[2])
          Par      = NP.load(pathToMPS()+'/data/background/paramsWeF.npy')
          rLCT     = 1.-100./696000.
          siLCT    = 100./(696000*2*NP.sqrt(2*NP.log(2)))
          rD       = 1.+Par[1]/696000
          siD      = Par[3]/696000.
          alpha    = -100.
          lmax     = 360
          lda      = 45
          r0       = 1+float(self.sgOptions[3])/696.
          sig      = float(self.sgOptions[4])/696.
          steprk   = float(self.sgOptions[5])
      except:
        raise Exception('Unable to read parameters for SUPERGRANULE. amp,rLCT,siLCT,rD,siD,lmax,lambda,r0,sig.')
        
      # Coordinates in supergranule region
      step = 0.025/696.
      hwhm = sig*NP.sqrt(2*NP.log(2))
      rG   = NP.arange(r0-3*hwhm,(r0+3*hwhm)+step,step); hG  = 696*(rG-1)
      thG  = NP.linspace(0,0.2,200)
      dr   = FDM_Compact(rG)
      # Density
      ModelS = NP.genfromtxt(pathToMPS()+'data/background/modelS_SI_HANSON.txt')
      radius = ModelS[:,0];dens = ModelS[:,2]
      rho    = NP.interp(rG,radius,dens)
      # Weighting functions
      Wd     = np.exp(-(rG-rD)**2/(2*siD**2))/np.sqrt(2*NP.pi*siD**2)
      Wh     = (np.exp(-(rG-rLCT)**2/(2*siLCT**2))/np.sqrt(2*NP.pi*siLCT**2))*(1+scipy.special.erf(alpha*hG/NP.sqrt(2)))/scipy.integrate.simps((np.exp(-(rG-rLCT)**2/(2*siLCT**2))/np.sqrt(2*NP.pi*siLCT**2))*(1+scipy.special.erf(alpha*hG/NP.sqrt(2))),rG)
      # Read fitted Legendre coefficients
      coef = NP.genfromtxt(pathToMPS()+'/data/background/amplitudes.txt')
      cR = coef[:,0];cTh = coef[:,1]      
      # Legendre polynomials
      with NP.errstate(divide='ignore',invalid='ignore'):
        P,dP = legendreArray(651,thG,True)
      ell    = NP.arange(1,lmax+1); L = NP.sqrt(ell*(ell+1))   
      steprk = steprk/696.
      knots  = NP.arange(min(rG),max(rG)+steprk/2.,steprk)
      phif   = self.quadSplines(rG,knots)                
      B      = NP.real(self.solveQuadratic(rG,lda,r0,sig,lmax,phif,Wd,Wh))

      ### VELOCITIES
      fell   = NP.sum(B[:,:,NP.newaxis]*phif,axis=1)
      hell   = NP.real(dr(fell*rho*rG**2)/rho*rG)
      radr   = cTh[:lmax,NP.newaxis]*fell*L[:,NP.newaxis]
      radth  = cTh[:lmax,NP.newaxis]*hell
      ur     = amp*NP.sum(P[1:lmax+1,:]*NP.transpose(radr[NP.newaxis,:]),axis=1)
      uth    = amp*NP.sum(dP[1:lmax+1,:]*NP.transpose(radth[NP.newaxis,:]),axis=1)
      return self.interpSGOnPoints((rG,thG),points,ur,uth)
    # ==================================================================   
    def quadSplines(self,t,knots):
      N0    = np.zeros((len(knots)-1,len(t)))
      N1    = np.zeros((len(knots)-2,len(t)))
      N2    = np.zeros((len(knots)-3,len(t)))

      for i in range(0,len(knots)-1):
        for j in range(0,len(t)):
          if t[j] < knots[i+1] and knots[i] <= t[j]:
            N0[i,j] = 1. 

      for i in range(0,len(knots)-2):    
        N1[i,:] = (t-knots[i])*N0[i,:]/(knots[i+1]-knots[i])+(knots[i+2]-t)*N0[i+1,:]/(knots[i+2]-knots[i+1])
  
      for i in range(0,len(knots)-3):    
        N2[i,:] = (t-knots[i])*N1[i,:]/(knots[i+2]-knots[i])+(knots[i+3]-t)*N1[i+1,:]/(knots[i+3]-knots[i+1])
    
      return N2  
    # ==================================================================        
    def solveQuadratic(self,rG,l1,r0,sig,lmax,phif,Wd,Wh):
	### GEOMETRY
        D      = self.getDataNew()
        th     = D[0,:]; d = 696*th
        vth    = D[1,:]
        sth    = D[2,:]
        vr     = D[3,:]
        sr     = D[4,:]
        dr     = FDM_Compact(rG)
        h      = 696*(rG-1)
        ell    = NP.arange(1,lmax+1)
        L      = NP.sqrt(ell*(ell+1))
        with NP.errstate(divide='ignore',invalid='ignore'):
            P,dP   = legendreArray(651,th,True)
        coef = NP.genfromtxt(pathToMPS()+'/data/background/amplitudes.txt')
        cr = coef[:,0];cth = coef[:,1]      

        ### BACKGROUND
        ModelS = NP.genfromtxt(pathToMPS()+'data/background/modelS_SI_HANSON.txt')
        rad    = ModelS[1:,0];rho = ModelS[1:,2]
        rho    = np.interp(rG,rad,rho)

	### BASIS
        N      = NP.shape(phif)[0]
        phih   = dr(rho*rG**2*phif)/(rho*rG)

	### MODEL TO APPROXIMATE
        fL     = NP.exp(-(rG-r0)**2/(2*sig**2))
        hL     = 1.e0/(rho*rG)*dr(rho*rG**2*fL)
        A      = 1./(scipy.integrate.simps(Wh*hL,rG))
        fmod   = A*fL
        hmod   = A*hL
        urmod  = NP.sum(P[1:lmax+1,:]*L[:,NP.newaxis]*cth[:lmax,NP.newaxis]*fmod[:,NP.newaxis,NP.newaxis],axis=1)
        uthmod = NP.sum(dP[1:lmax+1,:]*cth[:lmax,NP.newaxis]*hmod[:,NP.newaxis,NP.newaxis],axis=1)

        B      = []
        for n in range(0,lmax):
            Q0     = np.zeros((N,N))
            Q1     = np.zeros(np.shape(Q0))
            for i in range(0,N):
                Q0[i,:]  = L[n]**2*cth[n]**2*scipy.integrate.simps(phif[i,:]*phif*rG**2,x=rG)+cth[n]**2*scipy.integrate.simps(phih[i,:]*phih*rG**2,x=rG)
                Q1[i,:]  = L[n]**2*cth[n]**2*scipy.integrate.simps(phif[i,:]*Wd,x=rG)*scipy.integrate.simps(phif*Wd,x=rG)+cth[n]**2*scipy.integrate.simps(phih[i,:]*Wh,x=rG)*scipy.integrate.simps(phih*Wh,x=rG)

                ### RIGHT HAND SIDE
                F0     = L[n]**2*cth[n]**2*scipy.integrate.simps(phif*fmod*rG**2,x=rG)+cth[n]**2*scipy.integrate.simps(phih*hmod*rG**2,x=rG)
                F1     = cr[n]*L[n]*cth[n]*scipy.integrate.simps(phif*Wd,x=rG)+cth[n]**2*scipy.integrate.simps(phih*Wh,x=rG)
	   
                ### SOLVING
                bkn    = np.dot(np.linalg.inv(Q0+l1*Q1),F0+l1*F1) 
                B.append(bkn)

                ### CONTROL
                B      = NP.asarray(B)
        return B
    # ==================================================================   
    def getDataNew(self,sampling=True,padding=True):
      ### FILES
      Hor    = np.loadtxt(pathToMPS()+'data/background/horizontalvelocity120Mm.txt')
      Rad    = np.loadtxt(pathToMPS()+'data/background/verticalvelocity120Mm.txt')
      Ns     = np.loadtxt(pathToMPS()+'data/background/numbersgreal.txt')
      Np     = np.loadtxt(pathToMPS()+'data/background/numberpoints.txt')

      ### DEFINITIONS
      d      = Rad[:,0]   # Distance from center of the SG
      vtheta = Hor[:,1]   # Horizontal velocity
      vr     = Rad[:,1]   # Radial velocity
      theta  = d/696      # Angular distance from center of SG
      sigth  = Hor[:,2]   # Smoothed error estimate for vth
      sigr   = Rad[:,2]   # Smoothed error estimate for vr
      varth  = Hor[:,3]   # Error on the smoothed error estimate for vth
      varr   = Rad[:,3]   # Error on the smoothed error estimate for vr
    
      ### INDICES AND VARIABLE
      kr     = np.argmin(abs(d-20))+1
      n      = NP.mean(Np)/NP.mean(d)
      x      = (n*d+1)*Ns
  
      ### POLYNOMIAL COEFFS
      pr     = np.polyfit(np.log(x[0:kr]),np.log(sigr[0:kr]),1)
      pth    = np.polyfit(1./np.sqrt(x[1:kr]),sigth[1:kr],4)
      ar     = np.exp(pr[1])*x[kr]**(pr[0]+0.5)
      ath    = np.polyval(pth,1./np.sqrt(x[kr]))*NP.sqrt(x[kr])
  
      ### PADDING
      step   = theta[1]-theta[0]
      ext    = np.arange((max(theta)+step),np.pi,step)
      theta  = np.concatenate([theta,ext])
      D      = 696*theta; N      = np.size(vtheta); K      = np.size(theta); M      = np.size(vr)
      vthadd = np.zeros(K-N)
      vradd  = np.zeros(K-M)
      vtheta = np.concatenate([vtheta,vthadd])
      vr     = np.concatenate([vr,vradd])
      Nsadd  = Ns[-1]*np.ones(len(ext))
      Ns     = np.concatenate([Ns,Nsadd])
      xx     = (n*D+1)*Ns
      lefr   = np.exp(pr[1])*xx[0:kr]**pr[0]
      rigr   = ar/np.sqrt(xx[kr:])
      lefth  = np.polyval(pth,1./np.sqrt(xx[0:kr]))
      rigth  = ath/np.sqrt(xx[kr:])
      sigr   = np.concatenate((lefr,rigr))
      sigth  = np.concatenate((lefth,rigth))
  
      ### SAMPLING
      ind    = range(0,len(theta)+1,9)
      vr     = vr[ind]
      theta  = theta[ind]
      vtheta = vtheta[ind]
      sigr   = sigr[ind]
      sigth  = sigth[ind]
    
      ### WRITING
      D      = np.zeros((5,len(theta)))
      D[0,:] = theta
      D[1,:] = vtheta
      D[2,:] = sigth 
      D[3,:] = vr
      D[4,:] = sigr
      return D 
    # ==================================================================   
    def computeSGLegendre(self,points,Rho):
      ''' Supergranules deduced from fitted surface data '''
      try:
        amp      = float(self.sgOptions[1])
        rLCT     = float(self.sgOptions[2])
        siLCT    = float(self.sgOptions[3])
        rD       = float(self.sgOptions[4])
        siD      = float(self.sgOptions[5])
        lmax     = int(self.sgOptions[6])
        if self.sgOptions[0] == 'ROTH' or self.sgOptions[0] == 'ROTHMOD' or self.sgOptions[0] == 'FERRET':
          rB       = float(self.sgOptions[7])
          rT       = float(self.sgOptions[8])
        if self.sgOptions[0] == 'DUVALLLEG':
          r0       = float(self.sgOptions[7])
          sig      = float(self.sgOptions[8])
        elif self.sgOptions[0] == 'SPLINES':
          rB       = float(self.sgOptions[7])
          rT       = float(self.sgOptions[8])
          r0       = float(self.sgOptions[9])
      except:
        raise Exception('Unable to read parameters for SUPERGRANULE. amp,rLCT,siLCT,rD,siD,lmax,rB,rT,r0,sig.')
        
      # Normalization of parameters
      rB    = -(rB-696.)/696.
      rT    = (rT+696000.)/696000.
      rD    = 1+rD/696000.
      rLCT  = 1+rLCT/696000.
      siD   = siD/696000.
      siLCT = siLCT/696000.
      if self.sgOptions[0] == 'SPLINES':
        r0=-(r0-696.)/696.
      elif self.sgOptions[0] == 'DUVALL':
        r0=-(r0-696.)/696.
        sig=sig/696000.
        
      # Coordinates in supergranule region
      rG   = NP.linspace(0.95,1.01,500)
      thG  = NP.linspace(0,1,500)
      dr   = FDM_Compact(rG)
  
      # Density
      ModelS = NP.genfromtxt(pathToMPS()+'data/background/modelS_SI_HANSON.txt')
      radius = ModelS[:,0];dens = ModelS[:,2]
      rho    = NP.interp(rG,radius,dens)
      
      # Weighting functions
      Wd     = NP.exp(-(rG-rD)**2/(2*siD**2))/np.sqrt(2*pi*siD**2)
      Wlct   = (np.exp(-(rG-rLCT)**2/(2*siLCT**2))/np.sqrt(2*pi*siLCT**2))
      
      # Read fitted Legendre coefficients
      coef = NP.genfromtxt(pathToMPS()+'/data/background/amplitudes.txt')
      cR = coef[:,0];cTh = coef[:,1]
      
      # Legendre polynomials
      with NP.errstate(divide='ignore',invalid='ignore'):
        [P,dP] = legendreArray(651,thG,True)
      ell    = NP.arange(1,lmax+1); L = NP.sqrt(ell*(ell+1))   
      # Radial function
      if self.sgOptions[0] == 'FERRET':
        [fL,hL] = self.radialFerret(rG,rB,rT,cR,cTh,rD,siD,rLCT,siLCT,lmax)
      else:
        [fL,hL] = self.radialFunction(rG,rB,rT,r0,sig,typef=self.sgOptions[0])
        A       = self.getNormalization(rD,siD,rLCT,siLCT,rB,rT,r0,sig,typef=self.sgOptions[0])
      ### FULL VELOCITY
      if self.sgOptions[0] == 'FERRET':
        radr  = cTh[:lmax,NP.newaxis]*L[:,NP.newaxis]*fL
        radth = cTh[:lmax,NP.newaxis]*hL
        ur    = NP.sum(P[1:lmax+1,:]*NP.transpose(radr[NP.newaxis,:]),axis=1)
        uth   = NP.sum(dP[1:lmax+1,:]*NP.transpose(radth[NP.newaxis,:]),axis=1)
      else:
        ur  = NP.transpose(NP.sum(cTh[:lmax]*L*fL[:,NP.newaxis]*NP.transpose(P[1:lmax+1,NP.newaxis]),axis=2))
        uth = NP.transpose(NP.sum(cTh[:lmax]*hL[:,NP.newaxis]*NP.transpose(dP[1:lmax+1,NP.newaxis]),axis=2))          
          
      ur  = amp*ur
      uth = amp*uth
      return self.interpSGOnPoints((rG,thG),points,ur,uth)
    # ==================================================================      
    def getMonomials(self,rb,rt,lmax,cr,cth,Rd,sid,Rlct,silct):
      ### DENSITY PROFILE
      MS   = NP.genfromtxt('/home/ferret/Documents/mps_montjoie/data/background/modelS_SI_HANSON.txt')
      dens = MS[1:,2]; rad = MS[1:,0]
      r    = NP.linspace(rb,rt,1000)
      rho  = NP.interp(r,rad,dens)
      ### WEIGHTING FUNCTIONS
      Wd   = np.exp(-(r-Rd)**2/(2*sid**2))/np.sqrt(2*pi*sid**2)
      Wlct = (np.exp(-(r-Rlct)**2/(2*silct**2))/np.sqrt(2*pi*silct**2))
      ### DERIVATIVE
      dr   = FDM_Compact(r)
      ### FACTORS
      ell  = NP.arange(1,lmax+1)
      L    = NP.sqrt(ell*(ell+1)) 
      ### COMPONENTS
      W0   = scipy.integrate.simps(Wd*(r-rb)*(r-rt),r)
      W1   = scipy.integrate.simps(Wd*(r-rb)*(r-rt)*r,r)
      D3   = dr(rho*r**3*(r-rb)*(r-rt))
      D2   = dr(rho*r**2*(r-rb)*(r-rt))
      I3   = NP.real(scipy.integrate.simps(Wlct*D3/(rho*r),r))
      I2   = NP.real(scipy.integrate.simps(Wlct*D2/(rho*r),r))
      ### MONOMIALS
      al   = (L*cth[:lmax]/cr[:lmax]-I2/W0)/(I3-W1*I2/W0)
      bl   = (1-al*W1)/W0
      return al,bl   
    # ==================================================================
    def radialFunction(self,rG,rB=0,rT=0,r0=0,sig=0,typef='ROTH'):
      # Density
      ModelS  = NP.genfromtxt(pathToMPS()+'data/background/modelS_SI_HANSON.txt')
      radius  = ModelS[:,0];dens = ModelS[:,2]
      rho     = NP.interp(rG,radius,dens)
      # Coordinates in supergranule region
      dr   = FDM_Compact(rG)
      if typef == 'ROTH':
        fL = NP.sin(NP.pi*(rG-rB)/(rT-rB))
      elif typef == 'ROTHMOD':
        fL = NP.sin(NP.pi*(rG-rB)/(rT-rB))/rho
      elif typef == 'SPLINES':
        a  = (-2*r0+rB+rT)/2
        b  = (4*r0-rT-3*rB)/2
        fL = (-b+NP.sqrt(b**2-4*a*(rB-rG)))/(a)-((-b+NP.sqrt(b**2-4*a*(rB-rG)))/(2*a))**2
      elif typef == 'DUVALL':
        fL = NP.exp(-(rG-r0)**2/(2*sig**2))
      hL = 1.e0/(rho*rG)*dr(rho*rG**2*fL)
      hL = NP.real(hL)
      if typef != 'DUVALL':
        ninf       = where(rG < rB)[0]
        nsup       = where(rG > rT)[0]
        fL[ninf] = 0; fL[nsup] = 0
        hL[ninf] = 0; hL[nsup] = 0
      return fL,hL
    # ==================================================================      
    def radialFerret(self,rG,rB,rT,cR,cTh,Rd,sid,Rlct,silct,lmax):  
      ### BACKGROUND
      MS    = NP.genfromtxt(pathToMPS()+'data/background/modelS_SI_HANSON.txt')
      rad   = MS[:,0];dens = MS[:,2]
      rho   = NP.interp(rG,rad,dens)
      dr    = FDM_Compact(rG)
      D3    = dr(rho*rG**3*(rG-rB)*(rG-rT))
      D2    = dr(rho*rG**2*(rG-rB)*(rG-rT))
      ### MONOMIALS
      aL,bL = self.getMonomials(rB,rT,lmax,cR,cTh,Rd,sid,Rlct,silct)
      ell   = NP.arange(1,lmax+1,1)
      L     = NP.sqrt(ell*(ell+1))
      ### RADIAL FUNCTIONS
      fL    = cR[:lmax,NP.newaxis]*(rG-rB)*(rG-rT)*(aL[:,NP.newaxis]*rG+bL[:,NP.newaxis])/(L[:,NP.newaxis]*cTh[:lmax,NP.newaxis])
      hL    = cR[:lmax,NP.newaxis]*(aL[:,NP.newaxis]*D3+bL[:,NP.newaxis]*D2)/(L[:,NP.newaxis]*cTh[:lmax,NP.newaxis]*rho*rG)
      ### MAKE NULL
      ninf       = NP.where(rG < rB)[0]
      nsup       = NP.where(rG > rT)[0]
      fL[:,ninf] = 0; fL[:,nsup] = 0
      hL[:,ninf] = 0; hL[:,nsup] = 0
      return NP.real(fL),NP.real(hL)
    # ==================================================================      
    def getNormalization(self,Rd,sid,Rlct,silct,rb=0,rt=0,r0=0,sig=0,typef='ROTH',fit='HORIZONTAL'):
      ### BACKGROUND MODEL
      MS   = NP.genfromtxt(pathToMPS()+'data/background/modelS_SI_HANSON.txt')
      rad  = MS[1:,0];dens = MS[1:,2]
      if typef != 'DUVALL':
        r   = NP.linspace(rb,rt,1000)
        rho = NP.interp(r,rad,dens)
      else:
        r   = rad
        rho = dens
      ### WEIGHTING FUNCTIONS
      Wd    = np.exp(-(r-Rd)**2/(2*sid**2))/np.sqrt(2*pi*sid**2)
      Wh    = (np.exp(-(r-Rlct)**2/(2*silct**2))/np.sqrt(2*pi*silct**2))
      ### RADIAL FUNCTIONS  
      ninf = where(r<rb);nsup = where(r>rt)
      if typef == 'ROTH':
        fL = NP.sin(NP.pi*(r-rb)/(rt-rb))
      elif typef == 'ROTHMOD':
        fL = NP.sin(NP.pi*(r-rb)/(rt-rb))/rho
      elif typef == 'SPLINES':
        a  = (-2*r0+rb+rt)/2
        b  = (4*r0-rt-3*rb)/2
        fL = (-b+NP.sqrt(b**2-4*a*(rb-r)))/(a)-((-b+NP.sqrt(b**2-4*a*(rb-r)))/(2*a))**2
      elif typef == 'DUVALL':
        fL = NP.exp(-(r-r0)**2/(2*sig**2))
      dr     = FDM_Compact(r)
      hL     = 1.e0/(rho*r)*dr(rho*r**2*fL)
      hL     = NP.real(hL)
      fL[ninf] = 0; fL[nsup] = 0;
      hL[ninf] = 0; hL[nsup] = 0;
      if fit=='HORIZONTAL':
        A      = 1./(scipy.integrate.simps(Wh*hL,r))
      elif fit=='RADIAL':
        A      = 1./(scipy.integrate.simps(Wd*fL,r))
      return A
    # ==================================================================      
    def interpSGOnPoints(self,coordsG,points,ur,uth):
      ''' Interpolates values of the flow computed on SG points
          to the given points '''

      # Reorder points to be (N,3)
      Npts     = NP.product(points.shape[1:])
      points2  = points.reshape((3,Npts))

      rN  = NP.sqrt(NP.sum(points2**2,axis=0))
      with NP.errstate(all='ignore'):
        thN = NP.arccos(points2[2]/rN)
      thN = NP.nan_to_num(thN)


      # Interpolation on given points
      itp = ITG.interpGrid(coordsG,method='linear',\
                       fillOutside=True,fillWithZeros=True)
      itp.setNewCoords((rN,thN))

      # Get values in cartesian coordinates
      Mr  = itp(ur )
      Mth = itp(uth)

      u = NP.zeros(points2.shape)

      u[0] = Mr*NP.sin(thN)+Mth*NP.cos(thN)
      u[2] = Mr*NP.cos(thN)-Mth*NP.sin(thN)

      return u.reshape(points.shape)


    @staticmethod
    def writeReverseFlow(config):
      '''Reads the line containing the flow in the config and returns a new line with the reversed flow.'''
      oldFlow = config('Flow','CONSTANT 0 0 0').split()
      if oldFlow[0] == 'CONSTANT':
        newFlow = 'CONSTANT %s %s %s' % (oldFlow[1], oldFlow[2], oldFlow[3])
      elif oldFlow[0] in ['MERIDIONAL_CIRCULATION', 'DIFFERENTIAL_ROTATION','SUPERGRANULE']: 
        amp = - evalFloat(oldFlow[2])
        newFlow = '%s %s %s %s' % (oldFlow[0], oldFlow[1], amp, ' '.join(oldFlow[3:]))
      else:
        raise NotImplementedError('The writing of the reversed flow for this type of flow (%s) is not implemented' % oldFlow[0])

      return newFlow
      

    # =============================================================================================================

class FlowTypes:

  CONSTANT = 0
  MERIDIONAL_CIRCULATION = 1
  DIFFERENTIAL_ROTATION = 2
  SUPERGRANULE = 3
  NODAL_FULL = 6
  NODAL_MERID = 7
  NODAL_LONGI = 8
