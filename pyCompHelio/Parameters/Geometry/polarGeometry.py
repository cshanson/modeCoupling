import numpy    as     NP
from   .geometry import *

class polarGeom(geom):

    def Nr(self):
      return self.N_[0]

    def Ntheta(self):
      return self.N_[1]

    def r(self):
      return self.coords_[0]

    def theta(self):
      return self.coords_[1]

    def axisTheta(self):
      return 1

    def plotOnSlice(self,data,point=[0.0,0.0,0.0],normal=[0.0,1.0,0.0],\
                    fileName=None,rPlot=None,thetaPlot=None,\
                    Nr=None,Ntheta=None,rSample=1,thetaSample=1,\
                    **pythonPlotArgs):

      ''' Plots data on a slice of the sphere, cut by a plane
          for which we give a point and the normal.
          
          Points were data is plotted can be given in each dimension by either
          rPlot   : vector of radiuses
          Nr      : linspace from 0 to max(radius), (2pi for theta)
          rSample : original geometry vector (self.r_) sample by rSample
      '''

      #---------------
      # 0.Check data
      if self.checkPlotData(data) == 1:
        return

      #--------------------------------------------------------
      # 1.Generate list of points before rotation (on a disk)

      normal = normal/norm2(normal)
      # If the slice does not pass through the center of the sphere
      # the disk radius is smaller!
      # Equation of the plane is n(0)(x-p(0)) + n(1)(y-p(1)) + n(2)(z-p(2)) = 0
      # Distance to the plane from the center of the sphere is n(0)p(0)+n(1)p(1)+n(2)p(2)
      # If p is 0, we have rmax = max(self.r())
      rmax = NP.sqrt(self.coords_[0][-1]**2 - (dotProd(normal,point))**2)

      if type(Nr) is int:
        rPlot = NP.linspace(0.0,rmax,Nr)
      elif rPlot is None:
        rPlot = self.coords_[0][::rSample]
        rPlot = rPlot[rPlot<=rmax]

      if type(Ntheta) is int:
        thetaPlot = NP.linspace(0,2*NP.pi,Ntheta)
      elif thetaPlot is None:
        thetaPlot = self.theta()[::thetaSample]
        N         = len(thetaPlot)
        if thetaPlot[-1] == NP.pi:
          offset = 1
        else:
          offset = 0
        thetaTmp     = NP.zeros((N*2-offset,))
        thetaTmp[:N] = thetaPlot
        thetaTmp[N:] = 2.e0*NP.pi - thetaPlot[-1-offset::-1]
        thetaPlot    = thetaTmp

      rm,tm   = NP.meshgrid(rPlot,thetaPlot,indexing='ij')
      nPlot   = rm.shape
      rm      = rm.ravel()
      tm      = tm.ravel()
      ptsPlot = NP.asarray([rm*NP.cos(tm),rm*NP.sin(tm),NP.zeros(tm.shape)])
      
      #------------------------
      # 2. Rotate coordinates
      tr = -NP.arccos (normal[2])
      fr = -NP.arctan2(normal[1],normal[0])
      R  = NP.array([[ NP.cos(fr)*NP.cos(tr),-NP.sin(fr), NP.cos(fr)*NP.sin(tr)],\
                     [ NP.sin(fr)*NP.cos(tr), NP.cos(fr), NP.sin(fr)*NP.sin(tr)],\
                     [           -NP.sin(tr),          0,            NP.cos(tr)]])
      ptsPlot    = R.dot(ptsPlot)
      ptsPlotItp = NP.array(cartesianToSpherical(ptsPlot))

      #------------------------------
      # 3. Setup interpolation grid 
      # (complicated syntax necessary for 2D or 3D genericity)
      itpg = interpGrid(tuple(self.coords_))
      itpg.setNewCoords(tuple([ptsPlotItp[i,...] for i in range(self.Ndim_)]))

      #--------------
      # 4. Get data
      isScalar = data.ndim == self.Ndim_
      if isScalar:
        dataI = itpg(data)
      else:
        dataI = []
        for i in range(3):
          dataI.append(itpg(data[i]))
        dataI = NP.array(dataI)

      #------------
      # 5. Plot !
      if fileName is not None and getExtension(fileName) == 'vtk':
        plotOnMeshGridVTK(dataI,ptsPlot,nPlot,fileName,isScalar=isScalar)
      else:
        plotOnMeshGrid(dataI,ptsPlot,nPlot,isScalar,fileName,**pythonPlotArgs)

    def plotOnSphere(self,data,radius=1.0,\
                     fileName=None,thetaPlot=None,phiPlot=None,\
                     Ntheta=None,Nphi=None,thetaSample=1,phiSample=1,\
                     **pythonPlotArgs):

      ''' Plots data on a sphere centered at the origin.
          
          Points were data is plotted can be given in each dimension (theta,phi) by either
          thetaPlot   : vector of angles
          Ntheta      : linspace from 0 to pi (2pi for phi)
          thetaSample : original geometry vector (self.theta()) sample by thetaSample
      '''

      #---------------
      # 0.Check data
      if self.checkPlotData(data) == 1:
        return

      #--------------------------------------------------------
      # 1.Generate list of points before rotation (on a disk)

      if type(Ntheta) is int:
        thetaPlot = NP.linspace(0.0,NP.pi,Ntheta)
      elif thetaPlot is None:
        thetaPlot = self.coords_[1][::thetaSample]

      if type(Nphi) is int:
        phiPlot = NP.linspace(0.0,2.e0*NP.pi,Nphi)
      elif phiPlot is None:
        if isinstance(self,polarGeom3D):
          phiPlot = self.coords_[2][::phiSample]
        else:
          phiPlot = self.theta()[::phiSample]
          N         = len(phiPlot)
          if phiPlot[-1] == NP.pi:
            offset = 1
          else:
            offset = 0
          phiTmp     = NP.zeros((N*2-offset,))
          phiTmp[:N] = phiPlot
          phiTmp[N:] = 2.e0*NP.pi - phiPlot[-1-offset::-1]
          phiPlot    = phiTmp


      tm,fm      = NP.meshgrid(thetaPlot,phiPlot,indexing='ij')
      nPlot      = tm.shape
      tm         = tm.ravel()
      fm         = fm.ravel()
      rm         = radius*NP.ones(fm.shape)
      ptsPlot    = radius*NP.asarray([NP.sin(tm)*NP.cos(fm),NP.sin(tm)*NP.sin(fm),NP.cos(tm)])

      #------------------------------
      # 3. Setup interpolation grid 
      # (complicated syntax necessary for 2D or 3D genericity)
      itpg = interpGrid(tuple(self.coords_))
      itpg.setNewCoords(tuple([rm,tm,fm][0:self.Ndim_]))

      #--------------
      # 4. Get data
      isScalar = data.ndim == self.Ndim_
      if isScalar:
        dataI = itpg(data)
      else:
        dataI = []
        for i in range(3):
          dataI.append(itpg(data[i]))
        dataI = NP.array(dataI)

      #------------
      # 5. Plot !
      if fileName is not None and getExtension(fileName) == 'vtk':
        plotOnMeshGridVTK(dataI,ptsPlot,nPlot,fileName,isScalar=isScalar)
      else:
        plotOnMeshGrid(dataI,ptsPlot,nPlot,isScalar,fileName,**pythonPlotArgs)

    def cartesianGrad(self,U):

      dU       = self.sphericalGrad(U)
      r,th,phi = self.getSphericalCoordsMeshGrid()
      return sphericalToCartesianVector(dU,th,phi)

# ======================================================================

class polarGeom2D(polarGeom):
    ''' r,theta geometry
        By default : single half disk, 
                     r     from 0 to max(backgroundFile)
                     theta from 0 to pi
    '''

    def __init__(self,r=None,theta=None,Ntheta=None):
      self.initEmptyGeometry(2)
      if r is not None:
        self.setComponent(0,r)
      if theta is not None:
        self.setComponent(1,theta)
      if Ntheta is not None:
        self.setUniformComponent(1,Ntheta,NP.pi)

    def getCartesianCoordsMeshGrid(self):
      rm,tm = NP.meshgrid(self.coords_[0],self.coords_[1],indexing='ij')
      return rm*NP.sin(tm),NP.zeros(rm.shape),rm*NP.cos(tm)

    def getCopolarCoordsMeshGrid(self):
      return NP.meshgrid(self.coords_[0],self.coords_[1],indexing='ij')

    def getSphericalCoordsMeshGrid(self):
      rm,tm = NP.meshgrid(self.coords_[0],self.coords_[1],indexing='ij')
      phim  = NP.zeros(rm.shape)
      return rm,tm,phim

    def checkPlotData(self,data):

      if data.ndim == 2:
        if data.shape != tuple(self.N_):
          print(bColors.warning() + 'dimensions of given scalar data do not coincide with geometry. No plot generated.')
          return 1
      if data.ndim == 3:
        if data.shape[1:] != tuple(self.N_):
          print(bColors.warning() + 'dimensions[1:] of given vector data do not coincide with geometry. No plot generated.')
          return 1
        elif data.shape[0] not in [2,3]:
          print(bColors.warning() + 'dimensions[0] of given vector data is not 2 or 3. No plot generated.')
          return 1

    def plotOnAxiSlice(self,data,\
                       fileName=None,rPlot=None,thetaPlot=None,\
                       Nr=None,Ntheta=None,rSample=1,thetaSample=1,\
                       **pythonPlotArgs):

      ''' Plots data on a slice of the sphere, in the x0z plane
          
          Points were data is plotted can be given in each dimension by either
          rPlot   : vector of radiuses
          Nr      : linspace from 0 to max(radius), (2pi for theta)
          rSample : original geometry vector (self.r_) sample by rSample
      '''

      #---------------
      # 0.Check data
      if self.checkPlotData(data) == 1:
        return

      #--------------------------------------------------------
      # 1.Generate list of points
      if type(Nr) is int:
        rPlot = NP.linspace(0.0,rmax,Nr)
      elif rPlot is None:
        rPlot = self.coords_[0][::rSample]

      if type(Ntheta) is int:
        thetaPlot = NP.linspace(0,2*NP.pi,Ntheta)
      elif thetaPlot is None:
        thetaPlot = self.theta()[::thetaSample]
        N         = len(thetaPlot)
        if thetaPlot[-1] == NP.pi:
          offset = 1
        else:
          offset = 0
        thetaTmp     = NP.zeros((N*2-offset,))
        thetaTmp[:N] = thetaPlot
        thetaTmp[N:] = 2.e0*NP.pi - thetaPlot[-1-offset::-1]
        thetaPlot    = thetaTmp

      rm,tm   = NP.meshgrid(rPlot,thetaPlot,indexing='ij')
      nPlot   = rm.shape
      rm      = rm.ravel()
      tm      = tm.ravel()
      ptsPlot = NP.asarray([rm*NP.sin(tm),rm*NP.cos(tm)])
      
      #------------------------------
      # 3. Setup interpolation grid 
      # (complicated syntax necessary for 2D or 3D genericity)
      itpg = interpGrid(tuple(self.coords_))
      thetaItp = tm*(tm<NP.pi) + (2.e0*NP.pi-tm)*(tm>=NP.pi)
      itpg.setNewCoords((rm,thetaItp))

      #--------------
      # 4. Get data
      isScalar = data.ndim == self.Ndim_
      if isScalar:
        dataI = itpg(data)
      else:
        dataI = []
        for i in range(3):
          dataI.append(itpg(data[i]))
        dataI = NP.array(dataI)

      #------------
      # 5. Plot !
      if fileName is not None and getExtension(fileName) == 'vtk':
        plotOnMeshGridVTK(dataI,ptsPlot,nPlot,fileName,isScalar=isScalar)
      else:
        plotOnMeshGrid(dataI,ptsPlot,nPlot,isScalar,fileName,**pythonPlotArgs)

    def sphericalGrad(self,U):

      dims = list(U.shape)
      dims.insert(0,3)
      dU = NP.zeros(dims,dtype=U.dtype)

      if not hasattr(self,'diff_'):
        self.initDifferentiation()

      # dr(U), 1/r dtheta(U), 0.e0
      dU[0] = self.diff_[0](U,axis=0)
      with NP.errstate(all='ignore'):
        dU[1] = 1.e0/self.coords_[0][:,NP.newaxis]*self.diff_[1](U)
        dU[1] = NP.where( (self.coords_[0] != 0)[:,NP.newaxis], dU[1], 0)

      return dU

# ======================================================================

class polarGeom3D(polarGeom):
    ''' r,theta geometry with several modes
        By default : generating half disk
                     r     from 0 to max(backgroundFile)
                     theta from 0 to pi

                     phi   from 0 to pi with the computed number of modes
    '''

    def __init__(self,r=None,theta=None,Ntheta=None,phi=None,Nphi=None):
      self.initEmptyGeometry(3)
      if r is not None:
        self.setComponent(0,r)
      if theta is not None:
        self.setComponent(1,theta)
      if Ntheta is not None:
        self.setUniformComponent(1,Ntheta,NP.pi)
      if phi is not None:
        self.setComponent(2,phi)
      if Nphi is not None:
        self.setUniformComponent(2,Nphi,2*NP.pi)

    def Nphi(self):
      return self.N_[2]

    def r(self):
      return self.coords_[0]

    def theta(self):
      return self.coords_[1]

    def phi(self):
      return self.coords_[2]

    def getCartesianCoordsMeshGrid(self):
      rm,tm,phim = NP.meshgrid(self.coords_[0],self.coords_[1],self.coords_[2],indexing='ij')
      return rm*NP.sin(tm)*NP.cos(phim),rm*NP.sin(tm)*NP.sin(phim),rm*NP.cos(tm)

    def getCopolarCoordsMeshGrid(self):
      return NP.meshgrid(self.coords_[0],self.coords_[1],indexing='ij')

    def getSphericalCoordsMeshGrid(self):
      return NP.meshgrid(self.coords_[0],self.coords_[1],self.coords_[2],indexing='ij')

    def checkPlotData(self,data):

      if data.ndim == 3:
        if data.shape != tuple(self.N_):
          print(bColors.warning() + 'dimensions of given scalar data do not coincide with geometry. No plot generated.')
          return 1
      if data.ndim == 4:
        if data.shape[1:] != tuple(self.N_):
          print(bColors.warning() + 'dimensions[1:] of given vector data do not coincide with geometry. No plot generated.')
          return 1
        elif data.shape[0] not in [2,3]:
          print(bColors.warning() + 'dimensions[0] of given vector data is not 2 or 3. No plot generated.')
          return 1

    def sphericalGrad(self,U):

      dims = list(U.shape)
      dims.insert(0,3)
      dU = NP.zeros(dims,dtype=U.dtype)

      if not hasattr(self,'diff_'):
        self.initDifferentiation()

      # dr(U), 1/r dtheta(U), 1/rsin(theta) dphi(U)
      dU[0] = self.diff_[0](U,axis=0)
      with NP.errstate(all='ignore'):
        dU[1] = 1.e0/self.coords_[0][:,NP.newaxis,NP.newaxis]\
              *self.diff_[1](U,axis=1)
        dU[2] = 1.e0/self.coords_[0][:,NP.newaxis,NP.newaxis]\
              *NP.sin(self.coords_[1][NP.newaxis,:,NP.newaxis])\
              *self.diff_[2](U)

        dU[1] = NP.where( (self.coords_[0] != 0)[:,NP.newaxis,NP.newaxis], dU[1], 0)
        dU[2] = NP.where( (self.coords_[0] != 0)[:,NP.newaxis,NP.newaxis] * (NP.sin(self.coords_[1]) != 0)[NP.newaxis,:,NP.newaxis], dU[2], 0)


      return dU

