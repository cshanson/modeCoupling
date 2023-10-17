import numpy    as     NP
from   .geometry import *

class surfaceGeom(geom):

    def cartesianGrad(self,U):                                                 
                                                                               
      dU       = self.sphericalGrad(U)                                         
      r,th,phi = self.getSphericalCoordsMeshGrid()                             
      return sphericalToCartesianVector(dU,th,phi) 

    def Nr(self):
      return 1

    def Ntheta(self):
      return self.N_[0]

    def r(self):
      return self.r_

    def theta(self):
      return self.coords_[0]

    def axisTheta(self):
      return 0

class surfaceGeom1D(surfaceGeom):
    ''' Output at a single radius, depending on theta.
        By default, theta is from 0 to pi
    '''

    def __init__(self,r,theta=None,Ntheta=None):
      self.initEmptyGeometry(1)
      self.r_      = r
      if theta is not None:
        self.setComponent(0,theta)
      if Ntheta is not None:
        self.setUniformComponent(0,Ntheta,NP.pi)

    def getCartesianCoordsMeshGrid(self):
      return self.r_*NP.sin(self.coords_[0]),NP.zeros(self.N_[0]),self.r_*NP.cos(self.coords_[0])

    def getCopolarCoordsMeshGrid(self):
      return self.r_*NP.ones((self.N_[0],)),self.coords_[0]

    def getSphericalCoordsMeshGrid(self):
      return self.r_*NP.ones((self.N_[0],)),self.coords_[0],NP.zeros((self.N_[0],))

    def sphericalGrad(self,U):

      dims = list(U.shape)
      dims.insert(0,3)
      dU = NP.zeros(dims,dtype=U.dtype)

      if not hasattr(self,'diff_'):
        self.initDifferentiation()

      # 0.e0, 1/r dtheta, 0.e0
      dU[1] = 1.e0/self.r_*self.diff_[0](U,axis=0)

      return dU

####################################################################

class surfaceGeom2D(surfaceGeom):
    ''' Same as surface1D with modes. i.e. (theta,phi) geometry
    '''

    def __init__(self,r,theta=None,Ntheta=None,phi=None,Nphi=None):
      self.initEmptyGeometry(2)
      self.r_      = r
      if theta is not None:
        self.setComponent(0,theta)
      if Ntheta is not None:
        self.setUniformComponent(0,Ntheta,NP.pi)
      if phi is not None:
        self.setComponent(1,phi)
      if Nphi is not None:
        self.setUniformComponent(1,Nphi,2*NP.pi)

    def Nphi(self):
      return self.N_[1]

    def phi(self):
      return self.coords_[1]

    def getCartesianCoordsMeshGrid(self):
      tm,fm = NP.meshgrid(self.coords_[0],self.coords_[1],indexing='ij')
      return self.r_*NP.sin(tm)*NP.cos(fm),\
             self.r_*NP.sin(tm)*NP.sin(fm),\
             self.r_*NP.cos(tm)

    def getCopolarCoordsMeshGrid(self):
      return self.r_*NP.ones((self.N_[0],)),self.coords_[0]

    def getSphericalCoordsMeshGrid(self):
      tm,fm = NP.meshgrid(self.coords_[0],self.coords_[1],indexing='ij')
      return self.r_*NP.ones(tm.shape),tm,fm

    def sphericalGrad(self,U):

      dims = list(U.shape)
      dims.insert(0,3)
      dU = NP.zeros(dims,dtype=U.dtype)

      if not hasattr(self,'diff_'):
        self.initDifferentiation()

      # 0.e0, 1/r dtheta(U), 1/rsin(theta) dphi(U)
      dU[1] = 1.e0/self.r_*self.diff_[0](U,axis=0)
      dU[2] = 1.e0/self.r_*NP.sin(self.coords_[0][:,NP.newaxis])\
              *self.diff_[1](U,axis=1)

      return dU
