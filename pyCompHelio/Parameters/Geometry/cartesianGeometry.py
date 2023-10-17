import  numpy             as NP
import  scipy.interpolate as ITP
from    .geometry import *

class cartesianGeom(geom):

    def initCartesianCoords(self,L=None,N=None,coords=None):
      ''' lengths L, number of points N and/or coordinates coords
          should be given as 2-uples or 3-uples
      '''

      if L is not None and N is not None:

        L = list(L)
        N = list(N)
        if len(L) != self.Ndim_: 
          L = [L[0]]*self.Ndim_      
        if len(N) != self.Ndim_: 
          N = [N[0]]*self.Ndim_
        self.hk_ = []
        self.k_  = []

        for dim in range(self.Ndim_):
          self.setUniformComponent(dim,N[dim],L[dim],-L[dim]/2.0)
          self.hk_.append(2.e0*NP.pi/float(N[dim]))
          if (N[dim]%2==0):
            self.k_.append(self.hk_[dim]*NP.arange(-N[dim]/2,N[dim]/2))
          else:
            self.k_.append(self.hk_[dim]*NP.arange(-(N[dim]-1)/2,(N[dim]+1)/2))
          # Put the O frequency first
          self.k_[dim] = NP.fft.ifftshift(self.k_[dim])

      if coords is not None:
        try:
          for dim in range(self.Ndim_): 
            self.setComponent(dim,coords[dim])
        except:
          raise Exception("Were coordinates given as a tuple or list of vectors ?")

    def sphericalGrad(self,U):

      dU          = self.cartesianGradient(U)
      r,theta,phi = self.getSphericalCoordsMeshGrid()
      return cartesianToSphericalVector(dU,theta,phi)

    def axisTheta(self):
      raise Exception("No theta dimension on cartesian geometries !")

class cartesianGeom2D(cartesianGeom):
    ''' Classic x,y grid
    '''

    def __init__(self,L=None,N=None,coords=None):

      self.initEmptyGeometry(2)
      self.initCartesianCoords(L,N,coords)

    # =========================================================================

    def Nx(self):
      return self.N_[0]

    def Ny(self):
      return self.N_[1]

    def Nz(self):
      raise Exception('No 3rd dimension in cartesian2D geometry')

    def x(self):
      return self.coords_[0]

    def y(self):
      return self.coords_[1]

    def z(self):
      raise Exception('No 3rd dimension in cartesian2D geometry')

    def getCartesianCoordsMeshGrid(self):
      return NP.meshgrid(self.coords_[0],self.coords_[1],indexing='ij')

    def getCopolarCoordsMeshGrid(self):
      xm,ym = NP.meshgrid(self.coords_[0],self.coords_[1],indexing='ij')
      return NP.sqrt(xm*xm+ym*ym),NP.arctan2(xm,ym)

    def getSphericalCoordsMeshGrid(self):
      xm,ym = NP.meshgrid(self.coords_[0],self.coords_[1],indexing='ij')
      return NP.sqrt(xm*xm+ym*ym),NP.arctan2(xm,ym),NP.zeros(ym.shape)

    def getPointAndWeightingQuadrant(self,radius,typeOfAveraging):
      ''' Returns the required points to define an averaging over 
          the whole annulus or some quadrants.
          Also returns the given weights associated with the different 
          part of the average (+/-1)
      '''
      [xx,yy] = self.getCartesianCoordsMeshGrid()

      if typeOfAveraging == TypeOfTravelTimeAveraging.ANN:
        ind       = NP.where(NP.abs(NP.hypot(xx,yy)-radius)<self.h_[0]/2.e0)
        inds      = NP.zeros((2,len(ind[0])),dtype=int)
        inds[1,:] = NP.round(xx[ind]/self.h_[0]).astype(int)
        inds[0,:] = NP.round(yy[ind]/self.h_[1]).astype(int)
        weights   = NP.ones(inds.shape[1])
      else: # Quadrants
        if typeOfAveraging == TypeOfTravelTimeAveraging.EW:
          # East
          indsM = NP.where((NP.abs(NP.hypot(xx,yy)-radius)<self.h_[0]/2.e0))\
                          *(xx<=-radius*NP.sqrt(2.e0)/2.e0)
          # West
          indsP = NP.where((NP.abs(NP.hypot(xx,yy)-radius)<self.h_[0]/2.e0))\
                          *(xx>= radius*NP.sqrt(2.e0)/2.e0)

        elif typeOfAveraging == TypeOfTravelTimeAveraging.SN:
          # South
          indsM = NP.where((NP.abs(NP.hypot(xx,yy)-radius)<self.h_[0]/2.e0))\
                          *(yy<=-radius*NP.sqrt(2.e0)/2.e0)
          # North
          indsP = NP.where((NP.abs(NP.hypot(xx,yy)-radius)<self.h_[0]/2.e0))\
                          *(yy>= radius*NP.sqrt(2.e0)/2.e0)

        inds    = NP.concatenate((indsP,indsM),axis=1)
        inds    = NP.roll(inds,1,axis=0) # Put x in inds[1] and y in inds[0]
        weights = NP.ones((inds.shape[1],))
        weights[indPs.shape[1]:] = -1.e0

      return inds,weights

    def cartesianGrad(self,U):

      dims = list(U.shape)
      dims.insert(0,3)
      dU = NP.zeros(dims,dtype=U.dtype)

      if not hasattr(self,'diff_'):
        self.initDifferentiation()

      # dx(U),dy(U),0.e0
      dU[0] = self.diff_[0](U,axis=0)
      dU[1] = self.diff_[1](U)

      return dU

####################################################################

class cartesianGeom3D(cartesianGeom):
    ''' Classic x,y,z grid
    '''

    def __init__(self,L=None,N=None,X=None):
      self.initEmptyGeometry(3)
      self.initCartesianCoords(L,N,coords)

    def Nx(self):
      return self.N_[0]

    def Ny(self):
      return self.N_[1]

    def Nz(self):
      return self.N_[2]

    def x(self):
      return self.coords_[0]

    def y(self):
      return self.coords_[1]

    def z(self):
      return self.coords_[2]

    def getCartesianCoordsMeshGrid(self):
      return NP.meshgrid(self.coords_[0],self.coords_[1],self.coords_[2],indexing='ij')

    def getCopolarCoordsMeshGrid(self):
      xm,ym,zm = NP.meshgrid(self.coords_[0],self.coords_[1],self.coords_[2],indexing='ij')
      rm = NP.sqrt(xm*xm+ym*ym)
      return rm,NP.arctan2(rm,zm)

    def getSphericalCoordsMeshGrid(self):
      xm,ym,zm = NP.meshgrid(self.coords_[0],self.coords_[1],self.coords_[2],indexing='ij')
      rm = NP.sqrt(xm*xm+ym*ym)
      return NP.sqrt(xm*xm+ym*ym+zm*zm),NP.arctan2(rm,zm),NP.arctan2(xm,ym)

    def cartesianGrad(self,U):

      dims = list(U.shape)
      dims.insert(0,3)
      dU = NP.zeros(dims,dtype=U.dtype)

      if not hasattr(self,'diff_'):
        self.initDifferentiation()

      # dx(U),dy(U),dz(U)
      dU[0] = self.diff_[0](U,axis=0)
      dU[1] = self.diff_[1](U,axis=1)
      dU[2] = self.diff_[2](U)
