import numpy as NP

from ...Common import *

class geom(object):
    ''' Base class for informations about geometry:
        coordinates, number of points, etc...
        
        Not so much in here. See Polar geometry, cartesian geometry, etc...
    '''

    def setUniformComponent(self,dim,N,L,origin=0):
      self.coords_[dim] = NP.linspace(origin,L,N)
      self.N_[dim]      = N
      self.h_[dim]      = self.coords_[dim][1]-self.coords_[dim][0]

    def setComponent(self,dim,points):
      self.coords_[dim] = NP.array(points)
      self.N_[dim]      = len(self.coords_[dim])
      self.h_[dim]      = NP.diff(self.coords_[dim])

    def initEmptyGeometry(self,ndim):
      self.Ndim_   = ndim
      self.N_      = [0]*ndim
      self.h_      = [0.e0]*ndim
      self.coords_ = [None]*ndim

    def initDifferentiation(self):
      ''' store finite difference matrices
          int each dimension
      '''
    
      self.diff_ = []
      for dim in range(self.Ndim_):
        self.diff_.append(FDM_Compact(self.coords_[dim]))

    def getCartesianCoordsList(self):
      points = NP.array(self.getCartesianCoordsMeshGrid())
      points =  points.reshape((3,NP.product(self.N_)))
      return points.T 

    def getCopolarCoordsList(self):
      points = NP.array(self.getCopolarCoordsMeshGrid())
      points =  points.reshape((2,NP.product(self.N_)))
      return points.T 

    def getSphericalCoordsList(self):
      points = NP.array(self.getSphericalCoordsMeshGrid())
      points =  points.reshape((3,NP.product(self.N_)))
      return points.T 

    def getNearestIndex(self,x):

      c = self.getCartesianCoordsMeshGrid()
      d = NP.sqrt((c[0]-x[0])**2+(c[1]-x[1])**2+(c[2]-x[2])**2)
      return NP.unravel_index(NP.argmin(d),d.shape)


####################################################################
# Interpolation routines from a geometry to another

def initCart2DToRadial(r,x,y=None):

    ''' Creates an interpGrid structure for interpolation
        from the cartesian grid to a (r,theta) disk grid.
    '''

    if y is None:
      y = x
    theta  = NP.linspace(0,2*pi,len(x))
    rr,tt  = NP.meshgrid(r,theta)
    r1     = rr*NP.cos(tt)
    r2     = rr*NP.sin(tt)
    coords = (r1.ravel(),r2.ravel())

    interp = interpgrid([x, y])
    interp.set_new_coords(coords)

    return interp,len(r),len(theta)
  
def getCart2DToRadial(field2D,interp,Nr,Ntheta):

    ''' Returns the interpolated field, 
        averaged over the disk angle
    ''' 

    field1D = NP.reshape(interp(field2D.ravel()),(Nr,Ntheta))
    field1D = NP.mean(field1D,axis=0)
    return field1D

def cart2DToRadial(field2D,r,x,y=None):
    ''' same thing without storing itp structure '''

    if y is None:
      y = x
    theta  = NP.linspace(0,2*pi,len(x))
    rr,tt  = NP.meshgrid(r,theta)
    r1     = rr*NP.cos(tt)
    r2     = rr*NP.sin(tt)
    coords = (r1.ravel(),r2.ravel())

    interp = interpgrid([x, y])
    interp.set_new_coords(coords)


    return field1D

def radialToCart2D(field1D,r,x,y=None):

    ''' Projects an angle independant 1D field defined on r points
        into a 2D Cartesian map set by (x,y)
    '''

    if y is None:
      y = x
    xx,yy = NP.meshgrid(x,y,indexing='ij')
    rr    = NP.hypot(xx,yy) 
    
    itp   = ITP.UnivariateSpline(r,field1D)
    return itp(rr)
