# apply rotation matrices to points 

import numpy             as NP
import scipy.interpolate as INTERP
import time
from   numpy             import pi
from   .          import *


class rotation:

  def __init__(self,theta,phi):

    cf = cos(phi)
    sf = sin(phi)
    ct = cos(theta)
    st = sin(theta)

    self.R = NP.array([cf*ct,-sf,st*cf,sf*ct,cf,st*sf,-st,0.e0,ct])
    self.R = NP.reshape(self.R,(3,3))

  def inverseImageCart(self,M2):
    # returns M1 given M2, where M1 and M2 are expressed in cartesian coordinates
    return NP.dot(self.R.T,M2)

  def inverseImageSph(self,M2):
    # same if points are given in sph coordinates
    MC = sphericalToCartesian(M2)
    return cartesianToSpherical(NP.dot(self.R.T,MC))

  def rotateField(self,U1):
    # returns the rotation of U1 by class angles theta,phi
    # on the points where U1 is defined (coords Mr, Mt, Mf)
    tmp = self.itpm(U1)
    return NP.reshape(tmp,U1.shape)

  def setInterpolationField(self,coords):
    # stores the interpolation matrix that computes the images
    # by rotation of points defined by (r,theta,phi) from values 
    # given on the same (r,theta,phi) grid

    if len(coords) == 3:
      rm,tm,fm = NP.meshgrid(coords[0],coords[1],coords[2],indexing='ij')
    if len(coords) == 2:
      tm,fm    = NP.meshgrid(coords[0],coords[1],indexing='ij')
      rm       = NP.ones(NP.size(tm))

    coords2  = NP.vstack((rm.ravel(),tm.ravel(),fm.ravel()))
    coords1  = self.inverseImageSph(coords2) 

    if len(coords) == 3:
      # correct coordinates (remove epsilons outside that can stop interpolation)
      coords1[0,:] = NP.minimum(coords1[0,:],rm[-1])
      coords1[0,:] = NP.maximum(coords1[0,:],rm[0])

    self.itpm = interpGrid(coords)

    if len(coords) == 3:
      self.itpm.setNewCoords((coords1[0,...],coords1[1,...],coords1[2,...]))
    if len(coords) == 2:
      self.itpm.setNewCoords((coords1[1,...],coords1[2,...]))



