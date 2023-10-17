from abc             import ABCMeta, abstractmethod
from .basis           import *
import numpy         as NP
from ...Common       import *
from scipy.integrate import simps
from scipy import interpolate

#########################################################

class basis1D(basis):
#class basis1D(basis, metaclass=ABCMeta):
  '''Abstract class for all type of 1D basis functions such as polynomials, B splines, ... 
  The subclasses should implement the method __call__ to evalute the i^th basis function at position r and can overwrite the methods projectOnBasis, reconstructFromBasis and createSmoothnessMatrix if necessary.'''
  __metaclass__ = ABCMeta
  def __init__(self, x):
    self.x_ = x
    self.quadrature_ = False

  def projectOnBasis(self, quantity,axis = -1):
    '''Returns the coefficients of quantity on the given radial basis.'''
    x = self.x_
    if self.quadrature_:
      nodes = self.getGlobalNodes()
      A = NP.zeros(self.nbBasisFunctions_)
      if len(x) != self.nbBasisFunctions_:
        # interpolate the function at the nodal points
        INTERP = interpolate.interp1d(x, quantity,axis = axis)
        A = INTERP(nodes)
    else:
      # No quadrature formula, compute the basis coefficients
      # right hand side rhs_i = \int f B_i
      size = NP.array(quantity.shape)
      size[axis] = self.nbBasisFunctions_
      finalsize = NP.delete(size,axis)
      rhs =  []
      quantity = NP.moveaxis(quantity,axis,-1)
      for i in range(self.nbBasisFunctions_):
        quantityNEW = simps(quantity*self(i,x=x),x=x,axis=-1)
        quantityNEW = quantityNEW.reshape(quantityNEW.size)
        rhs.append(quantityNEW)
      rhs = NP.array(rhs)
        
      # Solve M coeff = rhs where M is the mass matrix
      A = NP.zeros((self.nbBasisFunctions_,rhs.shape[1]))
      for i in range(rhs.shape[1]):
        A[:,i] = NP.linalg.solve(self.mass_,rhs[:,i])
      # print 'A.shape',A.shape
      # if axis != 0:
      #   A = A.T
      A = A.reshape(NP.concatenate([[self.nbBasisFunctions_],finalsize]))
      A = NP.moveaxis(A,0,axis)
      # A = A.reshape(NP.rollaxis(NP.array(size),axis,quantity.ndim-1))
      # A = A.reshape(NP.swapaxes)

    return A          
            
  def reconstructFromBasis(self, coeffs, xFinal = None,axis=-1,derivative = 0):
    '''Reconstructs quantity at positions xFinal from its basis coefficients.'''
    if xFinal is None:
      xFinal = self.x_
    size = list(coeffs.shape)
    size[axis] = len(xFinal)
    quantity = NP.zeros(size)
    coeffs = NP.swapaxes(coeffs,axis,quantity.ndim-1)
    quantity = NP.swapaxes(quantity,axis,quantity.ndim-1)
    for i in range(self.nbBasisFunctions_):
      xvector = self(i, x=xFinal,derivative=derivative)
      for j in range(len(xFinal)):
        quantity[...,j] += coeffs[...,i] * xvector[j]
    quantity = NP.swapaxes(quantity,axis,quantity.ndim-1)
    return quantity
        
  def createSmoothnessMatrix(self, smoothnessOrder = 0, BCleft = None, BCright = None, x = None, scaling = None,\
                             smoothnessOrderj = None,weight=None):
    '''Creates a regularization matrix \int B_i B_j for the inversion.'''
    if x is None:
      x = self.x_
    if smoothnessOrderj is None:
      smoothnessOrderj = smoothnessOrder
    if weight is None:
      weight = NP.ones(len(x))
    A = NP.zeros((self.nbBasisFunctions_, self.nbBasisFunctions_))
    xS = subSampleVector(x, 0.01)# for accurate integration
    if scaling is not None:
      scaling = NP.interp(xS,x,scaling)
    else:
      scaling = 1
    # basis function on each interval
    bi = NP.zeros((self.nbBasisFunctions_, len(xS)))
    bj = NP.zeros((self.nbBasisFunctions_, len(xS)))
    for i in range(self.nbBasisFunctions_):
      bi[i,:] = self(i, derivative=smoothnessOrder,x=xS) * scaling
      if smoothnessOrderj is not None:
        bj[i,:] = self(i, derivative=smoothnessOrder,x=xS) * scaling

    # fill matrix the upper part of the matrix of \int B_i B_j
    for i in range(self.nbBasisFunctions_):
      for j in range(self.order_+1):
        if i+j < self.nbBasisFunctions_:
          A[i,i+j] = simps(bi[i,:] * bi[i+j,:], x=xS)

    return A + A.T - NP.diag(NP.diag(A)) # add the symmetric part
