import numpy as NP
from .basis1D import *
from scipy import interpolate
#########################################################

class constantBasis1D(basis1D):
  
  def __init__(self, x,subsample=1):
    x = x[::subsample]
    self.subsample_ = subsample
    basis1D.__init__(self, x)
    hx = NP.diff(self.x_)
    self.hx_ = NP.concatenate([hx, [hx[-1]]])
    self.mass_ = NP.diag(self.hx_)
    self.nbBasisFunctions_ = len(x)

  def __call__(self, i, r=None,derivative=0):
   '''Returns a rectangular function equal to 1 on the interval i and 0 otherwise: fi(r) = 1 if points[i] <= r < points[i+1], 0 otherwise.'''
   if derivative == 0:
    if r is None:
      r = self.x_
    if i==self.nbBasisFunctions_: # to include the last point
      return (r>=self.x_[i]).astype('int')
    else:
      if i == len(self.x_)-1:
        res = NP.zeros(len(self.x_))
        res[-1] = 1
        return res
      return (NP.logical_and(r >= self.x_[i], r < self.x_[i+1])).astype('int')

   if derivative == 1:
    res = NP.zeros(len(self.x_))
    if i == 0:
      res[0] = -1
      res[1] = 1
      return res/self.hx_[0]
    elif i == len(self.x_)-1:
      res[-1] = 1
      res[-2] = -1
      return res/self.hx_[-1]
    else:
      res[i] = (self.hx_[i+1]**2 - self.hx_[i]**2)
      res[i+1] = self.hx_[i]**2
      res[i-1] = -self.hx_[i+1]**2
      return res/(self.hx_[i+1]*self.hx_[i]*(self.hx_[i] + self.hx_[i+1]))

  def projectOnBasis(self, quantity,axis=-1):
    #hr = NP.diff(self.r_)
    #hr = NP.concatenate([hr, [hr[-1]]])
    #A = hr  * quantity
    tmp = NP.moveaxis(quantity,axis,0)
    tmp = tmp[::self.subsample_]
    tmp = NP.moveaxis(tmp,0,axis)
    return tmp
    # return quantity
    
  def reconstructFromBasis(self, coeffs, xFinal = None, axis=-1,derivative=0):
    if derivative == 1:
      coeffs = NP.gradient(coeffs,self.x_,axis=axis)
    elif derivative == 2:
      coeffs = NP.gradient(NP.gradient(coeffs,self.x_),self.x_,axis=axis)
    if xFinal is None:
        return coeffs
    else:
      INTERP = scipy.interpolate.interp1d(self.x_, coeffs, kind='linear', axis=axis)
      return INTERP(xFinal)
      # return NP.interp(xFinal, self.x_, coeffs)

            
  def createSmoothnessMatrix(self, smoothnessOrder = 0, BCleft = None, BCright = None, x = None,weight = None):
    '''compute the smoothness matrix L for regularization when the quantity to recover is 1D defined on the grid x. If smoothnessOrder = 0, L is the L2 norm, smoothnessOrder = 1, norm of the gradient, smoothnessOrder = 2, norm of the Laplacian'''
    if x is None:
      x = self.x_
    hx = NP.diff(x)
    L = NP.zeros((len(x),len(x)))

    if smoothnessOrder == 0:
      L[1:,1:]  = NP.diag(hx)
      L[0,0]    = hx[0]
      if BCleft == 'Neumann':
        L[0,1] = -hx[0]
      if BCright == 'Neumann':
        L[-1,-2] = -hx[-1]

    elif smoothnessOrder == 1: 
      for i in range(len(x)-1):
        L[i,i] = 1/hx[i]
        L[i,i+1] = -1/hx[i]

      if BCleft == None and BCright == None:
        L = L[:-1,:]
      elif BCleft == 'Neumann' and BCright == 'Neumann':
        L[-1,-1] = 0.
      elif BCleft == 'Neumann' and BCright == 'Dirichlet':
        L[-1,-1] = 1/hx[-1]
      elif BCleft == 'Dirichlet' and BCright == 'Neumann':
        L[0,0] = L[1,1]

    elif smoothnessOrder == 2:
      for i in range(1,len(x)-1):
        h1 = x[i+1]-x[i]
        h2 = x[i]-x[i-1]
        L[i,i-1] = -2*h1/(h1**3 + h2**3)
        L[i,i]   = 2*(h1+h2) / (h1**3+h2**3)
        L[i,i+1] = -2*h2/(h1**3+h2**3)

      # Boundary conditions
      if BCleft == None:
        L = L[1:,:]
      elif BCleft == 'Neumann':
        L[0,0] = -L[1,0]
        L[0,1] = L[1,0]
      elif BCleft == 'Dirichlet':
        L[0,0] = L[1,1]
        L[0,1] = L[1,0]
      elif BCleft == 'AntiReflective':
        L[0,0] = 0.
        L[0,1] = 0.    
      else:
        raise NotImplementedError('This type of boundary condition (%s) is not implemented' % BCleft)

      if BCright == None: 
        L = L[:-1,:]
      elif BCright == 'Neumann':
        L[-1,-1] = -L[-2,-1]
        L[-1,-2] = L[-2,-1]
      elif BCright == 'Dirichlet':
        L[-1,-1] = L[-2,-2]
        L[-1,-2] = L[-2,-1]
      elif BCright == 'AntiReflective':
        L[-1,-1] = 0.
        L[-1,-2] = 0. 
      else:
        raise NotImplementedError('This type of boundary condition (%s) is not implemented' % BCright)

    if weight is not None:
      L = NP.dot(NP.diag(weight),L)

    return L
