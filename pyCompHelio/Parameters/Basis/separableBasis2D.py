from abc   import ABCMeta, abstractmethod
from .basis import *
from scipy import sparse

#########################################################

class separableBasis2D(basis):
#class separableBasis2D(basis, metaclass=ABCMeta):
  '''Class for basis functions of two variables r1, r2 that are separable (f(r1,r2) = g(r1)h(r2)).'''
  __metaclass__ = ABCMeta
  def __init__(self, basis1, basis2):
   self.basis1_ = basis1
   self.basis2_ = basis2
   self.nbBasisFunctions_ = [self.basis1_.nbBasisFunctions_, self.basis2_.nbBasisFunctions_]

  def projectOnBasis(self, quantity):
    proj2 = self.basis2_.projectOnBasis(quantity,axis=-1)
    return self.basis1_.projectOnBasis(proj2,axis=-2)

  def reconstructFromBasis(self, coeffs, r1Final=None, r2Final=None):
    reconstr2 = self.basis2_.reconstructFromBasis(coeffs, r2Final, axis=-1)
    return self.basis1_.reconstructFromBasis(reconstr2, r1Final,axis=-2)

  def createSmoothnessMatrix(self, smoothnessOrder1 = 0, BCleft1 = None, BCright1 = None, r1 = None, scaling1 = None, smoothnessOrder2 = 0, BCleft2 = None, BCright2 = None, r2 = None, scaling2 = None):
    L1 = self.basis1_.createSmoothnessMatrix(smoothnessOrder1, BCleft1, BCright1, r1, scaling1)
    L1 = sparse.csr_matrix(L1)
    L2 = self.basis2_.createSmoothnessMatrix(smoothnessOrder2, BCleft2, BCright2, r2, scaling2)
    L2 = sparse.csr_matrix(L2)
    return sparse.kron(L1,L2)
