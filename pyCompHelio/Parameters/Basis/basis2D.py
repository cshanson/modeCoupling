from abc   import ABCMeta, abstractmethod
from .basis import *
from scipy import sparse

#########################################################

class basis2D(basis):
#class basis2D(basis, metaclass=ABCMeta):
  '''Abstract class for all type of basis functions depending on two variables. 
  The subclasses should implement at least the methods projectOnBasis, reconstructFromBasis and integrate.
  createSmoothnessMatrix should also be implemented to use this basis function in the inversions.'''
  __metaclass__ = ABCMeta

  def __init__(self, r1, r2):
   self.r1_ = r1
   self.r2_ = r2
