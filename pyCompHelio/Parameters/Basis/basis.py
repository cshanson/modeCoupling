from abc import ABCMeta, abstractmethod

#########################################################
class basis:
    #class basis(metaclass=ABCMeta):
    '''Abstract class for all type of basis functions such as polynomials, B splines, ... 
    The subclasses should implement at least the methods projectOnBasis and reconstructFromBasis.
    createSmoothnessMatrix should also be implemented to use this basis function in the inversions.'''
    __metaclass__ = ABCMeta
   
    @abstractmethod
    def projectOnBasis(self, quantity):
        raise NotImplementedError('subclasses must override projectOnBasis!')
    
    @abstractmethod
    def reconstructFromBasis(self, coeffs, xFinal):
        raise NotImplementedError('subclasses must override reconstructFromBasis!')
        
    @abstractmethod
    def createSmoothnessMatrix(self, smoothnessOrder = 0):
        raise NotImplementedError('subclasses must override createSmoothnessMatrix!')

    def computeMassMatrix(self,scaling=None):
      '''Compute the mass matrix M_{ij}\int B_i B_j.'''
      self.mass_ = self.createSmoothnessMatrix()
        
