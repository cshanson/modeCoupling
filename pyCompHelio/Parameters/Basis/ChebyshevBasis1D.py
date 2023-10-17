from .basis1D import *

#########################################################

class ChebyshevBasis1D(basis1D):
    '''Class for 1d B splines. It contains the position of the knots, the order of the spline and the radial vector r.'''   
     
    def __init__(self, x, NdegreeMax):
        basis1D.__init__(self, x)
        self.order_ = NdegreeMax+1
        self.nbBasisFunctions_ = NdegreeMax+1

        self.computeMassMatrix()

    def __call__(self, degNb, order = None, x = None, derivative = 0):
      '''Returns the value of the B spline at r. If derivative = 1 returns the value of the derivative.''' 
      if order is None:
        order = self.order_
      if x is None:
        x = self.x_

      ChebyshevClass = NP.polynomial.Chebyshev.basis(degNb,[x[0],x[-1]])

      return ChebyshevClass.deriv(derivative)(x)


    def projectOnBasis(self,quantity,axis=-1):

      ndims = quantity.ndim
      if ndims > 1:
        quantity = NP.moveaxis(quantity,axis,-1)
        dims = quantity.shape 
        quantity.reshape(-1,quantity.shape[axis])
        res = []
        for ii in range(len(quantity)):
          res.append(NP.polynomial.Chebyshev.fit(self.x_, quantity[ii], self.nbBasisFunctions_-1).coef)

        return NP.moveaxis(NP.reshape(res,dims[:-1] + (len(res[-1]),)),-1,axis)
      else:
        return NP.polynomial.Chebyshev.fit(self.x_, quantity, self.nbBasisFunctions_-1).coef

    # def computeMassMatrix(self):
    #   x = self.x_
    #   A = NP.zeros((self.nbBasisFunctions_, self.nbBasisFunctions_))

    #   xS = subSampleVector(x, 0.01)
    #   bi = NP.zeros((self.nbBasisFunctions_, len(xS)))
    #   for i in range(self.nbBasisFunctions_):
    #     bi[i,:] = self(i,x=xS)

    #   for i in range(self.nbBasisFunctions_):
    #     for j in range(self.order_+1):
    #       if i+j < self.nbBasisFunctions_:
    #         A[i,i+j] = simps(bi[i,:] * bi[i+j,:], x=xS)
    #   return A + A.T - NP.diag(NP.diag(A)) 