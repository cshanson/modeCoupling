from .basis1D import *

#########################################################

class BsplineBasis1D(basis1D):
    '''Class for 1d B splines. It contains the position of the knots, the order of the spline and the radial vector r.'''   
     
    def __init__(self, x, knots, order):
        basis1D.__init__(self, x)
        self.knots_ = NP.array(knots)
        self.order_ = order
        self.nbBasisFunctions_ = len(self.knots_)+order-1
        for i in range(0,order):
          self.knots_ = NP.insert(self.knots_, 0, self.knots_[0])
          self.knots_ = NP.insert(self.knots_, -1, self.knots_[-1])

        self.computeMassMatrix()

    def __call__(self, knotsNb, order = None, x = None, derivative = 0):
      '''Returns the value of the B spline at r. If derivative = 1 returns the value of the derivative.''' 
      if order is None:
        order = self.order_
      if x is None:
        x = self.x_
      if derivative == 0:
        if (order == 0):
          if  knotsNb+1 != self.nbBasisFunctions_:
            return (NP.logical_and(x >= self.knots_[knotsNb], x < self.knots_[knotsNb+1])).astype('int')
          else:
            return (x >= self.knots_[knotsNb]).astype('int')*(x <= self.knots_[-1]).astype('int')
        else:
          res = NP.zeros(x.shape)
          if (knotsNb+order < len(self.knots_) and self.knots_[knotsNb+order] != self.knots_[knotsNb]):
            res += (x -self.knots_[knotsNb]) / (self.knots_[knotsNb+order] - self.knots_[knotsNb]) * self(knotsNb, order-1, x)
          if (knotsNb+order+1 < len(self.knots_) and self.knots_[knotsNb+order+1] != self.knots_[knotsNb+1]):
            res += (self.knots_[knotsNb+order+1] - x) / (self.knots_[knotsNb+order+1] - self.knots_[knotsNb+1]) * self(knotsNb+1, order-1, x)
        return res

      elif derivative == 1:
        res = NP.zeros(x.shape)
        if (knotsNb+order < len(self.knots_) and self.knots_[knotsNb+order] != self.knots_[knotsNb]):
          res += self(knotsNb, order-1,x) / (self.knots_[knotsNb+order] - self.knots_[knotsNb])
        if (knotsNb+order+1 < len(self.knots_) and self.knots_[knotsNb+order+1] != self.knots_[knotsNb+1]):
          res +=  -self(knotsNb+1, order-1,x) / (self.knots_[knotsNb+order+1] - self.knots_[knotsNb+1]) 
        return order * res

      elif derivative == 2:
        res = NP.zeros(x.shape)
        if (knotsNb+order+1 < len(self.knots_) and self.knots_[knotsNb+order+1] != self.knots_[knotsNb+1] and self.knots_[knotsNb+order+1] != self.knots_[knotsNb+2]):
          res += self(knotsNb+2, order-2,x) / ((self.knots_[knotsNb+order+1] - self.knots_[knotsNb+1]) * (self.knots_[knotsNb+order+1] - self.knots_[knotsNb+2]) )
        if (knotsNb+order-1 < len(self.knots_) and self.knots_[knotsNb+order] != self.knots_[knotsNb+1]):
          tmp = -self(knotsNb+1, order-2,x) / (self.knots_[knotsNb+order] - self.knots_[knotsNb+1]) 
          if self.knots_[knotsNb+order+1] != self.knots_[knotsNb+1]:
            res += tmp / (self.knots_[knotsNb+order+1] - self.knots_[knotsNb+1])
          if  self.knots_[knotsNb+order] != self.knots_[knotsNb]:
            res +=  tmp / (self.knots_[knotsNb+order] - self.knots_[knotsNb])
        if (knotsNb+order < len(self.knots_) and self.knots_[knotsNb+order] != self.knots_[knotsNb] and self.knots_[knotsNb+order-1] != self.knots_[knotsNb]):
          res += self(knotsNb, order-2,x) / ((self.knots_[knotsNb+order] - self.knots_[knotsNb]) * (self.knots_[knotsNb+order-1] - self.knots_[knotsNb]) )
        return order * (order-1) * res

      else:
        raise NotImplementedError('Only up to second order derivatives of the B spline are implemented')
