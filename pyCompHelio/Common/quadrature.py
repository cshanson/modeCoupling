import numpy as NP
from   .misc  import *

class quadrature(object):

    ''' class storing quadrature points and weights
        for later integration
    '''
   
    # Newton-Cotes weights for orders until 7
    wNC_ = [\
     [1.],\
     [1., 1.],\
     [1./3., 4./3., 1./3.],\
     [1./4., 3./4., 3./4., 1./4.],\
     [7./45., 32./45., 12./45., 32./45., 7./45.],\
     [19./144., 75/144., 50./144., 50./144., 75./144., 19./144.],\
     [41./420., 216./420., 27./420., 272./420., 27./420., 216./420., 41./420.],\
     [751./8640., 3577./8640., 1323./8640., 2989./8640., 2989./8640., 1323./8640., 3577./8640., 751./8640.],\
    ]


    def __init__(self,x=None,qType='SIMPS',xmin=None,xmax=None,N=100):

      ''' points can be on a given grid (classical quadrature formulae)
          or placed to follow Gauss quadrature methods

          If a Gauss quadrature is selected, quad points are computed 
          within each subinterval of x
      '''

      if qType == 'SUM':
        qType = 'NEWTON_COTES 0'
      elif qType == 'TRAPZ':
        qType = 'NEWTON_COTES 1'
      elif qType == 'SIMPS':
        qType = 'NEWTON_COTES 2'

      options = qType.split() 
      self.type_ = options[0]
      try:
        self.order_ = int(options[1])
      except:
        self.order_ = 2

      
      # TODO
      # Newton-Cotes formulae for non evenly spaced points...

      if self.type_ == 'NEWTON_COTES':
        # Points
        if x is not None: 
          self.x_ = x
          # Check if interval is evenly partitioned 
          # and correct if necessary
          self.dx_ = x[1]-x[0]
          diff  = NP.diff(self.x_)
          check = abs(diff-self.dx_)/self.dx_ > 1.e-8
          if ( len(diff[check]) != 0 ):
            self.x_ = NP.linspace(x[0],x[-1],N)
        else:
          if (xmin is None or xmax is None):
            raise Exception("Please provide correct points (either x or xmin and xmax)")
          self.x_ = NP.linspace(xmin,xmax,N)

        self.w_ = NP.zeros(self.x_.shape)

        # Determine where the requested order formula stops
        # the remaining points are used with the best order possible

        if self.order_ == 0:
          self.w_[0   ] = 0.5*self.dx_
          self.w_[-1  ] = 0.5*self.dx_
          self.w_[1:-1] = self.dx_

        else:

          Npts       = len(self.x_)
          limitIndex = self.order_*(int(NP.floor((Npts-1-self.order_)/self.order_)))
          Nrmng      = Npts-limitIndex-self.order_

          if Nrmng == 1:
            for i in range(self.order_+1):
              self.w_[i:limitIndex+i+1:self.order_] += quadrature.wNC_[self.order_][i]*self.dx_*self.order_/2.
          else:
            # Average between remaining points at the end end at the beginning
            # after
            for i in range(self.order_+1):
              self.w_[i:limitIndex+i+1:self.order_] += quadrature.wNC_[self.order_][i]*self.dx_*self.order_/4.
            for i in range(Nrmng):
              self.w_[limitIndex+self.order_+i] += quadrature.wNC_[Nrmng-1][i]*self.dx_*(Nrmng-1)/4.
            # before
            for i in range(Nrmng):
              self.w_[i] += quadrature.wNC_[Nrmng-1][i]*self.dx_*(Nrmng-1)/4.
            for i in range(self.order_+1):
              if i==0:
                self.w_[Nrmng-1:-1:self.order_] += quadrature.wNC_[self.order_][i]*self.dx_*self.order_/4.
              else:
                self.w_[i+Nrmng-1::self.order_] += quadrature.wNC_[self.order_][i]*self.dx_*self.order_/4.

      # Gauss quadratures
      else:
        raise NotImplemented("Gauss quadratures not implemented yet")
         
    def __call__(self,data,axis=-1):
      return self.integrate(data,axis)
 
    def integrate(self,data,axis=-1):
      ''' returns w*f '''

      if data.shape[axis] != len(self.w_):
        raise Exception("Incompatible shapes along axis %d. Data: %d, weights: %d"%(axis,data.shape[axis],len(self.w_)))

      if self.type_ == 'NEWTON_COTES':
        if axis != -1 or axis != data.ndim-1:
          return NP.sum(NP.rollaxis(data,axis,data.ndim)*self.w_,axis=-1)
        else:
          return NP.sum(data*self.w_,axis=-1)
      # Gauss quadratures
      else:
        raise NotImplemented("Gauss quadratures not implemented yet")



