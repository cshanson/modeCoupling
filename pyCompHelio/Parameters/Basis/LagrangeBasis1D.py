from .basis1D import *

#########################################################

class LagrangeBasis1D(basis1D):
  '''class for Lagrange polynomial basis function of given order and defined on the mesh given by points.'''
  def __init__(self, x, points, order):
    basis1D.__init__(self, x)   
    if points[-1] != x[-1]:
      points = NP.append(points,x[-1])     
    self.points_ = points
    self.order_ = order
    self.nbBasisFunctions_ = (len(points)-1) * self.order_+1
    self.quadrature_ = True

    # Calculate Gauss-Lobatto nodes and their weigths
    nodes,w,P = getGaussLobattoNodes(self.order_+1,0,1)
    self.nodes_ = nodes
    self.w_ = w 

    self.computeMassMatrix() 


  def __call__(self, i, x=None, derivative=0):
    '''Returns the value of the i^{th} Lagrange interpolation basis function \phi_i(r).'''
    if x is None:
      x = self.x_

    res = NP.zeros(len(x))
    for k in range(len(x)):
      intervalId = self.getInterval(x[k])
      if i >= self.order_ * intervalId and i <= self.order_ * (intervalId + 1):
        xRef = self.getLocalPosition(intervalId, x[k])
        res[k] = self.evaluateRef(i - self.order_ * intervalId, xRef, derivative) / (self.points_[intervalId+1] - self.points_[intervalId])**derivative
    return res

  def getLocalPosition(self, i, x):
    '''Returns the local position of the point x on the interval [x_i, x_{i+1}].'''
    return (x - self.points_[i]) / (self.points_[i+1] - self.points_[i])

  def getGlobalPosition(self, i, x):
    '''Returns the global position of the point x defined on [0,1] on the interval [x_i, x_{i+1}].'''   
    return x * (self.points_[i+1] - self.points_[i]) + self.points_[i]

  def getGlobalNodes(self):
    '''Returns an array containing all the global nodes corresponding to the quadrature points.'''
    nodes = NP.zeros(self.nbBasisFunctions_)
    for i in range(len(self.points_)-1):
      globalNodes = self.getGlobalPosition(i, self.nodes_)
      nodes[i*self.order_:(i+1)*self.order_]= globalNodes[:-1] # to avoid to have a double node
    nodes[-1] = globalNodes[-1] # add the last point
    return nodes

  def getInterval(self, x):
    '''Returns the interval containing point x.'''
    i = 0
    while i < len(self.points_) and x > self.points_[i]:
      i += 1
    return i-1

  def evaluateRef(self, i, x, derivative = 0):
    '''Returns the value of the basis function \phi_i or its derivatives (up to order 2) at the point x defined on the reference element.'''
    res = 1.
    for j in range(self.order_+1):
      if j != i:
        res *= (x - self.nodes_[j]) / (self.nodes_[i] - self.nodes_[j])

    if derivative == 1:
      # L_i'(x) = L_i(x) \sum_{m=0, m \neq i} 1 / (x - x_m)
      res = 0.
      for m in range(self.order_+1):
        if m != i:
          resM = 1.
          for j in range(self.order_+1):
            if j != i and j != m:
              resM *= (x - self.nodes_[j])
          res += resM  

      fac = 1.  
      for m in range(self.order_+1):
        if m != i:   
          fac *= 1. / (self.nodes_[i] - self.nodes_[m])
      res *= fac

    elif derivative == 2:
      # L_i''(x) = L_i(x) ( (\sum_{m=0, m \neq i} 1 / (x - x_m))^2 + \sum_{m=0, m \neq i} 1 / (x - x_m)^2 )
      fac1 = 0.
      fac2 = 0.
      for m in range(self.order_+1):
        if m != i:
          fac1 += 1. / (x - self.nodes_[m])
          fac2 += 1. / (x - self.nodes_[m])**2
      res *= (fac1**2 + fac2)

    elif derivative > 2:
      raise NotImplementedError('Only the first derivative of the Lagrange polynomials are implemented.')
    return res



def getGaussLobattoNodes(N,a=-1,b=1):
  '''Returns the Gauss-Lobatto nodes of order N by computing the roots of P_{N-1}'. Returns the nodes x, the weigths for the quadrature w and the Legendre Vandermonde Matrix P.'''

  x  = -NP.cos(NP.pi*NP.linspace(0,1,N))
  P  = NP.zeros((N,N))
  dP = NP.zeros(N)
  xold=2
  eps = 1.e-10
  while NP.amax(NP.abs(x-xold))>eps:
    xold=x 
    # P contains the Legendre polynomials P_n(x) for n=0,...,N       
    P[:,0]=1
    P[:,1]=x
    for k in range(1,N-1):
      P[:,k+1] =( (2*k+1)*x*P[:,k]-k*P[:,k-1] )/ (k+1)

    # Derivative of the Legendre polynomial P'_N(x)
    dP[1:-1] = (N-1) * (x[1:-1]*P[1:-1,N-1]-P[1:-1,N-2] ) / (x[1:-1]**2-1) 
    dP[0]    = N * (N+1) / 2.
    dP[-1]   = (-1)**(N+1) * N * (N+1) / 2.

    # Newton-Raphson to find the zeros of P'
    x= xold - (N-1) * (x*P[:,N-1]-P[:,N-2]) / (-2.*x*dP + N*(N-1)*P[:,N-1])

  # Convert from [-1.1] to the interval [a,b]
  x = a + (x+1.) / 2. * (b-a)

  w=(b-a)/((N-1)*N*P[:,N-1]**2)
  return x,w,P
