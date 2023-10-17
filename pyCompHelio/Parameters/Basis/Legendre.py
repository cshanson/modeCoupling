from .basis1D import *

#########################################################

class Legendre(basis1D):
  '''Class for 1d Legendre basis. It contains the vector of ls and x (theta).'''  
 
  def __init__(self, x, ls, normalized=True, nbCores=1, pgBar=False):
    self.x_          = x # x = cos(theta)
    self.order_      = 0 # just for compatibility with other basis functions
    self.ls_         = NP.array(ls)
    self.nbBasisFunctions_ = len(ls)
    self.normalized_ = normalized
    self.nbCores_    = nbCores
    self.pgBar_      = pgBar


  def __call__(self, l, x,derivative=0):
    '''Basic legendre polynomial computations.
    It is not recommended to use them in a loop in l, l/m.
    Beware that x is theta and not cos(theta) 
    to be in agreement with other basis functions.'''

    if NP.amin(x) < -1.0 or NP.amax(x) > 1.0:
      raise ValueError('abscissa <-1 or >1')
  
    if self.normalized_:
      if hasattr(x,'shape'):
        p_l = 1/NP.sqrt(2.0)*NP.ones(x.shape)
      else:
        p_l = 1/NP.sqrt(2.0)*NP.ones((1,))
    else:
      if hasattr(x,'shape'):
        p_l = 1.0*NP.ones(x.shape)
      else:
        p_l = 1.0*NP.ones((1,))
    if l == 0:
      if derivative==1:
        return NP.zeros(p_l.shape)
      else:
        return p_l

    # Term l=1
    if self.normalized_:
      p_ll = x*NP.sqrt(1.5) 
    else:
      p_ll = p_l*x

    if derivative != 0:
      if self.normalized_:
        dp_ll = NP.sqrt(1.5)*NP.ones(p_ll.shape)
      else:
        dp_ll = NP.ones(p_ll.shape)

    if l == 1:
      if derivative == 1:
        return dp_ll
      else:
        return p_ll
    # Recursion
    for L in range(2,l+1):
      result = self.iterLegendre(L,x,p_ll,p_l)
      if derivative == 1:
        resD = self.iterLegendreDerivative(L,x,result,p_ll)    
      p_l  = p_ll
      p_ll = result

    if derivative == 1:
      return resD
    else:
      return result


  def iterLegendre(self,L,x,PLm1,PLm2):
    ''' given P_(L-1) and P_(L-2), returns P_(L)'''

    l = float(L)
    if self.normalized_:
      result = (2*l-1)/l*NP.sqrt((2*l+1)/(2*l-1))*x*PLm1 - (l-1)/l*NP.sqrt((2*l+1)/(2*l-3.0))*PLm2
    else:
      result = (x*(2*l-1)*PLm1 - (l-1)*PLm2)/l
    return result

  def iterLegendreDerivative(self,L,x,PL,PLm1):
    ''' given P_(L-1) and P_(L), returns P'_(L)'''

    l = float(L)
    if self.normalized_:
      result        = -l*(x*PL-NP.sqrt((2*l+1.0)/(2*l-1.0))*PLm1)/(1.0-x*x)
      result[x== 1] = l*(l+1)/2. * NP.sqrt((2*l+1.e0)/2.e0)
      result[x==-1] =  (-1)**(L+1)*l*(l+1)/2. * NP.sqrt((2*l+1.e0)/2.e0)
    else:
      result        = -l*(x*PL-PLm1)/(1-x*x)
      result[x== 1] = l*(l+1)/2.
      result[x==-1] = (-1)**(L+1)*l*(l+1)/2.
   
    return result

  def projectOnBasis(self,data,axis=-1):
    '''Returns the coefficients of data on the Legendre basis. The data contained in axis will be projected.'''

    # Put data in the correct shape
    data = NP.array(data)
    data = NP.moveaxis(data,axis,-1) # theta is now the last dimension
    dataShape = data.shape
    data = NP.reshape(data, (int(NP.prod(data.shape[:-1])), data.shape[-1])) # data is now N,theta where N contains all the extra dimensions

    # Project on the Legendre basis
    isParallelizable = (data.ndim > 1 and data.shape[0] > self.nbCores_)
    if self.nbCores_ != 1 and isParallelizable: # parallel version    
      data = NP.moveaxis(data,0,-1) # data is now theta,N
      dataL = reduce(projectOnBasisSerial, (self,data[NP.newaxis,...],False), data.shape[-1], self.nbCores_, None, progressBar=self.pgBar_) # dataL is 1,ls,N
      dataL = NP.squeeze(dataL) # dataL is ls,N
      dataL = NP.moveaxis(dataL,-1,0) # data is now N,ls

    else:
      dataL = projectOnBasisSerial(self,data,self.pgBar_) # dataL is N,ls

    # Put back dataL in the correct shape
    dataL = NP.reshape(dataL, (list(dataShape[:-1]) + [len(self.ls_)])) # data is now ...,ls
    dataL = NP.moveaxis(dataL,-1,axis) # put back ls at the initial place of theta
    return dataL


  def reconstructFromBasis(self,dataL,x=None,axis=-1,sumDerivatives=False):
    ''' Reconstruction from Legendre polynomial coefficients to actual function in x (cos th). If not given, it uses the theta of the class.
      It return \sum_l dataL P_l(x).
      If sumDerivatives, it also returns \sum_l dataL P_l'(x).
      Several dataL arrays can be given in a list (not tuple!)
    '''
    if x is None:
      x = self.x_

    # Put data in the correct shape
    dataL = NP.array(dataL)
    dataL = NP.moveaxis(dataL,axis,-1) # l is now the last dimension
    dataShape = dataL.shape
    dataL = NP.reshape(dataL, (int(NP.prod(dataL.shape[:-1])), dataL.shape[-1])) # data is now N,l where N contains all the extra dimensions

    # Reconstruct from the Legendre coefficients
    isParallelizable = (dataL.ndim > 1 and dataL.shape[0] > self.nbCores_)
    if self.nbCores_ != 1 and isParallelizable: # parallel version    
      dataL = NP.moveaxis(dataL,0,-1) # data is now theta,N
      data = reduce(reconstructFromBasisSerial, (self,dataL[NP.newaxis,...],x,sumDerivatives,False), dataL.shape[-1], self.nbCores_, None, progressBar=self.pgBar_) # data is 1,theta,N
      data = NP.squeeze(data) # data is theta,N
      data = NP.moveaxis(data,-1,0) # data is now N,theta

    else:
      data = reconstructFromBasisSerial(self,dataL,x,sumDerivatives,self.pgBar_) # dataL is N,theta

    # Put back dataL in the correct shape
    data = NP.reshape(data, (list(dataShape[:-1]) + [len(x)])) # data is now ...,x
    data = NP.moveaxis(data,-1,axis) # put back theta at the initial place of l
    return data


  def createSmoothnessMatrix(self, smoothnessOrder=0, BCleft=None, BCright=None, r=None):
    '''The smoothness matrix used in the inversion contains \int P_l P_l'.'''
    if smoothnessOrder==0:
      if self.normalized_:
        return NP.eye(len(self.ls_))
      else:
        return NP.diag(2./(2*ls+1))
    else:
      raise NotImplementedError('The smoothness matrix of Legendre basis is only implemented at order 0')


def projectOnBasisSerial(self,data,pgBar):
  '''Returns the coefficients of data on the Legendre basis. The data contained in the last axis will be projected (the moveaxis is already done in projectOnBasis).'''
  #th = self.x_
  x  = self.x_
  th = NP.arccos(self.x_)
  ls = self.ls_
  Nl    = len(self.ls_)
  data = NP.array(data)

  PLm2 = self(0,x)
  PLm1 = self(1,x) 
  l_now = 2

  res = NP.zeros((data.shape[0],Nl),dtype=data.dtype)
  if pgBar:
    PB = progressBar(Nl,'serial')

  for i in range(Nl):
    # Get correct legendre polynomial
    L = self.ls_[i]
    if L==0:
      Lgdr = PLm2
    elif L==1:
      Lgdr = PLm1
    else:
      while(l_now <= L):
        PL     = self.iterLegendre(l_now,x,PLm1,PLm2)
        PLm2   = PLm1
        PLm1   = PL
        l_now += 1
      Lgdr = PL

    # Project input on Pl
    for iData in range(data.shape[0]):
      res[iData,i] = simps(data[iData]*Lgdr*NP.sin(th),th)\
                       /simps(Lgdr**2*NP.sin(th),th)

    if pgBar:
      PB.update()
  if pgBar:
    del PB

  return res

def reconstructFromBasisSerial(self,data,x,sumDerivatives=False,pgBar=False):
  '''Reconstruct a function from its Legendre coefficients. It supposes that the l component is the last axis (the moveaxis is already done in reconstructFormBasis).'''

  PLm2  = self(0,x)
  PLm1  = self(1,x)
  l_now = 2
  res = NP.zeros((data.shape[0],len(x)),dtype=data.dtype)
  resD = NP.zeros((data.shape[0],len(x)),dtype=data.dtype)
  if pgBar:
    PB = progressBar(len(self.ls_),'serial')
  for i in range(len(self.ls_)):
    # Get correct legendre polynomial
    L = self.ls_[i]
    if L==0:
      Lgdr = PLm2
      if sumDerivatives:
        dPL = NP.zeros(PLm2.shape)
    elif L==1:
      Lgdr = PLm1
      if sumDerivatives:
        dPL = NP.ones(PLm1.shape)
        if normalized:
          dPL = dPL*NP.sqrt(1.5)
    else:
      while(l_now <= L):
        PL     = self.iterLegendre(l_now,x,PLm1,PLm2)
        if sumDerivatives:
          dPL  = self.iterLegendreDerivative(l_now,x,PL,PLm1)
        PLm2   = PLm1
        PLm1   = PL
        l_now += 1
      Lgdr = PL
      
    # Add contribution l
    for iCoeffs in range(data.shape[0]):
      res[iCoeffs] += data[iCoeffs][...,i,NP.newaxis]*Lgdr
      if sumDerivatives:
        resD[iCoeffs] += data[iCoeffs][...,i,NP.newaxis]*dPL

    if pgBar:
      PB.update()
  if pgBar:
    del PB

  if sumDerivatives:
    return res,resD
  else:
    return res
