# Finite differences on non-uniform grids

# Sparse matrices
import numpy as NP
from scipy.sparse        import lil_matrix
from scipy.sparse.linalg import spsolve

############################################################################################################################
############################################################################################################################


class FDM_LInterp:

# A class that computes derivatives of pointwise data up to a given degree, 
# with a user-defined order (that must lesser than the number of points - maximum derivative order)

# Important members
#    X_ : abscissa where the data is given
#    N_ : number of points
#    M_ : requested maximum order of derivation, instantiated at the begining
#    w_ : weights : array of 2D sparse matrices corresponding to the finite difference
#         coefficients to compute each derivative
#         as MATLAB does not handle 3D sparse matrices, the matrix will be stored by blocks, all m matrices being in a row
#    o_ : if order is odd, the formula is decentered (eg order 1 : f(i+1)-f(i))
#    max_ratio : evaluates how uniform a grid is
#


  ###############
  # Constructor #
  ###############

  def __init__(self,X,M=1,order=1):

    if (order <=0 ):
      print ('Error : order must be >= 1\n')
      return

    self.X_ = X
    self.N_ = len(X)
    self.M_ = M
    self.o_ = order
    self.w_ = lil_matrix((self.N_,M*self.N_))
    if (order > self.N_-M):
      print ('Error : impossible to compute %d-th derivative at requested order %d with %d points.\n',M,order,self.N_)
      print ('Order is set to %d\n',self.N_-M)
      self.fdm.o_ = self.N_-M
      if (order <=0 ):
        print ('Error : order must be >= 1\n')
        return

    self = self.Compute_weights(M)

  ###########################
  # Compute FD coefficients #
  ###########################

  def Compute_weights(self,M):
    
    self.max_ratio = -1337e42
   
    # Determine stencil length
    Ns = M + self.o_

    # Compute each line of the matrices
    for i in range(self.N_):

      # ===================================================
      # Get stencil : ix0 : index of first point in stencil

      # Number of points to the left of the base point (xi)
      if (Ns%2 == 0):
        Nsleft = Ns/2 - 1 
      else:
        Nsleft = (Ns-1)/2
      # First index of the stencil
      ix0 = int(min(max(0,i-Nsleft),self.N_-Ns))
 
      # Determine largest ratio
      minspacing =  1664
      maxspacing = -1664
      for k in range(1,Ns):
        spacing = self.X_[ix0+k]-self.X_[ix0+k-1]
        minspacing = min(spacing,minspacing)
        maxspacing = max(spacing,maxspacing)
      ratio = maxspacing/minspacing
      self.max_ratio = max(ratio,self.max_ratio)
     
      # Build a temporary matrix, its last lines being the coefficients to add in the global matrix
      A = NP.zeros((M+1,Ns,Ns))

      c1 = 1
      A[0,0,0] = 1
     
      for n in range (1,Ns):
        c2 = 1
       
        for nu in range(n):
          c3 = self.X_[ix0+n]-self.X_[ix0+nu]
          c2 = c2*c3
          if (n <= M):
            A[n,n-1,nu] = 0
         
          for m in range(min(n,M)+1):
            if (m==0):
              A[m,n,nu] = ((self.X_[ix0+n]-self.X_[i])*A[m,n-1,nu] )/c3
            else:
              A[m,n,nu] = ((self.X_[ix0+n]-self.X_[i])*A[m,n-1,nu]-m*A[m-1,n-1,nu])/c3
          
        for m in range(min(n,M)+1):
          if (m==0):
            A[m,n,n] = -c1/c2*((self.X_[ix0+n-1]-self.X_[i])*A[m,n-1,n-1])
          else:
            A[m,n,n] =  c1/c2*(m*A[m-1,n-1,n-1] - (self.X_[ix0+n-1]-self.X_[i])*A[m,n-1,n-1])

        c1 = c2
    
      # Store coefficients in big matrix
      for m in range(1,M):
        istart = ix0+(m-1)*self.N_
        for k in range(Ns):
          self.w_[i,istart+k] = A[m,-1,k]

    return self

  #####################
  # return derivative #
  #####################

  def Compute_derivative(self,U,m=1):
    tmpmat = lil_matrix(self.w_[:,(m-1)*self.N_:m*self.N_])
    res = tmpmat.tocsr().dot(U)
    return res[:,0]

  def __call__(self,u,m=1):
    return self.Compute_derivative(u,m=1)


############################################################################################################################
############################################################################################################################

class FDM_Compact:

# A class that computes first and second derivatives of pointwise data
# using compact finite differences scheme
# See Gamet, 1999 (Int J Numer Meth Fluids)

#
#    X_      : abscissa where the data is given
#    N_      : number of points
#    A1_     : implicit matrix for first derivative
#    B1_     : explicit side matrix
#    A2_,B2_ : second derivative
#  

  ###############
  # Constructor #
  ###############

  def __init__(self,X,BC=None):

    self.X_ = X
    self.N_ = len(X)
      
    N = self.N_
    
    self.A1_ = lil_matrix((N,N),dtype=complex)
    self.B1_ = lil_matrix((N,N),dtype=complex)
    self.A2_ = lil_matrix((N,N),dtype=complex)
    self.B2_ = lil_matrix((N,N),dtype=complex)
      
    ############################  
    # Central points (2<i<N-1) #
    ############################

    for i in range(2,N-2):

      him1 = X[i-1]-X[i-2]
      hi   = X[i  ]-X[i-1]
      hip1 = X[i+1]-X[i  ]
      hip2 = X[i+2]-X[i+1]

      #
      # First derivative 
      #

      alpha = 1.0/3.0
      beta  = 1.0/3.0

      self.A1_[i,i-1] = alpha
      self.A1_[i,i  ] = 1.0
      self.A1_[i,i+1] = beta

      # A_i coefficient in the paper
      self.B1_[i,i+1] = ( him1*hi*hip1 + hi*hi*hip1 + him1*hi*hip2 + hi*hi*hip2 -him1*hi*hi*alpha - him1*hi*hip1*alpha\
                        - him1*hi*hip2*alpha - him1*hi*hip1*beta - hi*hi*hip1*beta - him1*hip1*hip1*beta -2.0*hi*hip1*hip1*beta - hip1*hip1*hip1*beta\
                        + him1*hi*hip2*beta + hi*hi*hip2*beta + 2.0*him1*hip1*hip2*beta + 4.0*hi*hip1*hip2*beta + 3.0*hip1*hip1*hip2*beta)/\
                        (hip1*(hi+hip1)*(him1+hi+hip1)*hip2)

      # B_i coefficient
      self.B1_[i,i-1] = ( -him1*hip1*hip1 - hi*hip1*hip1 - him1*hip1*hip2 - hi*hip1*hip2 - 3.0*him1*hi*hi*alpha - 4.0*him1*hi*hip1*alpha\
                       + hi*hi*hi*alpha + 2.0*hi*hi*hip1*alpha -him1*hip1*hip1*alpha + hi*hip1*hip1*alpha -2.0*him1*hi*hip2*alpha -him1*hip1*hip2*alpha\
                       + hi*hi*hip2*alpha + hi*hip1*hip2*alpha + him1*hip1*hip2*beta + hi*hip1*hip2*beta + hip1*hip1*hip2*beta )/\
                       (him1*hi*(hi+hip1)*(hi+hip1+hip2))

      # C_i coefficient
      self.B1_[i,i+2] = ( -him1*hi*hip1 - hi*hi*hip1 + him1*hi*hi*alpha + him1*hi*hip1*(alpha+beta) + hi*hi*hip1*beta\
                       + him1*hip1*hip1*beta + 2.0*hi*hip1*hip1*beta + hip1*hip1*hip1*beta)\
                       / (hip2*(hip1+hip2)*(hi+hip1+hip2)*(him1+hi+hip1+hip2))

      # D_i coefficient
      self.B1_[i,i-2] = ( hi*hip1*hip1 + hi*hip1*hip2 - hi*hi*hi*alpha -2.0*hi*hi*hip1*alpha - hi*hip1*hip1*alpha - hi*hi*hip2*alpha - hi*hip1*hip2*alpha\
                       - hi*hip1*hip2*beta - hip1*hip1*hip2*beta )\
                       / (him1*(him1+hi)*(him1+hi+hip1)*(him1+hi+hip1+hip2))

      # easy E_i
      self.B1_[i,i] = - (self.B1_[i,i-2] + self.B1_[i,i-1] + self.B1_[i,i+1] + self.B1_[i,i+2])

      #
      # Second derivative 
      #

      alpha = 2.0/11.0
      beta  = 2.0/11.0

      self.A2_[i,i-1] = alpha
      self.A2_[i,i  ] = 1.0
      self.A2_[i,i+1] = beta

      # A_i coefficient
      self.B2_[i,i+1] = 2.0*( -him1*hi - hi*hi + him1*hip1 + 2.0*hi*hip1 + him1*hip2 + 2.0*hi*hip2 + 2.0*him1*hi*alpha - hi*hi*alpha\
                            + him1*hip1*alpha - hi*hip1*alpha + him1*hip2*alpha - hi*hip2*alpha -him1*hi*beta - hi*hi*beta\
                            - 2.0*him1*hip1*beta - 4.0*hi*hip1*beta - 3.0*hip1*hip1*beta + him1*hip2*beta + 2.0*hi*hip2*beta + 3.0*hip1*hip2*beta)\
                            / (hip1*hip2*(hi+hip1)*(him1+hi+hip1))

      # B_i coefficient
      self.B2_[i,i-1] = 2.0*( 2.0*him1*hip1 + 2.0*hi*hip1 -hip1*hip1 +him1*hip2 + hi*hip2 -hip1*hip2 + 2.0*him1*hip1*alpha\
                            - 3.0*hi*hi*alpha + 3.0*him1*hi*alpha - 4.0*hi*hip1*alpha - hip1*hip1*alpha + him1*hip2*alpha -2.0*hi*hip2*alpha -hi*hip1*beta\
                            - him1*hip1*beta - hip1*hip2*alpha - hip1*hip1*beta + him1*hip2*beta + hi*hip2*beta + 2.0*hip1*hip2*beta)\
                           / (him1*hi*(hi+hip1)*(hi+hip1+hip2))

      # C_i coefficient
      self.B2_[i,i+2] = 2.0*( him1*hi + hi*hi - him1*hip1 -2.0*hi*hip1 -2.0*him1*hi*alpha + hi*hi*alpha -him1*hip1*alpha\
                            + hi*hip1*alpha + him1*hi*beta + hi*hi*beta + 2.0*him1*hip1*beta + 4.0*hi*hip1*beta + 3.0*hip1*hip1*beta)\
                            / (hip2*(hip1+hip2)*(hi+hip1+hip2)*(him1+hi+hip1+hip2))

      # D_i coefficient
      self.B2_[i,i-2] = 2.0*( -2.0*hi*hip1 + hip1*hip1 - hi*hip2 + hip1*hip2 + 3*hi*hi*alpha + 4.0*hi*hip1*alpha + hip1*hip1*alpha\
                            + 2.0*hi*hip2*alpha + hip1*hip2*alpha + hi*hip1*beta + hip1*hip1*beta - hi*hip2*beta - 2.0*hip1*hip2*beta)\
                            / (him1*(him1+hi)*(him1+hi+hip1)*(him1+hi+hip1+hip2))
 
      # E_i
      self.B2_[i,i] = - (self.B2_[i,i-2] + self.B2_[i,i-1] + self.B2_[i,i+1] + self.B2_[i,i+2])

    ######################
    # Boundary treatment #
    ######################

    #
    # PERIODIC f0 = fN-1
    #

    if (BC == 'Periodic'):

      # Copypasta from central points with appropriate indices
      for i in [0,1,N-2,N-1]:
  
        if (i==0 or i==N-1):
          IM1  = N-2
          IM2  = N-3
          IP1  = 1
          IP2  = 2
          him1 = X[N-2]-X[N-3]
          hi   = X[N-1]-X[N-2]
          hip1 = X[1  ]-X[0  ]
          hip2 = X[2  ]-X[1  ]
        elif (i==1):
          IM1  = 0
          IM2  = N-2
          IP1  = 2
          IP2  = 3
          him1 = X[N-1]-X[N-2]
          hi   = X[1  ]-X[0  ]
          hip1 = X[2  ]-X[1  ]
          hip2 = X[3  ]-X[2  ]
        elif (i==N-2):
          IM1  = N-3
          IM2  = N-4
          IP1  = N-1
          IP2  = 1
          him1 = X[N-3]-X[N-4]
          hi   = X[N-2]-X[N-3]
          hip1 = X[N-1]-X[N-2]
          hip2 = X[1  ]-X[0  ]
  
        
        # First derivative 
  
        alpha = 1.0/3.0
        beta  = 1.0/3.0
  
        self.A1_[i,IM1] = alpha
        self.A1_[i,i  ] = 1.0
        self.A1_[i,IP1] = beta
  
        # A_i coefficient in the paper
        self.B1_[i,IP1] = ( him1*hi*hip1 + hi*hi*hip1 + him1*hi*hip2 + hi*hi*hip2 -him1*hi*hi*alpha - him1*hi*hip1*alpha\
                         - him1*hi*hip2*alpha - him1*hi*hip1*beta - hi*hi*hip1*beta - him1*hip1*hip1*beta -2.0*hi*hip1*hip1*beta - hip1*hip1*hip1*beta\
                         + him1*hi*hip2*beta + hi*hi*hip2*beta + 2.0*him1*hip1*hip2*beta + 4.0*hi*hip1*hip2*beta + 3.0*hip1*hip1*hip2*beta)/\
                         (hip1*(hi+hip1)*(him1+hi+hip1)*hip2)
  
        # B_i coefficient
        self.B1_[i,IM1] = ( -him1*hip1*hip1 - hi*hip1*hip1 - him1*hip1*hip2 - hi*hip1*hip2 - 3.0*him1*hi*hi*alpha - 4.0*him1*hi*hip1*alpha\
                         + hi*hi*hi*alpha + 2.0*hi*hi*hip1*alpha -him1*hip1*hip1*alpha + hi*hip1*hip1*alpha -2.0*him1*hi*hip2*alpha -him1*hip1*hip2*alpha\
                         + hi*hi*hip2*alpha + hi*hip1*hip2*alpha + him1*hip1*hip2*beta + hi*hip1*hip2*beta + hip1*hip1*hip2*beta )/\
                         (him1*hi*(hi+hip1)*(hi+hip1+hip2))
  
        # C_i coefficient
        self.B1_[i,IP2] = ( -him1*hi*hip1 - hi*hi*hip1 + him1*hi*hi*alpha + him1*hi*hip1*(alpha+beta) + hi*hi*hip1*beta\
                         + him1*hip1*hip1*beta + 2.0*hi*hip1*hip1*beta + hip1*hip1*hip1*beta)\
                         / (hip2*(hip1+hip2)*(hi+hip1+hip2)*(him1+hi+hip1+hip2))
  
        # D_i coefficient
        self.B1_[i,IM2] = ( hi*hip1*hip1 + hi*hip1*hip2 - hi*hi*hi*alpha -2.0*hi*hi*hip1*alpha - hi*hip1*hip1*alpha - hi*hi*hip2*alpha - hi*hip1*hip2*alpha\
                         - hi*hip1*hip2*beta - hip1*hip1*hip2*beta )\
                         / (him1*(him1+hi)*(him1+hi+hip1)*(him1+hi+hip1+hip2))
  
        # easy E_i
        self.B1_[i,i] = - (self.B1_[i,IM2] + self.B1_[i,IM1] + self.B1_[i,IP1] + self.B1_[i,IP2])
  
        # Second derivative 
  
        alpha = 2.0/11.0
        beta  = 2.0/11.0
  
        self.A2_[i,IM1] = alpha
        self.A2_[i,i  ] = 1.0
        self.A2_[i,IP1] = beta
  
        # A_i coefficient
        self.B2_[i,IP1] = 2.0*( -him1*hi - hi*hi + him1*hip1 + 2.0*hi*hip1 + him1*hip2 + 2.0*hi*hip2 + 2.0*him1*hi*alpha - hi*hi*alpha\
                             + him1*hip1*alpha - hi*hip1*alpha + him1*hip2*alpha - hi*hip2*alpha -him1*hi*beta - hi*hi*beta\
                             - 2.0*him1*hip1*beta - 4.0*hi*hip1*beta - 3.0*hip1*hip1*beta + him1*hip2*beta + 2.0*hi*hip2*beta + 3.0*hip1*hip2*beta)\
                             / (hip1*hip2*(hi+hip1)*(him1+hi+hip1))
  
        # B_i coefficient
        self.B2_[i,IM1] = 2.0*( 2.0*him1*hip1 + 2.0*hi*hip1 -hip1*hip1 +him1*hip2 + hi*hip2 -hip1*hip2 + 2.0*him1*hip1*alpha\
                             - 3.0*hi*hi*alpha + 3.0*him1*hi*alpha - 4.0*hi*hip1*alpha - hip1*hip1*alpha + him1*hip2*alpha -2.0*hi*hip2*alpha -hi*hip1*beta\
                             - him1*hip1*beta - hip1*hip2*alpha - hip1*hip1*beta + him1*hip2*beta + hi*hip2*beta + 2.0*hip1*hip2*beta)\
                            / (him1*hi*(hi+hip1)*(hi+hip1+hip2))
  
        # C_i coefficient
        self.B2_[i,IP2] = 2.0*( him1*hi + hi*hi - him1*hip1 -2.0*hi*hip1 -2.0*him1*hi*alpha + hi*hi*alpha -him1*hip1*alpha\
                             + hi*hip1*alpha + him1*hi*beta + hi*hi*beta + 2.0*him1*hip1*beta + 4.0*hi*hip1*beta + 3.0*hip1*hip1*beta)\
                             / (hip2*(hip1+hip2)*(hi+hip1+hip2)*(him1+hi+hip1+hip2))
  
        # D_i coefficient
        self.B2_[i,IM2] = 2.0*( -2.0*hi*hip1 + hip1*hip1 - hi*hip2 + hip1*hip2 + 3*hi*hi*alpha + 4.0*hi*hip1*alpha + hip1*hip1*alpha\
                             + 2.0*hi*hip2*alpha + hip1*hip2*alpha + hi*hip1*beta + hip1*hip1*beta - hi*hip2*beta - 2.0*hip1*hip2*beta)\
                             / (him1*(him1+hi)*(him1+hi+hip1)*(him1+hi+hip1+hip2))
   
        # E_i
        self.B2_[i,i] = - (self.B2_[i,IM2] + self.B2_[i,IM1] + self.B2_[i,IP1] + self.B2_[i,IP2])

    else:

    #
    # Decentered formulas
    #

      # i = 0, first derivative
  
      h2 = X[1] - X[0]
      h3 = X[2] - X[1]
  
      self.A1_[0,0] = 1.0
      self.A1_[0,1] = (h2+h3)/h3
      self.B1_[0,0] = -(3.0*h2+2.0*h3)/(h2*(h2+h3))
      self.B1_[0,1] = (h2+h3)*(2.0*h3-h2)/(h2*h3*h3)
      self.B1_[0,2] = h2*h2/(h3*h3*(h2+h3))
  
      # i = 1, first derivative
  
      self.A1_[1,0] = h3*h3/((h2+h3)*(h2+h3))
      self.A1_[1,1] = 1.0
      self.A1_[1,2] = h2*h2/((h2+h3)*(h2+h3))
      self.B1_[1,0] = -2.0*h3*h3*(2.0*h2+h3)/(h2*(h2+h3)**3)
      self.B1_[1,1] = 2.0*(h3-h2)/(h2*h3)
      self.B1_[1,2] = 2.0*h2*h2*(h2+2.0*h3)/(h3*(h2+h3)**3)
      
      # i = N-1, first derivative
  
      h2 = X[N-2] - X[N-1]
      h3 = X[N-3] - X[N-2]
  
      self.A1_[N-1,N-1] = 1.0
      self.A1_[N-1,N-2] = (h2+h3)/h3
      self.B1_[N-1,N-1] = -(3.0*h2+2.0*h3)/(h2*(h2+h3))
      self.B1_[N-1,N-2] = (h2+h3)*(2.0*h3-h2)/(h2*h3*h3)
      self.B1_[N-1,N-3] = h2*h2/(h3*h3*(h2+h3))
  
      # i = N-2, first derivative
  
      self.A1_[N-2,N-1] = h3*h3/((h2+h3)*(h2+h3))
      self.A1_[N-2,N-2] = 1.0
      self.A1_[N-2,N-3] = h2*h2/((h2+h3)*(h2+h3))
      self.B1_[N-2,N-1] = -2.0*h3*h3*(2.0*h2+h3)/(h2*(h2+h3)**3)
      self.B1_[N-2,N-2] = 2.0*(h3-h2)/(h2*h3)
      self.B1_[N-2,N-3] = 2.0*h2*h2*(h2+2.0*h3)/(h3*(h2+h3)**3)
      
      # i = 0 second derivative
  
      a = 2.0/11.0
      b = 2.0/11.0
  
      h2 = X[1] - X[0]
      h3 = X[2] - X[1]
      h4 = X[3] - X[2]
  
      self.A2_[0,0] = 1.0
      self.A2_[0,1] = a
      self.B2_[0,0] = 2.0*(3.0*h2+2.0*h3+h4+2.0*h3*a+h4*a)/(h2*(h2+h3)*(h2+h3+h4))
      self.B2_[0,1] = -2.0*(2.0*h2+2.0*h3+h4-h2*a+2.0*h3*a+h4*a)/(h2*h3*(h3+h4))
      self.B2_[0,2] = 2.0*(2.0*h2+h3+h4-h2*a+h3*a+h4*a)/(h3*h4*(h2+h3))
      self.B2_[0,3] = -2.0*(2.0*h2+h3-h2*a+h3*a)/(h4*(h3+h4)*(h2+h3+h4))
      
      # i = 1 second derivative
  
      self.A2_[1,0] = a
      self.A2_[1,1] = 1.0 
      self.A2_[1,2] = b
      self.B2_[1,0] = 2.0*(2.0*h3+h4+(3.0*h2+2.0*h3+h4)*a-(h3-h4)*b)/(h2*(h2+h3)*(h2+h3+h4))
      self.B2_[1,1] = -2.0*(2.0*h3-h2+h4+(2.0*h2+2.0*h3+h4)*a-(h2+h3-h4)*b)/(h2*h3*(h3+h4))
      self.B2_[1,2] = 2.0*(-h2+h3+h4+(2.0*h2+h3+h4)*a-(h2+2.0*h3-h4)*b)/(h3*(h2+h3)*h4)
      self.B2_[1,3] = 2.0*(h2-h3-(2.0*h2+h3)*a+(h2+2.0*h3)*b)/(h4*(h3+h4)*(h2+h3+h4))
  
      # i = N-1, second derivative
  
      h2 = X[N-2] - X[N-1];
      h3 = X[N-3] - X[N-2];
      h4 = X[N-4] - X[N-3];
  
      self.A2_[N-1,N-1] = 1.0
      self.A2_[N-1,N-2] = a
      self.B2_[N-1,N-1] = 2.0*(3.0*h2+2.0*h3+h4+2.0*h3*a+h4*a)/(h2*(h2+h3)*(h2+h3+h4))
      self.B2_[N-1,N-2] = -2.0*(2.0*h2+2.0*h3+h4-h2*a+2.0*h3*a+h4*a)/(h2*h3*(h3+h4))
      self.B2_[N-1,N-3] = 2.0*(2.0*h2+h3+h4-h2*a+h3*a+h4*a)/(h3*h4*(h2+h3))
      self.B2_[N-1,N-4] = -2.0*(2.0*h2+h3-h2*a+h3*a)/(h4*(h3+h4)*(h2+h3+h4))
  
      # i = N-2 second derivative
  
      self.A2_[N-2,N-1] = a
      self.A2_[N-2,N-2] = 1.0
      self.A2_[N-2,N-3] = b
      self.B2_[N-2,N-1] = 2.0*(2.0*h3+h4+(3.0*h2+2.0*h3+h4)*a-(h3-h4)*b)/(h2*(h2+h3)*(h2+h3+h4))
      self.B2_[N-2,N-2] = -2.0*(2.0*h3-h2+h4+(2.0*h2+2.0*h3+h4)*a-(h2+h3-h4)*b)/(h2*h3*(h3+h4))
      self.B2_[N-2,N-3] = 2.0*(-h2+h3+h4+(2.0*h2+h3+h4)*a-(h2+2.0*h3-h4)*b)/(h3*(h2+h3)*h4)
      self.B2_[N-2,N-4] = 2.0*(h2-h3-(2.0*h2+h3)*a+(h2+2.0*h3)*b)/(h4*(h3+h4)*(h2+h3+h4))

  #####################
  # return derivative #
  #####################

  def Compute_derivative(self,U,axis=-1,m=1):
      
    if (m>2 or m<1):
      print ('Second argument must be 1 or 2\n');
      return
    if U.ndim == 1:
      if len(U) != self.N_:
        raise Exception('Vector U not the same size as FDM matrix!') 
      # First derivative
      if (m==1):
        rhs = self.B1_.dot(U)
        return spsolve(self.A1_.tocsr(),rhs)
      # Second derivative
      else:
        rhs = self.B2_.dot(U)
        return spsolve(self.A2_.tocsr(),rhs)

    # Full array
    else:
      if U.shape[axis] != self.N_:
        raise Exception('Vector U not the same size as FDM matrix!')
      # Put axis as last dimension to be able to extract data easily 
      if axis != -1:
        UU = NP.rollaxis(U,axis,U.ndim)
      else:
        UU = U
      dU = NP.zeros(UU.shape,dtype=complex)
      # Loop through all indices 
      if  m==1:
        AA = self.A1_.tocsr()
        BB = self.B1_
      else:
        AA = self.A2_.tocsr()
        BB = self.B2_

      for index in NP.ndindex(UU.shape[:-1]):
        dU[index] = spsolve(AA,BB.dot(UU[index])) 
      if axis != -1:
        dU = NP.rollaxis(dU,-1,axis)

      if U.dtype is not complex:
        dU = NP.real(dU)
      return dU


  def __call__(self,u,axis=-1,m=1):
    return self.Compute_derivative(u,axis,m)



def diffX(U,dx):
  ''' returns X-derivative of a 2D or 3D cartesian array (xy, xyz)
      4th order finite difference scheme, regular grid '''

  dU = NP.zeros(U.shape,dtype=U.dtype)

  dU[ 0,:] = (-3.e0* U[ 0,:] +4.e0* U[ 1,:] - U[ 2,:])/(2.0*dx)
  dU[-1,:] = ( 3.e0* U[-1,:] -4.e0* U[-2,:] + U[-3,:])/(2.0*dx)

  dU[ 1,:] = (U[ 2,:]-U[ 0,:])/(2.0*dx)
  dU[-2,:] = (U[-1,:]-U[-3,:])/(2.0*dx)

  dU[2:-2,:] = ( -U[4:,:] + 8.e0* U[3:-1,:] - 8.e0* U[1:-3,:] + U[0:-4,:])/(12.e0*dx) 

  return dU

def diffY(U,dy):
  ''' returns y-derivative of a 2D or 3D cartesian array (xy, xyz)
      4th order finite difference scheme, regular grid '''

  dU = NP.zeros(U.shape,dtype=U.dtype)

  dU[:, 0] = (-3.e0* U[:, 0] +4.e0* U[:, 1] - U[:, 2])/2.0
  dU[:,-1] = ( 3.e0* U[:,-1] -4.e0* U[:,-2] + U[:,-3])/2.0

  dU[:, 1] = (U[:, 2]-U[:, 0])/2.0
  dU[:,-2] = (U[:,-1]-U[:,-3])/2.0

  dU[:,2:-2] = ( -U[:,4:] + 8.e0* U[:,3:-1] - 8.e0* U[:,1:-3] + U[:,0:-4])/12.e0 

  return dU/dy

def diffZ(U,dz):
  ''' returns y-derivative of a 3D cartesian array (xyz)
      4th order finite difference scheme, regular grid '''

  dU = NP.zeros(U.shape,dtype=U.dtype)

  dU[..., 0] = (-3.e0* U[..., 0] +4.e0* U[..., 1] - U[..., 2])/2.0
  dU[...,-1] = ( 3.e0* U[...,-1] -4.e0* U[...,-2] + U[...,-3])/2.0

  dU[..., 1] = (U[..., 2]-U[..., 0])/2.0
  dU[...,-2] = (U[...,-1]-U[...,-3])/2.0

  dU[...,2:-2] = ( -U[...,4:] + 8.e0* U[...,3:-1] - 8.e0* U[...,1:-3] + U[...,0:-4])/12.e0 

  return dU/dy

