import numpy         as NP
from .basis2D         import *
from .constantBasis1D import *
from .separableBasis2D import *
from scipy           import sparse


#########################################################

class constantBasis2D(basis2D):
  
      def __init__(self, r1, r2):
        basis2D.__init__(self, r1, r2)   
        hr1 = NP.diff(self.r1_)
        hr1 = NP.concatenate([hr1, [hr1[-1]]])
        hr2 = NP.diff(self.r2_)
        hr2 = NP.concatenate([hr2, [hr2[-1]]])  
        mass = hr1[:,NP.newaxis] * hr2[NP.newaxis,:]

        self.nbBasisFunctions_ = [len(self.r1_),len(self.r2_)]
        N = len(self.r1_) * len(self.r2_)
        mass = NP.reshape(mass,  (N))
        massMatrix =  sparse.lil_matrix((N,N))
        massMatrix.setdiag(mass)
        self.mass_ = massMatrix.tocsr()

        #self.mass_ = NP.diag(mass)
  
      def projectOnBasis(self, quantity):
        return quantity
    
      def reconstructFromBasis(self, coeffs, x = None):
        '''The interpolation will be done directly in runMontjoie.'''
        #return coeffs
        return NP.reshape(coeffs, [len(self.r1_), len(self.r2_)])
            
      def createSmoothnessMatrix(self, smoothnessOrder1 = 0, BCleft1 = None, BCright1 = None, r1 = None, smoothnessOrder2 = 0, BCleft2 = None, BCright2 = None, r2 = None ):
        '''compute the smoothness matrix L for regularization when the quantity to recover is 2D defined on a (r,theta) grid. If smoothnessOrder = 0, L is the L2 norm, smoothnessOrder = 1, norm of the gradient, smoothnessOrder = 2, norm of the Laplacian'''
        basis1 = constantBasis1D(self.r1_)
        basis2 = constantBasis1D(self.r2_)
        basis  = separableBasis2D(basis1,basis2)
        return basis.createSmoothnessMatrix(smoothnessOrder1,BCleft1,BCright1,r1, smoothnessOrder2,BCleft2,BCright2,r2)
        #Nr = len(self.r_)
        #Nth = len(self.theta_)
        ##L = NP.zeros((Nr*Nth, Nr*Nth))
        #L = sparse.lil_matrix((Nr*Nth, Nr*Nth))
        #dr = NP.diff(self.r_)
        #dth = NP.diff(self.theta_)
        #if smoothnessOrder == 0:
        #  dr = NP.concatenate([dr,[dr[-1]]])
        #  dth = NP.concatenate([dth,[dth[-1]]])
        #  diagTerm = NP.zeros((Nr*Nth))
        #  for i in range(Nr):
        #    ks = i * Nth
        #    kl = (i+1) * Nth
        #    indDiag = [range(ks,kl), range(ks,kl)]
        #    diagTerm[ks:kl] =  dr[i] * dth
        #    #L[indDiag] = dr[i] * dth
        #  L.setdiag(diagTerm.tolist())
        #elif smoothnessOrder == 1:
        #  for i in range(1,Nr-1):
        #    # Radial discretization
        #    ks = i * Nth
        #    kl = (i+1) * Nth
        #    indDiag = [range(ks,kl), range(ks,kl)]
        #    L[indDiag] = - 1./dr[i]
        #    indOver = [range(ks,kl), range(ks+1,kl+1)]
        #    L[indOver] = 1. / dr[i]

        #  for i in range(Nr):
        #    # Angular discretization
        #    ks = i * Nth + 1
        #    kl = (i+1) * Nth - 1
        #    if i != Nr - 1:
        #      indDiag = [range(ks,kl), range(ks,kl)]
        #      L[indDiag] += - 1./dth
        #      indOver = [range(ks,kl), range(ks+Nth,kl+Nth)]
        #      L[indOver] = 1. / dtheta
        #    else:
        #      indDiag = [range(ks,kl), range(ks,kl)]
        #      L[indDiag] += 1./dth
        #      indOver = [range(ks,kl), range(ks-Nth,kl-Nth)]
        #      L[indOver] = -1. / dtheta

        #elif smoothnessOrder == 2:
        #      L = NP.zeros((len(self.r_),len(self.r_)))
        #      for i in range(1,len(self.r_)-1):
        #          h1 = self.r_[i+1]-self.r_[i]
        #          h2 = self.r_[i]-self.r_[i-1]
        #          L[i,i-1] = 2*h1/(h1**3 + h2**3)
        #          L[i,i]   = -2*(h1+h2) / (h1**3+h2**3)
        #          L[i,i+1] = 2*h2/(h1**3+h2**3)
        #      L = L[1:-1,:]
        #return L.tocsr()
