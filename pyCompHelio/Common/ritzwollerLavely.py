from   scipy.integrate   import quad  as quad
from   scipy.integrate   import simps as simps
from   matplotlib.pyplot import *
from   .assocLegendre     import *

import numpy as NP

########################################################################
# Polynomials described in Schou 1994 

def ritzwollerLavelyPolynomial(ll,i):

  ''' returns an array of size (i,2*ll+1)
      of coefficients Pcal_i(m) where -l <= m <= +l
  '''

  l  = float(ll)
  Nm = 2*ll+1
  L  = NP.sqrt(l*(l+1))
  ms = NP.arange(-ll,ll+1)

  if i<0:
    print ("Index i should be >= 0")
    return

  res = []
  if i>=0:
    res.append(l*NP.ones((Nm,)))
  if i>=1:
    res.append(ms)
  if i>=2:
    res.append((3.e0*ms*ms-l*(l+1))/(2*l-1.e0))
  
  inow = 3
  while inow <=i:

    PPPi = L*legendre(inow,ms/L)
    
    # Compute the cij coefficients
    c = NP.zeros((inow,))
    for j in range(inow):
      c[j] = NP.sum(PPPi*res[j])/sum(res[j]**2)
    
    PPi = PPPi
    for j in range(inow):
      PPi -= sum(c[j]*res[j])

    res.append(l*PPi/PPi[-1])
    inow +=1

  return NP.array(res)





