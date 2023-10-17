from ..Common     import *
from ..Parameters import *
#from ..visuND     import loadND


import numpy as NP


# FIXME !

def computeConsistancyError(U,MJO,eqn,resdir,tol=0.9):
  ''' returns the L2 median consistancy error of solution U to equation eqn.
      100tol% of points are kept to exclude boundaries and source 
      error is measure between the omega^2 U term and the div(mu grad(u)) term'''

  GM  = MJO.geom_
  IF  = MJO.initFile_

  # Load rho, sigma and mu coefficients from Montjoie
  # (not background coeffs)

  if IF.verbose_ : 
    print( "Compute consistancy error" )
  # single frequency runs
  omega = MJO.time_.omega_[0]*RSUN

  [xm,ym] = NP.meshgrid(GM.x_,GM.y_)

  if eqn=="HELMHOLTZ":

    rho   = loadND(resdir+"rho_U0.dat")[4]
    mu    = loadND(resdir+"mu_U0.dat")[4]
    sigma = loadND(resdir+"sigma_U0.dat")[4]
   
    if IF.debug_:
      plot_cart2D(NP.real(rho  ),GM,"rho.png")
      plot_cart2D(NP.real(mu   ),GM,"mu.png")
      plot_cart2D(NP.real(sigma),GM,"sigma.png")

    if IF.verbose_ : 
      print( "- Compute first term")
    U1 = U*(rho*omega**2 + 1j*sigma*omega)
    U1 = NP.nan_to_num(U1)

    if IF.verbose_ : 
      print ("- Compute second term")
    dU = GM.computeCartesianGradient(U)
    dU = NP.nan_to_num(dU)
    U2 = GM.computeCartesianDivergence(xm[:,:,NP.newaxis]*mu[:,:,NP.newaxis]*dU)/xm
    U2 = NP.nan_to_num(U2)

  else:
   
    rho   = loadND(resdir+"rho_U0.dat")[4]
    mu    = loadND(resdir+"mu_U0.dat")[4]
    sigma = loadND(resdir+"sigma_U0.dat")[4]
    alpha = loadND(resdir+"alpha_U0.dat")[4]
    beta  = loadND(resdir+"beta_U0.dat")[4]

    if IF.debug_:
      plot_cart2D(NP.real(rho  ),GM,"rho.png")
      plot_cart2D(NP.real(mu   ),GM,"mu.png")
      plot_cart2D(NP.real(sigma),GM,"sigma.png")
      plot_cart2D(NP.real(alpha),GM,"alpha.png")
      plot_cart2D(NP.real(beta),GM,"beta.png")

    if IF.verbose_ : 
      print( "- Compute first term")
    U1 = U*(rho*omega**2 + 1j*sigma*omega)
    U1 = NP.nan_to_num(U1)

    if IF.verbose_ : 
      print( "- Compute second term")
    dU = GM.computeCartesianGradient(U*alpha)
    U2 = GM.computeCartesianDivergence(xm[:,:,NP.newaxis]*mu[:,:,NP.newaxis]*dU)/xm
    U2 = beta*U2
    U2 = NP.nan_to_num(U2)



  # Error computation ====================================================================================

  # 1st version === Median error

  #val  = NP.absolute(U1.ravel())
  #diff = NP.absolute(U1.ravel()+U2.ravel())

  #thr  = 1.e-7*NP.max(val)
  #mask = (val>thr)*(diff>thr)

  #diff = diff*mask
  #val  = val*mask

  #diff   = NP.sort(diff)
  #val    = NP.sort(val)
  #offset = NP.searchsorted(1*(val>0),1)
  #N      = size(diff)-offset
  #pos    = NP.floor(N*tol+offset)
 
  #if IF.debug_:
  #  print "max U1 " , NP.max(val)
  #  print "max diff ", NP.max(diff)
  #  print "size, total ", size(diff)
  #  print "offset ", offset
  #  print "pos ", pos
  #  print "val[pos] ", val[pos]
  #  print "diff[pos] ", diff[pos] 

  #return diff[pos]/val[pos]

 
  # 2nd version === L2 norm of points below median error ========

  # Build mask : select points inside circle
  r       = sqrt(xm*xm+ym*ym) 
  mask_r  = r < MJO.initFile_.R_
  mask_x  = abs(GM.x_) > 2*GM.hx_ # avoid wrong 1/r dr(rdr) along axis, on x<0, wrong signs compensate
  mask_r  = mask_r*mask_x

  # Exclude points where U1 is too small (avoid /0)
  AU1      = NP.absolute(U1)
  AU2      = NP.absolute(U2)
  mask_z1  = AU1 > 1.e-7*NP.max(AU1)
  mask_z2  = AU2 > 1.e-7*NP.max(AU1)
  mask_z   = mask_z1*mask_z2

  if IF.debug_:
    plot_cart2D(AU1,GM,"abs_U1.png",vmin=0,vmax=0.01*NP.max(AU1))
    plot_cart2D(AU2,GM,"abs_U2.png",vmin=0,vmax=0.01*NP.max(AU1))

  # tol% of points r<R
  diff    = NP.absolute(U2+U1)
  diff    = NP.nan_to_num(diff)
  diff    = diff*mask_z
  expdd   = diff[mask_r]
  expdd   = NP.sort(expdd)
  expdd   = expdd[expdd!=0]
  limit   = expdd[NP.floor(tol*size(expdd))]
  mask_v  = diff < limit

  mask    = mask_r * mask_v * mask_z

  if IF.debug_:
    plot_cart2D(1*mask   ,GM,"mask.png"     ,vmin=0,vmax=1    )
    plot_cart2D(diff     ,GM,"diff.png"     ,vmin=0,vmax=1.e-3*NP.max(AU1))
    plot_cart2D(diff*mask,GM,"diff_mask.png",vmin=0,vmax=limit)
    plot_cart2D((diff/AU1)*mask,GM,"diff_mask_rel.png",vmin=0,vmax=1.e-2)

  err = norm2(diff*mask)/norm2(AU1)
  return err

