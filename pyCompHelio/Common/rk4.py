from scipy.interpolate import interp1d

def rk4(f,x0,y0,x1,n):
    ''' RK4 method
        f  = ODE (i.e. lambda x,y : x*sqrt(y)
        x0 = intial x
        y0 = y(x0)
        x1 = final xpoint
        n  = number of steps
    '''

    # Allocation
    vx    = [0]*(n+1)
    vy    = [0]*(n+1)
    h     = (x1-x0)/float(n)
    vx[0] = x = x0
    vy[0] = y = y0
    for i in range(1,n+1):
      k1 = h*f(x      ,y       )
      k2 = h*f(x+0.5*h,y+0.5*k1)
      k3 = h*f(x+0.5*h,y+0.5*k2)
      k4 = h*f(x+h    ,y+k3    )
      vx[i] = x = x + h
      vy[i] = y = y + (k1+k2+k2+k3+k3+k4)/6.0
    return vx, vy

