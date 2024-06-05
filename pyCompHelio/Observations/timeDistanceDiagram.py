import numpy             as NP
import matplotlib.pyplot as PLOT

from ..Common     import *
from ..Parameters import *

def timeDistanceDiagram(params,XCxt=None,XCxw=None,\
                        tMin=0,tMax=300,thetaMin=0,thetaMax=180,\
                        ucmap='Greys',title=None,\
                        thetaScaling=None,colorScaling=1,anglesSlice=[],\
                        fileName=None,**plotArgs):

    ''' Generates a plot of the cross correlation in (theta,t) domain.
        Cross correlation can be provided in frequency or time domain.
        If given in frequency domain, no padding is performed before the FFT!

        If padding was done before, make sure you used solarfft.temporal with 
        the return new time feature.

        For each given angle (in DEGREES) in anglesSlice, a plot of the slice 
        of the TD-diagram is drawn in a box to the right.
        If no angle is specified, this box is removed.

        If given, thetaMin and thetaMax are in DEGREES,
        and tMin and tMax in MINUTES.
        This is plotting stuff here ;)
    '''

    # Check data
    if XCxt is None:
      if XCxw is None:
        print(bColors.warning() + 'timeDistanceDiagram: please provide a cross correlation.')
        return 1
      XCxt = solarfft.temporalifft(XCxt,params.time_)
    
    if XCxt.shape[1] != params.time_.Nt_:
      print(bColors.warning() + 'timeDistanceDiagram : data does not match time parameters.')
    if XCxt.shape[0] != params.geom_.Ntheta():
      print(bColors.warning() + 'timeDistanceDiagram : data does not match theta parameters.')
    
    # Slice data array
    t         = params.time_.t_
    theta     = params.geom_.theta()
    itMin     = NP.argmin(abs(t    -tMin*60 ))
    itMax     = NP.argmin(abs(t    -tMax*60 ))+1
    ithetaMin = NP.argmin(abs(theta-thetaMin))
    ithetaMax = NP.argmin(abs(theta-thetaMax))+1
    
    # Scaling the plot in theta if necessary
    toPlot = real(XCxt[ithetaMin:ithetaMax,itMin:itMax]).T
    if thetaScaling is not None:
      toPlot *= (NP.cos(theta-NP.pi)+1.e0)[NP.newaxis,:]
    
    # We use the maximum of data at theta = pi/2 as reference for color scaling
    vm = NP.amax(abs(real(XSt[XSt.shape[0]/2,tMin:tMax])))*colorScaling
    
    # Determine if we plot slices and set the subplot grid
    if len(anglesSlice) == 0:
      grid = (4,5)
    else:
      grid = (4,6)
    
    fig = PLOT.figure()
    PLOT.rc('text',usetex=True)
    PLOT.rc('font',family='serif')
    
    # TD diagram
    ax1 = PLOT.subplot2grid(grid,(0,0),rowspan=4,colspan=5)
    ax1.imshow(toPlot,aspect='auto',cmap=ucmap,origin='bottom',\
               vmin=-vm,vmax=vm,extent=[thetaMin,thetaMax,tMin,tMax],**plotArgs)
    ax1.set_xticks([0, 45, 90, 135, 180],\
    	       [r'$0$',r'$45^\circ$',r'$90^\circ$',r'$135^\circ$',r'$180^\circ$'],\
    	       fontsize = 25)
    for i in anglesSlice:
      ax1.plot([i,i],[tMin,tMax])
    
    ax1.set_yticks(fontsize=20)
    ax1.set_xlabel(r'Angular Distance',fontsize=25)
    ax1.set_ylabel(r'Time (minutes)',fontsize=25)
    if isinstance(title,str):
      ax1.set_title(title)
    
    # Plot the slice(s):
    if len(anglesSlice)>0:
      ax2 = PLOT.subplot2grid(grid,(0,5),rowspan=4)
      for i in anglecut:
        iTheta = NP.argmin(abs(theta-i*pi/180.0))
        ax2.plot(toPlot[iTheta,itMin:itMax]/amax(abs(toPlot[iTheta,itMin:itMax])),arange(itMin,itMax))
      ax2.set_ylim([itMin,itMax])
      ax2.set_xlim([-1.2,1.2])
      ax2.tick_params(labelleft='off')
      ax2.tick_params(labeltop='on')
      ax2.tick_params(labelbottom='off')
      ax2.locator_params(axis='x',nbins=3)
    
    fig.tight_layout()
    
    PLOT.ion()
    try:
      PLOT.show()
    except:
      pass
    if isinstance(fileName,str):
      PLOT.savefig(fileName)
    
