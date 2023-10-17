import os
def pathToMPS():
  return os.getcwd().split('mps_montjoie')[0]+'/mps_montjoie/'

RSUN = 696e6     # Sun radius
GSUN = 274       # Solar gravitional acceleration at the surface
NfilesPerDir = 300

VERBOSE = False
DEBUG   = False


from .parallelTools    import *
from .visuND           import *
from .FDM              import *
from .quadrature       import *
from .enum             import *
from .projections      import *
from .misc             import *
from .memory           import *
from .plot             import *
from .rotation         import *
from .writeFITS        import *
from .rk4              import *
from .solarFFT         import *
from .assocLegendre    import *
from .ritzwollerLavely import *
from .rayTracing       import *
from .simulation       import *
from .nodalPoints      import *
from .equationMJ       import *
from .peakdet          import *
from .FWHM             import *
from .Gaunt            import *
from .SaveSparse       import *
from .Obs_Geometries   import *
from .interpGrid       import *




