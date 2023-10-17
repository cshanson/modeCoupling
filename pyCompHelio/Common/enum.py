import numpy as NP
import matplotlib.pyplot as plt
from .     import *
#from ..Parameters import *

class TypeOfModeFilter:
  FMODE = 0
  P1MODE = 1
  P2MODE = 2
  P3MODE = 3
  P4MODE = 4

  @staticmethod
  def toString(typeOfModeFilter):
    ''' return fmode, p1mode, ...'''
    return '%smode' % TypeOfModeFilter.toStringShort(typeOfModeFilter)

  @staticmethod
  def toStringShort(typeOfModeFilter):
    ''' return f, p1, ...'''
    if (typeOfModeFilter == TypeOfModeFilter.FMODE):
      return 'f'
    else:
      return 'p%s' % typeOfModeFilter
      
def radialToCart2DOmegaParallel(args):
  '''to overpass the one argument limit in the case of the use of the parallel function cartesianGeometry2D.radialToCart2D.'''
 
  res = TypeOfModeFilter.radialToCart2DOmega(*args)
  if ('PG' in globals()):
    PG.update()
  return res

class SpatialTypes:

  UNIFORM = 0
  RADIAL = 1
  INTERP2D = 2
  NODAL = 3

  @staticmethod
  def toString(spatialType):
    if spatialType == SpatialTypes.UNIFORM:
      return 'Uniform'
    elif spatialType == SpatialTypes.RADIAL:
      return 'Radial'
    elif spatialType == SpatialTypes.INTERP2D:
      return 'Interp2D'
    elif spatialType == SpatialTypes.NODAL:
      return 'Nodal'
    else:
      raise Exception('Spatial type %s not implemented' % spatialType)

  @staticmethod
  def toSpatialType(spatialType):
    if spatialType.upper() == 'UNIFORM':
      return SpatialTypes.UNIFORM
    elif spatialType.upper() == 'RADIAL':
      return SpatialTypes.RADIAL
    elif spatialType.upper() == 'INTERP2D':
      return SpatialTypes.INTERP2D
    elif spatialType.upper() == 'NODAL':
      return SpatialTypes.NODAL
    else:
      raise Exception('Spatial type %s not implemented' % spatialType)

class TypeOfBackgroundParameter:
  RADIUS = 0
  C = 1
  RHO = 2
  P = 3

  @staticmethod
  def toString(typeOfParameter):
    if (typeOfParameter == TypeOfBackgroundParameter.RADIUS):
      out = 'RADIUS'
    elif (typeOfParameter == TypeOfBackgroundParameter.C):
      out = 'C'
    elif (typeOfParameter == TypeOfBackgroundParameter.RHO):
      out = 'RHO'
    else:
      out = 'P'
    return out

class TypeOfKernel:
  SOUNDSPEED = 1
  DENSITY    = 2
  FLOW       = 3
  FLOWR      = 4
  FLOWTHETA  = 5
  FLOWPHI    = 6
  DAMPING    = 7
  SOURCE     = 8
  FORWARDMODEL = 9
  DEBUG      = 10

  @staticmethod
  def toTypeOfKernel(typeOfKernel): 
    if typeOfKernel.upper() == 'DAMPING':
      return TypeOfKernel.DAMPING
    elif typeOfKernel.upper() == 'SOUNDSPEED' or typeOfKernel.upper() == 'C':
      return TypeOfKernel.SOUNDSPEED
    elif typeOfKernel.upper() == 'DENSITY' or typeOfKernel.upper() == 'RHO':
      return TypeOfKernel.DENSITY
    elif typeOfKernel.upper() == 'FLOW':
      return TypeOfKernel.FLOW
    elif typeOfKernel.upper() == 'SOURCE':
      return TypeOfKernel.SOURCE
    else:
      raise Exception('Type of kernel not implemented')

  @staticmethod
  def toString(typeOfKernel): 
    if typeOfKernel == TypeOfKernel.SOUNDSPEED:
      return 'SoundSpeed'
    elif typeOfKernel == TypeOfKernel.DENSITY:
      return 'Density'
    elif typeOfKernel == TypeOfKernel.FLOW:
      return 'Flow'
    elif typeOfKernel == TypeOfKernel.DAMPING:
      return 'Damping'
    elif typeOfKernel == TypeOfKernel.Source:
      return 'Source'
    else:
      raise Exception('Type of kernel not implemented')
 
  @staticmethod
  def toStringShort(typeOfKernel): 
    if typeOfKernel == TypeOfKernel.SOUNDSPEED:
      return 'c'
    elif typeOfKernel == TypeOfKernel.DENSITY:
      return 'rho'
    elif typeOfKernel == TypeOfKernel.FLOW:
      return 'Flow'
    elif typeOfKernel == TypeOfKernel.DAMPING:
      return 'gammaDamping'
    elif typeOfKernel == TypeOfKernel.Source:
      return 's'
    else:
      raise Exception('Type of kernel not implemented')

class TypeOfOutput:
  Cartesian2D = 1
  Polar2D = 2
  Polar3D = 3
  Surface1D = 4
  Surface2D = 5
  Cartesian3D = 6
  Spherical3D = 7

  @staticmethod
  def toString(typeOfOutput):
    if (typeOfOutput == TypeOfOutput.Cartesian2D):
      str = 'Cartesian2D'
    elif (typeOfOutput == TypeOfOutput.Polar2D):
      str = 'Polar2D'
    elif (typeOfOutput == TypeOfOutput.Polar3D):
      str = 'Polar3D'
    elif (typeOfOutput == TypeOfOutput.Surface1D):
      str = 'Surface1D'
    elif (typeOfOutput == TypeOfOutput.Surface2D):
      str = 'Surface2D'
    elif (typeOfOutput == TypeOfOutput.Cartesian3D):
      str = 'Cartesian3D'
    else:
      error('Unknown type of output')
    return str

class TypeOfTravelTime:
  PLUS = 1
  MINUS = 2
  MEAN = 3
  DIFF = 4

  @staticmethod
  def toTypeOfTravelTime(typeOfTravelTime): 
    if typeOfTravelTime.upper() == 'PLUS':
      return TypeOfTravelTime.PLUS
    elif typeOfTravelTime.upper() == 'MINUS':
      return TypeOfTravelTime.MINUS
    elif typeOfTravelTime.upper() == 'DIFF':
      return TypeOfTravelTime.DIFF
    elif typeOfTravelTime.upper() == 'MEAN':
      return TypeOfTravelTime.MEAN
    else:
      raise Exception('Type of travel time not implemented')

class TypeOfTravelTimeAveraging:
  PtP_EW = 0 # point to point East-West
  PtP_SN = 1 # point to point South-North
  ANN    = 2 # annulus
  EW     = 3 # east-west
  SN     = 4 # south-north

  @staticmethod
  def toString(travelTimeAvg):
    if (travelTimeAvg == TypeOfTravelTimeAveraging.PtP_EW):
      str = 'PtP_EW'
    elif (travelTimeAvg == TypeOfTravelTimeAveraging.PtP_SN):
      str = 'PtP_SN'
    elif (travelTimeAvg == TypeOfTravelTimeAveraging.ANN):
      str = 'ANN'
    elif (travelTimeAvg == TypeOfTravelTimeAveraging.EW):
      str = 'EW'
    elif (travelTimeAvg == TypeOfTravelTimeAveraging.SN):
      str = 'SN'
    else:
      raise Exception('Type of travel time averaging not defined yet')
    return str

  @staticmethod
  def toString2(travelTimeAvg):
    if (travelTimeAvg == TypeOfTravelTimeAveraging.PtP_EW):
      str = 'PtP_EW'
    elif (travelTimeAvg == TypeOfTravelTimeAveraging.PtP_SN):
      str = 'PtP_SN'
    elif (travelTimeAvg == TypeOfTravelTimeAveraging.ANN):
      str = 'cos_m0'
    elif (travelTimeAvg == TypeOfTravelTimeAveraging.EW):
      str = 'cos_m1'
    elif (travelTimeAvg == TypeOfTravelTimeAveraging.SN):
      str = 'sin_m1'
    else:
      raise Exception('Type of travel time averaging not defined yet')
    return str

class TypeOfWindowFunction:
  HEAVISIDE = 1
  RECTANGULAR = 2
  GAUSSIAN = 3
  CUSTOM = 4
  SMOOTHRECTANGLE = 5

  @staticmethod
  def toString(wtype):
    if   wtype == TypeOfWindowFunction.HEAVISIDE:
      return "Heaviside"
    elif wtype == TypeOfWindowFunction.RECTANGULAR:
      return "Rectangle (sharp)"
    elif wtype == TypeOfWindowFunction.GAUSSIAN:
      return "Gaussian"
    elif wtype == TypeOfWindowFunction.CUSTOM:
      return "User defined"
    elif wtype == TypeOfWindowFunction.SMOOTHRECTANGLE:
      return "Rectangle (smooth)"

  @staticmethod
  def toTypeOfWindowFunction(typeOfWindowFunction): 
    if typeOfWindowFunction.upper() == 'HEAVISIDE':
      return TypeOfWindowFunction.HEAVISIDE
    elif typeOfWindowFunction.upper() == 'RECTANGULAR':
      return TypeOfWindowFunction.RECTANGULAR
    elif typeOfWindowFunction.upper() == 'GAUSSIAN':
      return TypeOfWindowFunction.GAUSSIAN
    elif typeOfWindowFunction.upper() == 'CUSTOM':
      return TypeOfWindowFunction.CUSTOM
    elif typeOfWindowFunction.upper() == 'SMOOTHRECTANGLE':
      return TypeOfWindowFunction.SMOOTHRECTANGLE
    else:
      raise Exception('Type of window function not implemented')

    
class TypeOfObservable:
  rhoc2DivXi = 1
  cDivXi = 2
  sqrtrhoc2DivXi = 3

  @staticmethod
  def toTypeOfObservable(typeOfObservable):
    if typeOfObservable.lower() == 'rhoc2divxi':
      return TypeOfObservable.rhoc2DivXi
    elif typeOfObservable.lower() == 'cdivxi':
      return TypeOfObservable.cDivXi
    elif typeOfObservable.lower() == 'sqrtrhoc2divxi':
      return TypeOfObservable.sqrtrhoc2DivXi
    else:
      raise Exception('This type of observable %s is not defined' % typeOfObservable)

class TypeOfOTF:
  HMI_OTF = 1
  MDI_OTF = 2
  HMI_2_MDI_REBIN = 3
  MDI_2_MediumEll = 4
  CUSTOM_FILE = 5

class TypeOfInversion:
  LEASTSQUARE       = 0
  TIKHONOV          = 1
  ITERATIVETIKHONOV = 2
  TSVD              = 3
  NEWTONCG          = 4
  IRGNMCG           = 5
  LEVENBERG         = 6

class TypeOfFilter:
  ''' describes the dependancies of the filters '''
  PhaseSpeed        = 1 # L-omega
  lFilter           = 2 # L
  Mode              = 3 # L-omega following ridges
  lmFilter          = 4 # L-M
  lmwFilter         = 5 # L-M-omega

class Mesh2DLineReferences:
  XAXIS_SYMMETRY    = 1
  ROTATION_AXIS     = 2
  TACHOCLINE        = 3
  CIRCULAR_BOUNDARY = 4
  FILTERED_DIRAC    = 5

class Mesh2DSurfaceReferences:
  MAIN = 1
