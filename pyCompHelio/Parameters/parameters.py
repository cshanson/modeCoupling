import os
import sys
import numpy as NP

from ..Common       import *
from ..Background   import *
from .Geometry       import *
from .timeParameters import *

class parameters:
    ''' Container of basic time and geometry parameters 
        and background data read from .init file '''

    def __init__(self,configFile,typeOfOutput,nbProc=1,timeSampling=1,modeMax=None):

      if not os.path.exists(configFile):
        raise Exception('Configuration file %s not found.' % configFile)

      self.config_       = myConfigParser(configFile)
      self.configFile_   = configFile
      self.nbProc_       = nbProc 
      self.typeOfOutput_ = typeOfOutput

      self.BGfile_  = self.config_('BackgroundFile',pathToMPS()+'/data/background/modelS_SI_reversed.txt')
      self.time_    = timeParameters(self.config_,timeSampling)
      self.geom_    = self.createGeometry(modeMax)
      self.bgm_     = backgroundModel(self.config_)
      self.modeMax_ = modeMax
      self.unidim_  = 'EDGE' in self.config_('TypeElement','TRIANGLE_LOBATTO')

    def createGeometry(self,modeMax=None):
      ''' Parses the 'Geometry' line of config file and creates given geometry.
          If not present, creates geometry given the output files for MJ.
      '''

      #########################################################################
      # Given geometry
      outputs = self.config_('Geometry',0)
      if outputs:
        outputs = outputs.split(',')
  
        for i in range(len(outputs)):
  
          options = output[i].split()
          try:
            # Load outputs matching the given arguments
            if options[0] == TypeOfOutput.toString(typeOfOutput):
              count = 1
              dim   = 0
              if   self.typeOfOutput == TypeOfOutput.Surface1D:
                geom   = surfaceGeom1D(evalFloat(outputCrt[count]))
                count += 1
              elif self.typeOfOutput == TypeOfOutput.Surface2D:
                geom   = surfaceGeom2D(evalFloat(outputCrt[count]))
                count += 1
              elif self.typeOfOutput == TypeOfOutput.Polar2D:
                geom   = polarGeom2D()
              elif self.typeOfOutput == TypeOfOutput.Polar3D:
                geom   = polarGeom3D()
              elif self.typeOfOutput == TypeOfOutput.Cartesian2D:
                geom   = cartesianGeom2D()
              elif self.typeOfOutput == TypeOfOutput.Cartesian3D:
                geom   = cartesianGeom3D()
              else:
                raise Exception('Unknown geometry')
              # Parse other options, one set for each dimension
              while (count<len(options)):
                type = options[count]
                # Regular grid
                if type == 'UNIFORM':
                  N      = int(options[count+1])
                  count += 2
                  if count < len(options):
                    try:
                      h      = evalFloat(options[count])
                      L      = h*N
                      count += 1
                    except ValueError:
                      L = None  
                  else:
                    L = None
                  geom.setUniformComponent(dim,N,L)

                # Loaded from radius file
                elif type == 'FILE':
                  data = NP.loadtxt(options[count+1],comments='#')[:,0]
                  geom.setComponent(dim,data)
                  count += 2
                elif type == 'SAMPLE':
                  data = NP.loadtxt(options[count+1],comments='#')[::options[count+2],0]
                  geom.setComponent(dim,data)
                  count += 3
                else:
                  raise Exception('Unknown keyword %s for output description'%options[count]) 
                dim += 1
  
          except:
            raise Exception('Error while reading output options in config file.')

      #########################################################################
      # Montjoie Outputs
      else:

        # Cartesian grid output
        if self.typeOfOutput_ == TypeOfOutput.Cartesian2D:
          try:

            opts  = self.config_('FileOutputPlane',0).split()
            iStrX = opts.index("X")
            iStrZ = opts.index("Z")
            X     = readCoordinatesOptions(opts[iStrX:iStrZ],self.BGfile_)
            Y     = readCoordinatesOptions(opts[iStrZ:     ],self.BGfile_)
            geom = cartesianGeom2D(coords=[X,Y])

          except:
            raise Exception('Unable to read options of cartesian output')

        # Half circle output
        elif self.typeOfOutput_ in [TypeOfOutput.Surface1D,TypeOfOutput.Surface2D]:
          #try:

            opts   = self.config_('FileOutputCircle',0).split()
            iStrR  = opts.index("R")
            iStrTH = opts.index("THETA")

            if self.typeOfOutput_ == TypeOfOutput.Surface1D:
              geom = surfaceGeom1D(r     = evalFloat(opts[iStrR+1]),\
                                   theta = readCoordinatesOptions(opts[iStrTH:],self.BGfile_))

            else:
              Nphi = getModes(self.config_,modeMax)[1]
              geom = surfaceGeom2D(r     = evalFloat(opts[iStrR+1]),\
                                   theta = readCoordinatesOptions(opts[iStrTH:],self.BGfile_),\
                                   Nphi  = Nphi)

          #except:
          #  raise Exception('Unable to read options of surface output')      

        # Half disk output
        elif self.typeOfOutput_ in [TypeOfOutput.Polar2D,TypeOfOutput.Polar3D]:
          #try:
            opts   = self.config_('FileOutputDisk',0).split()
            iStrR  = opts.index("R")
            iStrTH = opts.index("THETA")

            if self.typeOfOutput_ == TypeOfOutput.Polar2D:

              geom = polarGeom2D(r     = readCoordinatesOptions(opts[iStrR  :iStrTH ],self.BGfile_),\
                                 theta = readCoordinatesOptions(opts[iStrTH :       ],self.BGfile_))
            else:
              Nphi = getModes(self.config_,modeMax)[1]
              geom = polarGeom3D(r     = readCoordinatesOptions(opts[iStrR  :iStrTH ],self.BGfile_),\
                                 theta = readCoordinatesOptions(opts[iStrTH :       ],self.BGfile_),\
                                 Nphi  = Nphi)

          #except:
          #  raise Exception('Unable to read options of polar output')

        # Sphere output
        elif self.typeOfOutput_ in [TypeOfOutput.Spherical3D]:
          try:

            opts = self.config_('FileOutputSphere').split()

            iStrR      = opts.index("R")
            iStrTH     = opts.index("THETA")
            iStrPHI    = opts.index("PHI")
            if 'ROTATION' in opts:
              iEnd = opts.index('ROTATION')
            else:
              iEnd = None            

            geom = sphericalGeom(r     = readCoordinatesOptions(opts[iStrR  :iStrTH ],self.BGfile_),\
                                 theta = readCoordinatesOptions(opts[iStrTH :iStrPHI],self.BGfile_),\
                                 phi   = readCoordinatesOptions(opts[iStrPHI:iEnd   ],self.BGfile_))
            
          except:
            raise Exception('Unable to read options of spherical3D output')

        else:
          raise ValueError('Type of Output %d is not compatible with Montjoie Outputs' % typeOfOutput)

      return geom
      #########################################################################

    def getModes(self):
      return getModes(self.config_)

    def getNumberModes(self):
      return getModes(self.config_)

def getModes(config,modeMax=None,montjoieIndices=False,degree=None):
  ''' Returns a vector with all the modes done in the computation 
      and the total number of modes (the length of modes except 
      for POSITIVE_ONLY where it adds the negative ones
  '''

  # Default values
  type    = "SINGLE"
  Mmax    = 0
  Ms      = [0]
  limit   = 1
  indices = [0]
  opts = config('Modes',0)

  if degree is None:
    #try:
      if opts:
        type = opts.split()[0]
        Mmax = int(opts.split()[1])
        if   type == 'SINGLE':
          Ms = [Mmax]
        elif type == 'POSITIVE_ONLY':
          if modeMax is not None:
            Mmax  = modeMax
          Ms      = NP.arange(-Mmax,Mmax+1)
          indices = NP.arange(Mmax+1)
        elif type == 'ALL':
          offset   = 0
          if modeMax is not None:
            offset = Mmax-modeMax
            Mmax   = modeMax
          Ms       = NP.arange(-Mmax,Mmax+1)
          indices  = NP.arange(2*Mmax+1)+offset
        elif type == 'SEQ':
          Mmin    = int(opts.split()[1])
          Mstep   = int(opts.split()[2])
          Mmax    = int(opts.split()[3])
          Ms      = NP.arange(Mmin,Mmax+1,Mstep)
          indices = NP.arange(len(Ms))

      if not montjoieIndices:
        return NP.array(Ms),len(Ms),len(indices)
      else:
        return NP.array(Ms),len(Ms),len(indices),NP.array(indices)

    #except:
    #  raise Exception('Unable to read parameters of keyword "Modes".')
  else:
    opts = config('MaximumDegree',0)
    LMax = int(opts.split()[0])
    if hasattr(degree,"__len__"):
      Ls = NP.array(degree)
    else:
      Ls = NP.array([degree])
    return Ls,len(Ls),len(Ls),Ls

def getModesForRunMontjoie(config):

  # Default values
  opts = config('Modes','SINGLE 0').upper().split()
  try:
    Mmax = int(opts[1])
    if   opts[0] == 'SINGLE':
      Modes = [Mmax]
    elif opts[0] == 'POSITIVE_ONLY':
      Modes = NP.arange(0,Mmax+1)
    elif opts[0] == 'ALL':
      Modes = NP.arange(-Mmax,Mmax+1)
    elif opts[0] == 'SEQ':
      Mmin  = int(opts[1])
      Mstep = int(opts[2])
      Mmax  = int(opts[3])
      Modes = NP.arange(Mmin,Mmax+1,Mstep)
  except:
    raise Exception('Unable to read parameters of keyword "Modes".')

  return Modes
