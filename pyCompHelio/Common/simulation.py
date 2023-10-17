import os
import numpy as NP

from .             import *
from .nodalPoints  import *
from .equationMJ   import *
from .. Parameters import *
from .. Filters    import *
from .. Mesh       import *

# ======================================================================

class Simulation:
    ''' Class that regroup all information to generate a .ini file 
        to perform a single Montjoie compFutation '''

    def __init__(self,cfg,bgm,ID,IF,IM,freq,m,verbose=False):

        self.config     = cfg
        self.KW         = cfg.sections()[0]
        self.bgm        = bgm
        self.ID         = ID
        self.freq       = freq
        self.mode       = m
        self.verbose    = verbose

        self.modeSuffix = '_m'+str(IM)
        self.freqSuffix = '_f'+str(IF)
        self.freqFolder = 'f%i' % int(NP.floor(IF/NfilesPerDir)*NfilesPerDir) + '_' + 'f%i' % int(NP.floor(IF/NfilesPerDir)*NfilesPerDir + NfilesPerDir - 1)
        self.mFolder    = 'm%i' % int(NP.floor(IM/NfilesPerDir)*NfilesPerDir) + '_' + 'm%i' % int(NP.floor(IM/NfilesPerDir)*NfilesPerDir + NfilesPerDir - 1)


        self.varyingListSources = (not isinstance(self,Simulation1D))\
                                   and cfg('FilteredDirac',0)\
                                   and cfg('FilteredDirac').upper().split()[0] in ['LMFILTER','PHASE_SPEED','LFILTER','MODE_FILTER']

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load conf for basic parameters (if not present, default values)

        self.typeEquation = cfg('TypeEquation','HELMHOLTZ').split()[0]
        if self.typeEquation == 'HELMHOLTZ':
          self.equation = EquationMJHelmholtz(self.bgm)
        elif self.typeEquation == 'HELIO_HELMHOLTZ':
          raise NotImplementedError('Other types of equations not implemented yet')
        elif self.typeEquation == 'HELMHOLTZ_GAMMA_LAPLACE':
          self.equation = EquationMJHelmholtzGammaLaplace(self.bgm)
          # raise NotImplementedError('Other types of equations not implemented yet')
        elif self.typeEquation == 'HELIO_HELMHOLTZ_V2':
          raise NotImplementedError('Other types of equations not implemented yet')
        elif self.typeEquation == 'GALBRUN':
          self.equation = EquationMJGalbrun(self.bgm)
        else:
          raise ValueError('Equation type '+typeEquation+' not known.')

        self.subFolderString = '%i_%i' % (NP.floor(float(self.ID)/NfilesPerDir)*NfilesPerDir,NP.floor(float(self.ID)/NfilesPerDir)*NfilesPerDir+NfilesPerDir-1)

    # ==================================================================

    def setOutput(self):
        ''' generates entries for Montjoie .ini file for output'''

        self.outputLinesToWrite = []

        # ===============================================================================================
        # Formats which don't require list of points files

        options = self.config('FileOutputMesh',0)
        if options: 
          try:
            options = options.split()
            self.outputLinesToWrite.append('SismoMeshVolumetric = AUTO REFERENCE ALL\n')
            self.outputLinesToWrite.append('FileOutputMeshVolumetric = diffracJunk %s\n'% (options[0]+self.modeSuffix+self.freqSuffix))
          except:
            raise IOError('Impossible to read options for FileOutputMesh')

        # ==================================================================================
        # Formats requiring points files

        options = self.config('FileOutputDisk',0)
        if options: 
          try:
            options  = options.split()
            fileName = self.config('OutDir')+'/disk_points_rth.txt'
            self.writeRTHPointsFile(fileName,options)
            self.outputLinesToWrite.append('SismoPointsFile = %s\n'% fileName)
            self.outputLinesToWrite.append('FileOutputPointsFile = diffracJunk %s\n'% (options[0]+self.modeSuffix+self.freqSuffix))
          except:
            raise IOError('Impossible to read options for FileOutputDisk or writing points went wrong.')

        options = self.config('FileOutputPlane',0)
        if options: 
          try:
            options = options.split()
            fileName = self.config('OutDir')+'/plane_points_rth.txt'
            self.writeXZPointsFile(fileName,options)
            self.outputLinesToWrite.append('SismoPointsFile = %s\n'% fileName)
            self.outputLinesToWrite.append('FileOutputPointsFile = diffracJunk %s\n'% (options[0]+self.modeSuffix+self.freqSuffix))
            #self.outputLinesToWrite.append('SismoPlaneAxi = -%s 0 -%s %s 0 -%s -%s 0 %s %s %s\n' % tuple(6*[options[1]]+2*[options[2]]))
            #self.outputLinesToWrite.append('FileOutputPlaneAxi = diffracJunk %s\n'% (options[0]+self.modeSuffix+self.freqSuffix))
          except:
            raise IOError('Impossible to read options for FileOutputPlane')

        options = self.config('FileOutputCircle',0)
        if options: 
          try:
            options  = options.split()
            fileName = self.config('OutDir')+'/half_circle_points_rth.txt'
            self.writeTHPointsFile(fileName,options)
            self.outputLinesToWrite.append('SismoPointsFile = %s\n'% fileName)
            self.outputLinesToWrite.append('FileOutputPointsFile = diffracJunk %s\n'% (options[0]+self.modeSuffix+self.freqSuffix))
          except:
            raise IOError('Impossible to read options for FileOutputHalfCircle or writing points went wrong.')

        options = self.config('FileOutputSphere',0)
        if options: 
          try:
            options  = options.split()
            fileName = self.config('OutDir')+'/sphere_points_rth.txt'
            self.writeRTHPHIPointsFile(fileName,options,threeDim = True)
            self.outputLinesToWrite.append('SismoPointsFileAxi = %s\n'% fileName)
            self.outputLinesToWrite.append('FileOutputPointsFileAxi = diffracJunk %s\n'% (options[0]+self.modeSuffix+self.freqSuffix))
          except:
            raise IOError('Impossible to read options for FileOutputSphere or writing points went wrong.')

        options = self.config('FileOutputPointsFile',0)
        if options: 
          try:
            options  = options.split()
            self.outputLinesToWrite.append('SismoPointsFileAxi = %s\n'% options[1])
            self.outputLinesToWrite.append('FileOutputPointsFileAxi = diffracJunk %s\n'% (options[0]+self.modeSuffix+self.freqSuffix))
          except:
            raise IOError('Impossible to read options for FileOutputPointsFile or writing points went wrong.')

    def writeRTHPointsFile   (self,fileName,options,threeDim=False):
        ''' Write coordinates of output poins for a half disk '''

        if not os.path.exists(fileName):
          if VERBOSE:
            print ('Generating coordinates list for outputs')
          if not ('R' in options and 'THETA' in options):
            raise Exception("Missing coordinate for parameters line FileOutputDisk")

          iStrR  = options.index('R')
          iStrTH = options.index('THETA')
          R  = readCoordinatesOptions(options[iStrR :iStrTH],self.bgm.fileName)
          TH = readCoordinatesOptions(options[iStrTH:      ],self.bgm.fileName)

          rr,tt  = NP.meshgrid(R,TH,indexing='ij')
          points = NP.array([rr.ravel()*NP.sin(tt.ravel()),\
                             rr.ravel()*NP.cos(tt.ravel())]).T
          with open(fileName,'w') as OF:
            for p in points:
              if threeDim:
                OF.write('%1.12f 0.e0 %1.12f\n'% tuple(p))
              else:
                OF.write('%1.12f %1.12f\n'% tuple(p))

    def writeTHPointsFile    (self,fileName,options,threeDim=False):
        ''' Write coordinates of output poins for a half circle '''

        if not os.path.exists(fileName):
          if VERBOSE:
            print ('Generating coordinates list for outputs')

          iStrR  = options.index('R')
          iStrTH = options.index('THETA')
          R      = evalFloat(options[iStrR+1])
          TH     = readCoordinatesOptions(options[iStrTH:],self.bgm.fileName)

          points = NP.array([R*NP.sin(TH),R*NP.cos(TH)]).T
          with open(fileName,'w') as OF:
            for p in points:
              if threeDim:
                OF.write('%1.12f 0.e0 %1.12f\n'% tuple(p))
              else:
                OF.write('%1.12f %1.12f\n'% tuple(p))

    def writeRTHPHIPointsFile(self,fileName,options,threeDim=False):
        ''' Write coordinates of output poins for a half circle '''

        if not os.path.exists(fileName):

          if VERBOSE:
            print ('Generating coordinates list for outputs')
          if not ('R' in options and 'THETA' in options and 'PHI' in options):
            raise Exception("Missing coordinate for parameters line FileOutputSphere")

          iStrR   = options.index("R")
          iStrTH  = options.index("THETA")
          iStrPHI = options.index("PHI")

          R        = readCoordinatesOptions(options[iStrR  :iStrTH ],self.bgm.fileName)
          TH       = readCoordinatesOptions(options[iStrTH :iStrPHI],self.bgm.fileName)
          PHI      = readCoordinatesOptions(options[iStrPHI:       ],self.bgm.fileName)

          rr,tt,ff = NP.meshgrid(R,TH,PHI,indexing='ij')
          points   = NP.array([rr.ravel(),tt.ravel(),ff.ravel()]).T

          with open(fileName,'w') as OF:
            for p in points:
              x  = p[0]*sin(p[1])*cos(p[2])
              y  = p[0]*sin(p[1])*sin(p[2])
              z  = p[0]*cos(p[1])
              if threeDim:
                OF.write('%1.16e %1.16e %1.16e\n'% (x,y,z))
              else:
                OF.write('%1.16e %1.16e\n'% (x,z))

    def writeXZPointsFile (self,fileName,options,threeDim=False):
        ''' Write coordinates of output poins for a half disk '''

        if not os.path.exists(fileName):
          if self.verbose:
            print ('Generating coordinates list for outputs')
          if not ('X' in options and 'Z' in options):
            raise Exception("Missing coordinate for parameters line FileOutputPlane")

          iStrX = options.index('X')
          iStrZ = options.index('Z')
          X = readCoordinatesOptions(options[iStrX:iStrZ])
          Z = readCoordinatesOptions(options[iStrZ:     ])
          xx,zz  = NP.meshgrid(X,Z,indexing='ij')
          points = NP.array([xx.ravel(),zz.ravel()]).T
          with open(fileName,'w') as OF:
            for p in points:
              if threeDim:
                OF.write('%1.12f 0.e0 %1.12f\n'% tuple(p))
              else:
                OF.write('%1.12f %1.12f\n'% tuple(p))

    # ==================================================================

    def writeMJiniFile(self):
        configDirectory = self.config('OutDir') + '/configFiles/%s/' % self.subFolderString
        mkdir_p(configDirectory)

        self.MJInputFile = configDirectory + '/' + self.config('MontjoieInputFileName','config')\
                         + '_' + str(self.ID) + '.ini'
 
        F = open(self.MJInputFile,'w')

        # Equation
        if self.typeEquation == 'HELMHOLTZ_GAMMA_LAPLACE':
          F.write('TypeEquation = HELMHOLTZ\n')
          F.write('GammaLaplace = YES\n')
        elif self.typeEquation == 'GALBRUN':
          F.write('TypeEquation = HARMONIC_AEROACOUSTIC\n')
          self.duplicateInitToMJIni(F,'Polarization')
          self.duplicateInitToMJIni(F,'PenalizationDG')
          self.duplicateInitToMJIni(F,'EnergyConservingAeroacousticModel')
        else:
          F.write('TypeEquation = %s\n' % self.typeEquation)
          F.write('GammaLaplace = NO\n')
        # F.write('FormulationAxisymmetric = R3\n')

        # Solver
        F.write('TypeSolver = %s\n' % self.config('TypeSolver', 'DIRECT'))
        F.write('NonLinearSolver = %s\n' % self.config('NonLinearSolver', 'MINPACK 1e-15 50'))
        if self.config('EstimateConditionNumber',0):
          F.write('EstimationConditionNumber = YES %s\n' % self.config('EstimateConditionNumber'))     
        self.duplicateInitToMJIni(F,'ScalingMatrix')
        F.write('StorageMatrix = %s mat.dat\n' % self.config('StorageMatrix','NO'))


        # Source term
        srcLoc,srcType = getSource(self.config,relocate=self.config('MoveSourceToVertex','YES').upper() == 'YES')
        if srcType == 'SRC_DIRAC':
          if self.config('FilteredDirac',0) or self.config('PSF_OTF',0):
            diracFileName = []
            for jj in range(len(self.config('FilteredDirac').split(','))):
              diracFileNameBase = self.config('OutDir') + '/diracAmplitudes_s%i' % jj
              diracFileName.append(self.writeFilteredDirac(diracFileNameBase,jj))
            F.write('TypeSource = SRC_USER %1.16e %1.16e %1.16e %d %s\n'
                    % (srcLoc[0][0],-srcLoc[0][1],srcLoc[0][2],Mesh2DLineReferences.FILTERED_DIRAC,diracFileName[0]))
            # minus sign in the y component of the above equation because of something michael did (dont know why but it works)
            if self.varyingListSources:
              mkdir_p(self.config('OutDir') + '/ListSources/%s/' % self.subFolderString)
              listSourceFileName = self.config('OutDir') + '/ListSources/%s/' % self.subFolderString + '/listSources_%d.dat' % self.ID
            else:
              listSourceFileName = self.config('OutDir')+'/listSources.dat'
            with open(listSourceFileName,'w') as listSrc:
              listSrc.write('FILTERED\n')
              for src in srcLoc:
                listSrc.write('%1.16e %1.16e %1.16e\n' % (src[0],-src[1],src[2]))
              jj = 0
              for src in srcLoc:
                if len(self.config('FilteredDirac').split(',')) > 1:
                  diracFileNameCurr = diracFileName[jj%len(diracFileName)]
                else:
                  diracFileNameCurr = diracFileName[0]
                listSrc.write('SRC_USER %1.16e %1.16e %1.16e %d %s\n'\
                           % (src[0],-src[1],src[2],Mesh2DLineReferences.FILTERED_DIRAC,diracFileNameCurr))
                jj +=1
          else:
            if self.typeEquation == 'GALBRUN':
              F.write('TypeSource = SRC_DIRAC \n')
              F.write('OriginePhase = ')
              for src in srcLoc:
                F.write('%1.16e %1.16e %1.16e' % (src[0],src[1],src[2]))
              F.write('\n')
            else:
              F.write('TypeSource = SRC_DIRAC\n')
              with open(self.config('OutDir')+'/listSources.dat','w') as listSrc:
                for src in srcLoc:
                  listSrc.write('%1.16e %1.16e %1.16e\n' % (src[0],src[1],src[2]))
        else:
          raise NotImplemented('Only sources of type SRC_DIRAC are supported.')

        # Finite Elements - Mesh
        if self.typeEquation == 'GALBRUN':
          typeElement = self.config('TypeElement', 'TRIANGLE_CLASSICAL')
        else:
          typeElement = self.config('TypeElement', 'TRIANGLE_RADAU')
        F.write('TypeElement = %s\n' % typeElement)

        meshDirectory = self.config('MeshPath',pathToMPS()+'/data/meshes/')
        F.write('MeshPath = %s/\n' % meshDirectory)
        meshName =  self.bgm.nodalPoints.getFileMesh(self.freq)
        F.write('FileMesh = %s\n' % meshName)
        fileMesh = meshDirectory + meshName
        meshC = mesh(fileMesh)
        refOut = meshC.getMaxIndex('Vertices')
        if refOut == 5: # The last reference is for the convenient source
          refOut = refOut-1
        F.write('OrderDiscretization = %s\n' % self.config('OrderDiscretization','10')  )

        srcRef = self.config('SourceRefinement','5 2')
        if srcRef!='OFF' and srcType == 'SRC_DIRAC':
          srcLocStr,srcType = getSource(self.config,relocate=self.config('MoveSourceToVertex','YES').upper() == 'YES',returnMeshStr=True)
          coordsX = [] 
          coordsZ = [] 
          for src in srcLocStr:
            if self.config('MoveSourceToVertex','YES').upper() == 'NO':
              if self.config('AddVertexAtSource','YES').upper() != 'NO':
                F.write('AddVertex = %s %s\n' % (src[0],src[1]))
            elif srcRef!='OFF':
                F.write('RefinementVertex = %s %s %s\n' \
                                   % (src[0],src[1],srcRef))
            coordsX.append(src[0])
            coordsZ.append(src[1])
            
        if not self.config('TypeCurve',0):
          self.config.set('TypeCurve','%s CIRCLE' % refOut)
        self.duplicateInitToMJIni(F,'TypeCurve')

        # Boundary conditions

        dimension = int(self.config('Dimension', 3))
        if dimension == 3:
          # Add Neumann boundary condition on the axis
          F.write('ConditionReference = %d NEUMANN\n' % Mesh2DLineReferences.ROTATION_AXIS)
        F.write('ConditionReference = %d %s\n'% (refOut,self.getBoundaryConditionString()))
        self.duplicateInitToMJIni(F,'ModifiedImpedance')
        self.duplicateInitToMJIni(F,'OrderAbsorbingBoundaryCondition')
        self.duplicateInitToMJIni(F,'GibcSymmetric')
        self.duplicateInitToMJIni(F,'AddPML')
        self.duplicateInitToMJIni(F,'DampingPML')

        # Mode and Frequency
        if self.config('ForceAllMComputations',0):
          F.write('NumberModes = %d\n' % int(self.config('ForceAllMComputations',0)))
        else:
          F.write('NumberModes = SINGLE %d\n' % self.mode)
        F.write('Frequency = %f 0.0\n' % (self.freq*RSUN))

        # Input coefficients
        for line in self.coeffsLinesToWrite:
          F.write(line)

        # Output
        mkdir_p('%s/%s/%s' % (self.config('OutDir'),self.freqFolder,self.mFolder))
        F.write('DirectoryOutput = %s/%s/%s/\n' % (self.config('OutDir'),self.freqFolder,self.mFolder))
        if self.config('Output_Format',0):
          F.write('OutputFormat = %s\n'   % self.config('OutputFormat'))
        for line in self.outputLinesToWrite:
          F.write(line)

        if self.config("ComputeGradient",0) == "YES":
          F.write('ElectricOrMagnetic = -1\n')
        else:
          F.write('ElectricOrMagnetic = -2\n')

        if self.config("UseIterativeSolverM",0) == "YES":
          F.write('UseIterativeSolverM = YES\n')

        # Misc
        F.write('DataNonRegressionTest = maxwell_axi.x 0 8\n')
        F.write('PrintLevel = %s\n' % self.config('MontjoiePrintLevel',0))

        if self.config("Kernel",0):
          KernelOpts = self.config("Kernel").split()
          if not self.config("KernelPairs",0):
            raise Exception('Kernel specified, KernelPairs must also be specified')
          if len(srcLoc) < 2:
            raise Exception('No. of sources must be greater than 1')
          F.write('Kernel = %s%s%s %s\n' % (KernelOpts[0],self.freqSuffix,self.modeSuffix,' '.join(KernelOpts[1:])))
          # KernelPairStr = writeKernelPairs(self.config)
          F.write('KernelPairs = %s\n' % self.config("KernelPairs"))
  
          computeXS = self.config('Kernel',0).split()
          if computeXS:
            computeXS = (computeXS[1] == 'FORWARDMODEL')
          if not computeXS and (self.config("KernelDimension",0) and not self.config('FileOutputDisk',0) and not self.config('FileOutputSphere',0)):
            raise Exception('Can only perform integral on non-FileOutputDisk file')
          if self.config('FileOutputDisk',0):
            Nr = len(readCoordinatesOptions(self.config('FileOutputDisk').split()[self.config('FileOutputDisk').split().index('R')  :self.config('FileOutputDisk').split().index('THETA') ],self.bgm.fileName))
            Ntheta = len(readCoordinatesOptions(self.config('FileOutputDisk').split()[self.config('FileOutputDisk').split().index('THETA'):],self.bgm.fileName))
            KernelDim = self.config("KernelDimension","2D")
            F.write('KernelDimension = %s %i %i \n' % (KernelDim,Nr,Ntheta))
          elif self.config('FileOutputSphere',0):
            Nr = len(readCoordinatesOptions(self.config('FileOutputSphere').split()[self.config('FileOutputSphere').split().index('R')  :self.config('FileOutputSphere').split().index('THETA') ],self.bgm.fileName))
            Ntheta = len(readCoordinatesOptions(self.config('FileOutputSphere').split()[self.config('FileOutputSphere').split().index('THETA'):self.config('FileOutputSphere').split().index('PHI')],self.bgm.fileName))
            Nphi = len(readCoordinatesOptions(self.config('FileOutputSphere').split()[self.config('FileOutputSphere').split().index('PHI'):],self.bgm.fileName))
            KernelDim = self.config("KernelDimension","3D")
            F.write('KernelDimension = %s %i %i %i\n' % (KernelDim,Nr,Ntheta,Nphi))
          if self.config("KernelDimension","2D") == '1D':
            F.write('SismoLineAxi = 0. 0.1 0. 0.1 0. 0.1 %i \n' % Nr)
          elif self.config("KernelDimension","2D") == '0D':
            F.write('SismoPointAxi = 0.1 0.1 0.1 \n')
          if self.config("LogKernel","NO").upper() == 'YES':
            F.write('LogKernel = YES\n')

        F.close()

    def duplicateInitToMJIni(self,F,key):
        ''' if key is in init file, copies to Monjtoie ini file (F)'''
        if self.config(key,0):
          F.write(key+' = '+self.config(key)+'\n')

    def addFlowTermToLinesToWrite(self):

      if 'DOUBLE_GRAD' in self.config('TypeEquation').split():
        self.coeffsLinesToWrite.append('AddFlowTerm = GRAD\n')
      else:
        self.coeffsLinesToWrite.append('AddFlowTerm = YES\n')

    def getBoundaryConditionString(self):

      BCBase = self.config('BoundaryCondition')
      if BCBase == 'ABSORBING' and isinstance(self,Simulation1D):
        self.config('BoundaryCondition','ATMO RBC 1')
        BCBase = 'ATMO RBC 1'
        print((bColors.warning() + " ABSORBING boundary condition not compatible with 1D. Replace with ATMO RBC1. Be sure that the background was uniform close to the surface."))
      if 'ATMO' not in BCBase:
        return BCBase
      else:
        # Get maximum height of mesh
        if isinstance(self,Simulation1D):
          meshStr = self.config('MeshOptions').upper().split()
          if 'LAYERED' in meshStr:
            rMax  = float(meshStr[meshStr.index("MANUAL")-1])
          else:
            rMax  = NP.loadtxt(self.config('OutDir') + '/meshPoints.txt')[-1]
          BcMjIds = {'ATMO RBC 1':1,'ATMO HAI 0':2,'ATMO RBC HF1':3,'ATMO RBC MICRO LOCAL':5,'ATMO HAI 1':6,'ATMO RBC NON LOCAL':4}
        else:
          meshPath = os.getcwd().split('mps_montjoie')[0]+'mps_montjoie/data/meshes/'
          rMax    = getMaximumRadiusMesh(self.config('MeshPath', meshPath) + self.config('FileMesh', meshPath), fullName = True)
          BcMjIds = {'ATMO RBC 1':1,'ATMO HAI 0':2,'ATMO RBC HF1':4,'ATMO RBC MICRO LOCAL':5,'ATMO HAI 1':3}
        # Get values of drho and rho at rMax 
        drho  = self.bgm.rho.getGradient(points=[rMax,0,0])
        rho   = self.bgm.rho(points=[rMax,0,0])
       
        try:
          return 'ATMOSPHERE ABC %d %1.16e'%(BcMjIds[BCBase],-drho[0]/rho*RSUN)
        except:
          raise ValueError('Boundary condition unknown: '+BCBase+'. Valid BCs are ' + ', '.join(list(BcMjIds.keys())))

    def writeFilteredDirac(self,fileNameBase,nSrc = 0):

      ''' Reads the type of l-Filter or lomega-Filter from the config
          and write the coefficients in a file to give to Monjoie 
      '''

      options = self.config('FilteredDirac',0)
      options_PSF = self.config('PSF_OTF',0)
      FdepFreq = 0;FdepM = 0
      GdepFreq = 0;GdepM = 0

      if not options:
        toWrite = NP.ones(1500)
      else:
        options = options.split(',')[nSrc]
        Lmax = evalInt(self.config('MaximumDegree','100').split()[0])
        if Lmax > 999:
          Lmax = 999
        F    = getFilterFromString(options,Lmax+1,[self.freq])
        FdepM = F.dependsOnMMode

        # Write if necessary
        if F.dependsOnFrequency:
          FdepFreq = 1
          toWrite  = F.F_[0]
        else:
          toWrite = F.F_

      if not options_PSF:
        toWrite_PSF = NP.ones(1500)
      else:
        options_PSF = options_PSF.split(',')[nSrc]
        Lmax = evalInt(self.config('MaximumDegree','100').split()[0])
        if Lmax > 999:
          Lmax = 999
        G    = getPSFFromString(options_PSF,Lmax+1,[self.freq])
        GdepM = G.dependsOnMMode

        if G.dependsOnFrequency:
          GdepFreq    = 1
          toWrite_PSF = G.F_[0]
        else:
          toWrite_PSF = G.F_

      # Get filter amplitudes filename
      fileName = fileNameBase
      if FdepFreq + GdepFreq:      
        fileName = fileName + self.freqSuffix
      if FdepM + GdepM:
        fileName = fileName + self.modeSuffix
      fileName = fileName + '.dat'

      if not os.path.exists(fileName):
        if FdepM + GdepM:
        # Filter depends upon (l,m). We write all the coefficients
        # in one file for the 1D code and only the coefficients of
        # a specific m modes for the 2D

          if isinstance(self,Simulation1D):
            with open(fileName,'w') as OF:
              OF.write('%d\n'%(Lmax))
              for l in range(Lmax+1):
                for m2 in range(2*l+1):
                  OF.write('%1.16e\n'%toWrite[l][m2])
            fileName = 'LM ' + fileName

          else:
            m2 = 2*abs(self.mode)
            if self.mode <0: 
              m2 -= 1
            with open(fileName,'w') as OF:
              OF.write('%d\n'%(Lmax))
              for l in range(Lmax+1):
                if l>=abs(self.mode):
                  OF.write('%1.16e\n'%toWrite[l][m2])
                else:
                  OF.write('0.e0\n')
        else:
          with open(fileName,'w') as OF:
            OF.write('%d\n'%(Lmax))
            for i in range(Lmax+1):
              OF.write('%1.16e\n'%(toWrite[i]*toWrite_PSF[i]))

      return fileName

# ======================================================================

class Simulation1D(Simulation):
    ''' Radially symmetric case '''

    def __init__(self,cfg,bgm,ID,IF,freq,verbose=False,IM = 0):

        self.config     = cfg
        self.KW         = cfg.sections()[0]
        self.bgm        = bgm
        self.ID         = ID
        self.freq       = freq
        self.verbose    = verbose

        self.freqSuffix = '_f'+str(IF)
        self.freqFolder = 'f%i' % int(NP.floor(IF/NfilesPerDir)*NfilesPerDir) + '_' + 'f%i' % int(NP.floor(IF/NfilesPerDir)*NfilesPerDir + NfilesPerDir - 1)
        self.mFolder    = 'm%i' % int(NP.floor(IM/NfilesPerDir)*NfilesPerDir) + '_' + 'm%i' % int(NP.floor(IM/NfilesPerDir)*NfilesPerDir + NfilesPerDir - 1)

        self.modeSuffix = ''
        self.varyingListSources = False

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load conf for basic parameters (if not present, default values)

        self.typeEquation = cfg('TypeEquation','HELMHOLTZ').split()[0]
        if self.typeEquation == 'HELMHOLTZ':
          self.equation = EquationMJHelmholtz1D(self.bgm)
        else:
          raise ValueError('Equation type '+typeEquation+' not known or implemented for the 1D case.')

    # ==================================================================

    def setOutput(self):
        ''' generates entries for Montjoie .ini file for output'''

        self.outputLinesToWrite = []

        # ==================================================================================
        # Formats requiring points files

        options = self.config('FileOutputDisk',0)
        if options: 
          try:
            options  = options.split()
            fileName = self.config('OutDir')+'/disk_points_rth.txt'
            self.writeRTHPointsFile(fileName,options,threeDim=True)
            self.outputLinesToWrite.append('SismoPointsFileAxi = %s\n'% fileName)
            self.outputLinesToWrite.append('FileOutputPointsFileAxi = diffracJunk %s\n'% (options[0]+self.modeSuffix+self.freqSuffix))
          except:
            raise IOError('Impossible to read options for FileOutputDisk or writing points went wrong.')

        options = self.config('FileOutputPlane',0)
        if options: 
          try:
            options = options.split()
            fileName = self.config('OutDir')+'/plane_points_rth.txt'
            self.writeXZPointsFile(fileName,options)
            self.outputLinesToWrite.append('SismoPointsFileAxi = %s\n'% fileName)
            self.outputLinesToWrite.append('FileOutputPointsFileAxi = diffracJunk %s\n'% (options[0]+self.modeSuffix+self.freqSuffix))
          except:
            raise IOError('Impossible to read options for FileOutputPlane')

        options = self.config('FileOutputCircle',0)
        if options: 
          try:
            options  = options.split()
            fileName = self.config('OutDir')+'/half_circle_points_rth.txt'
            self.writeTHPointsFile(fileName,options,threeDim=True)
            self.outputLinesToWrite.append('SismoPointsFileAxi = %s\n'% fileName)
            self.outputLinesToWrite.append('FileOutputPointsFileAxi = diffracJunk %s\n'% (options[0]+self.modeSuffix+self.freqSuffix))
          except:
            raise IOError('Impossible to read options for FileOutputHalfCircle or writing points went wrong.')

        options = self.config('FileOutputSphere',0)
        if options: 
          #try:
            options  = options.split()
            fileName = self.config('OutDir')+'/sphere_points_rth.txt'
            self.writeRTHPHIPointsFile(fileName,options,threeDim=True)
            self.outputLinesToWrite.append('SismoPointsFileAxi = %s\n'% fileName)
            self.outputLinesToWrite.append('FileOutputPointsFileAxi = diffracJunk %s\n'% (options[0]+self.modeSuffix+self.freqSuffix))
          #except:
          #  raise IOError('Impossible to read options for FileOutputSphere or writing points went wrong.')

        options = self.config('FileOutputPointsFile',0)
        if options: 
          try:
            options  = options.split()
            self.outputLinesToWrite.append('SismoPointsFileAxi = %s\n'% options[1])
            self.outputLinesToWrite.append('FileOutputPointsFileAxi = diffracJunk %s\n'% (options[0]+self.modeSuffix+self.freqSuffix))
          except:
            raise IOError('Impossible to read options for FileOutputPointsFile or writing points went wrong.')

    # ==================================================================

    def writeMJiniFile(self):
        configDirectory = self.config('OutDir') + '/configFiles/%i_%i/' % \
                          (NP.floor(float(self.ID)/NfilesPerDir)*NfilesPerDir,NP.floor(float(self.ID)/NfilesPerDir)*NfilesPerDir+NfilesPerDir-1)
        mkdir_p(configDirectory)

        self.MJInputFile = configDirectory + '/' + self.config('MontjoieInputFileName','config')\
                         + '_' + str(self.ID) + '.ini'
 
        F = open(self.MJInputFile,'w')

        # Solver
        if self.config('EstimateConditionNumber',0):
          F.write('EstimationConditionNumber = YES %s\n' % self.config('EstimateConditionNumber'))     
        self.duplicateInitToMJIni(F,'ScalingMatrix')
        F.write('StorageMatrix = %s mat.dat\n' % self.config('StorageMatrix','NO'))

        # Source term
        srcLoc   ,srcType = getSource(self.config,relocate=self.config('MoveSourceToVertex','YES').upper() == 'YES')
        srcLocStr,srcType = getSource(self.config,returnMeshStr=True,relocate=self.config('MoveSourceToVertex','YES').upper() == 'YES')
        srcRef            = self.config('SourceRefinement','5 2')

        if srcType == 'SRC_DIRAC': 
          F.write('TypeSource = SRC_DIRAC %1.16e %1.16e %1.16e\n'%tuple(srcLoc[0]))
          with open(self.config('OutDir')+'/listSources.dat','w') as listSrc:
            for src in srcLoc:
              listSrc.write('SRC_DIRAC %1.16e %1.16e %1.16e\n' % tuple(src))  
        else:
          raise NotImplemented('Supported source types are SRC_DIRAC')

        # Finite Elements - Mesh
        F.write('TypeElement = EDGE_LOBATTO\n')
        F.write('OrderDiscretization = %s\n' % self.config('OrderDiscretization','10')  )

        # Refinement vertex
        if srcRef!='OFF' and srcType == 'SRC_DIRAC':
          coordsX = [] 
          coordsZ = [] 
          for src in srcLocStr:
            if (src[0] not in coordsX and src[1] not in coordsZ):
              if srcRef!='OFF':
                F.write('RefinementVertex = %s %s\n' % (src,srcRef))
                coordsX.append(src[0])
                coordsZ.append(src[1])

        # Give the updated mesh
        meshOpts = self.config('MeshOptions').split()
        if meshOpts[0].upper() == 'SAMPLE':
          try:
            try:
              rF = meshOpts[2]
              r  = NP.loadtxt(rFileName)
            except:
              r  = self.bgm.getRadius()
            r= r[::int(meshOpts[1])]
            # We insert the source points if needed
            if srcRef!='OFF':
              for src in srcLocStr:
                if NP.min(abs(r-float(src)))>1.e-15:
                  r = NP.insert(r,r.searchsorted(float(src)),float(src))
            rFileName = self.config('OutDir') + '/meshPoints.txt'
            if not os.path.exists(rFileName):
              NP.savetxt(rFileName,r)
            toWriteFileMesh=[rFileName] 
          except:
            raise Exception('Could not open file or wrong sampling ratio to create 1D mesh. Syntax is MeshOptions = SAMPLE ratio <file>')
        elif meshOpts[0].upper() == 'LAYERED':
          toWriteFileMesh = meshOpts
        else:
          raise Exception('Unknown mesh type. Should be LAYERED or SAMPLE.')

        F.write('FileMesh = %s\n' % " ".join(toWriteFileMesh))

        # Boundary conditions
        F.write('BoundaryCondition = NEUMANN %s\n'% self.getBoundaryConditionString())
        # print (self.config('IncomingWave', 0))
        if self.config('IncomingWave', 0):
          F.write('IncomingWave = %s\n' % self.config('IncomingWave'))

        # Filtered Dirac source ?
        if self.config('FilteredDirac',0) or self.config('PSF_OTF',0):
          diracFileNameBase = self.config('OutDir') + '/diracAmplitudes'
          diracFileName     = self.writeFilteredDirac(diracFileNameBase)
          F.write('FilteredDirac = %s\n' % diracFileName)

        # Mode and Frequency
        tol = evalFloat(self.config('DegreeStoppingCriterion','1.e-14'))
        if tol == 0.e0:
          F.write('NumberModes = %s\n' % self.config('MaximumDegree','100'))
        else:
          F.write('NumberModes = AUTO %s %1.16e\n' % (self.config('MaximumDegree','100'),tol) )
        self.duplicateInitToMJIni(F,'PrintL')
        # self.duplicateInitToMJIni(F,'MModes')
        F.write('MModes = %s\n' % self.config('Modes','ALL'))
        F.write('Frequency = %1.16e 0.0\n' % (self.freq*RSUN))

        # Input coefficients

        # if Damping depend on L, create file called damping_1D_f0.don which is a Lmax*3+1 length file

        for line in self.coeffsLinesToWrite:
          F.write(line)
        for line in self.coeffsLinesToWriteConvention:
          F.write(line)

        # Write Damping as function L

        # Output
        mkdir_p('%s/%s/%s' % (self.config('OutDir'),self.freqFolder,self.mFolder))
        F.write('DirectoryOutput = %s/%s/%s/\n' % (self.config('OutDir'),self.freqFolder,self.mFolder))
        if self.config('Output_Format',0):
          F.write('OutputFormat = %s\n'   % self.config('OutputFormat'))
        for line in self.outputLinesToWrite:
          F.write(line)

        if self.config("ComputeGradient",0) == "YES":
          F.write('ElectricOrMagnetic = -1\n')
        else:
          F.write('ElectricOrMagnetic = -2\n')

        # Misc
        F.write('CoordinatesLaplacian = SPHERICAL\n')
        F.write('DataNonRegressionTest = helmholtz_radial.x 0 1\n')
        F.write('PrintLevel = %s \n' % self.config('MontjoiePrintLevel',0))

        if self.config("Kernel",0):
          KernelOpts = self.config("Kernel").split()
          if not self.config("KernelPairs",0):
            raise Exception('Kernel specified, KernelPairs must also be specified')
          if len(srcLoc) < 2:
            raise Exception('No. of sources must be greater than 1')
          F.write('Kernel = %s%s %s\n' % (KernelOpts[0],self.freqSuffix,' '.join(KernelOpts[1:])))
          # KernelPairStr = writeKernelPairs(self.config)
          F.write('KernelPairs = %s\n' % self.config("KernelPairs"))
          KernelDim = self.config("KernelDimension","2D")
          Nr = len(readCoordinatesOptions(self.config('FileOutputDisk').split()[self.config('FileOutputDisk').split().index('R')  :self.config('FileOutputDisk').split().index('THETA') ],self.bgm.fileName))
          Ntheta = len(readCoordinatesOptions(self.config('FileOutputDisk').split()[self.config('FileOutputDisk').split().index('THETA'):],self.bgm.fileName))
          F.write('KernelDimension = %s %i %i \n' % (KernelDim,Nr,Ntheta))
          if self.config("KernelDimension","2D") == '1D':
            F.write('SismoLineAxi = 0. 0.1 0. 0.1 0. 0.1 %i \n' % Nr)
          elif self.config("KernelDimension","2D") == '0D':
            F.write('SismoPointAxi = 0. 0. 0. \n')

        if self.config("SaveEllCoeffs",0):
          SaveEllCoeffsOpts = self.config("SaveEllCoeffs").split()
          if SaveEllCoeffsOpts[0].upper() == 'ALL':
            F.write('SaveEllCoeffs = ALL %s \n' % self.config('MaximumDegree','100'))
          elif SaveEllCoeffsOpts[0].upper() == 'RANGE':
            F.write('SaveEllCoeffs = RANGE %s %s \n' % (SaveEllCoeffsOpts[1],SaveEllCoeffsOpts[2]))
          elif SaveEllCoeffsOpts[0].upper() == 'SINGLE':
            F.write('SaveEllCoeffs = SINGLE %s \n' % SaveEllCoeffsOpts[1])
          else:
            raise Exception('KEYWORD for SaveEllCoeffs not Recognized. Current = %s' % SaveEllCoeffsOpts[0])

        F.close()

    # ===================================================================


# ======================================================================
# writeConfigFile has to be outside for parallelization purposes
def writeConfigFile(sim):

    if not isinstance(sim,Simulation1D):
      sim.bgm.nodalPoints.checkMesh(sim.freq)
    else:
      # update the mesh to include sources
      srcLocStr,srcType = getSource(sim.config,returnMeshStr=True,relocate=sim.config('MoveSourceToVertex','YES').upper() == 'YES')
      srcRef            = sim.config('SourceRefinement','5 2')
      if srcRef!='OFF' and srcType == 'SRC_DIRAC':
        for src in srcLocStr:
          update1DMeshSource(sim.config,src)

    sim.equation.writeInputFiles(sim)
    sim.setOutput()
    sim.writeMJiniFile()
    return 1

# ======================================================================

def parseArguments(args):
  ''' Returns a list of all .init files from command line, as well as run flags '''

  paramFiles     = []
  verbose        = False
  debug          = False
  clearFiles     = False
  nbProc         = 1
  useOldConf     = False
  delConfigFiles = False
  runJobs        = True

  for arg in args:
    if (arg.split('.')[-1]=='init'):
      paramFiles.append(arg)
    elif arg == '-v':
      verbose = True
    elif arg == '-oc':
      useOldConf = True
    elif arg == '-d':
      debug = True
    elif len(arg)>3 and arg[:3] == '-np':
      nbProc = int(arg[3:])
    elif arg == '-c':
      clearFiles = True
    elif arg == '-dc':
      delConfigFiles = True
    elif arg == '-norun':
      runJobs = False

    elif arg != args[0]:
      print(('Unrecognized argument:', arg))

  if len(paramFiles)==0:
    raise Exception('At least one parameter file has to be specified!')

  return paramFiles,clearFiles,nbProc,verbose,debug,useOldConf,delConfigFiles,runJobs

# def writeKernelPairs(config):
#   if not config('Kernel',0):
#     raise Exception('Routine Writes kernel pairs, Keyword Kernel not found in config file')
#   KernelOpts = config('KernelPairs').split()
#   if KernelOpts[0].upper() == 'PT2PT':
#     if len(KernelOpts[1:]) != len(config('Source').splits(',')):
#       raise Exception('Kernel Pt2Pt chosen but Number of Sources/2 != Number of Pairs')
#     KernelPairStr =  'KernelPairs = '
#     for i in NP.arange(len(KernelOpts[1:])/2)*2:
#       KernelPairStr = KernelPairStr + '%s ' % KernelOpts[i+1] + '%s ' % KernelOpts[i+2]
#   elif KernelOpts[0].upper() == 'PT2ANNULUS':
#     ExampleSource = config('KernelSourceType').split()
#     if ExampleSource[-1].upper() != 'SPHERICAL':
#       raise Exception('error: KernelSourceType != SRC_DIRAC R THETA PHI SPHERICAL')
#     latcenter  = float(KernelOpts[1]);  latmin   = float(KernelOpts[2]);
#     latmax     = float(KernelOpts[3]);  NKernels = int(KernelOpts[4]);

#     KernelPairStr = 'KernelPairs = '
#     KernelSourceStr = 'Source = %s %1.16e %1.16e %1.16e spherical' % ()

#     for i in range(NKernels):
#       KernelPairStr = KernelPairStr + '0 ' + '%i ' % i

