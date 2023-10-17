import os
import numpy as NP

from .             import *
from ..Background import *
from .nodalPoints  import *
from .simulation   import *

# =========================================================================================
#
# Equations solved by MJ, each class has its method to return Montjoie coefficients
# from physical background
#
# =========================================================================================

# ==========
# Base Class

class EquationMJ:
    ''' contains all common writeIndex routines '''


    def __init__(self):
        pass

    # =================================================================================
    # Generate MJ configfile lines about the physical coefficients

    def writeNodalLine(self,coeffName,sim,fileName,\
                       nokey=False,backslash=False,isotropy='ISOTROPE'):

        mesh = sim.bgm.nodalPoints.getOutputFileMesh(sim.freq)

        if nokey:
          toWrite = ''
        else:
          toWrite = 'PhysicalMedia = ' + coeffName + ' '+str(Mesh2DSurfaceReferences.MAIN)+' '+isotropy+' '

        orderNodal = sim.config('OrderDiscretization',10)
        orderNodal = sim.config('OrderDiscretization_coef',orderNodal)

        toWrite = toWrite + ' SAME_MESH ' + mesh + ' 0.0 1.0 ' + fileName + ' ' + str(orderNodal) + ' 0.0'

        if backslash:
          toWrite = toWrite + ' \\'

        sim.coeffsLinesToWrite.append(toWrite+'\n')

    def writeRadialLine(self,coeffName,sim,fileName,\
                        nokey=False,backslash=False,isotropy='ISOTROPE'):

        mesh    = sim.config('OutDir') + sim.bgm.nodalPoints.getFileMesh(sim.freq)

        if nokey:
          toWrite = ''
        else:
          toWrite = 'PhysicalMedia = ' + coeffName + ' '+str(Mesh2DSurfaceReferences.MAIN)+' ' + isotropy + ' '

        toWrite = toWrite + ' RADIAL ' + fileName 

        if backslash:
          toWrite = toWrite + ' \\'

        sim.coeffsLinesToWrite.append(toWrite+'\n')

    def writeUniformLine(self,coeffName,sim,value,\
                         nokey=False,backslash=False,isotropy='ISOTROPE'):

        mesh    = sim.config('OutDir') + sim.bgm.nodalPoints.getFileMesh(sim.freq)

        if nokey:
          toWrite = ''
        else:
          toWrite = 'PhysicalMedia = ' + coeffName + ' '+str(Mesh2DSurfaceReferences.MAIN)+' '+isotropy+' '

        toWrite = toWrite + ' ' + str(value)

        if backslash:
          toWrite = toWrite + ' \\'

        sim.coeffsLinesToWrite.append(toWrite+'\n')

    # =================================================================================
    # Create indices files

    def writeIndexFiles(self,coefNames,fileNames,spatialTypes,dataFuncs,sim,nokeys,backslashes,isotropies):

        for i in range(len(coefNames)):
          if   spatialTypes[i] == SpatialTypes.UNIFORM:
            self.writeUniformLine(coefNames[i],sim,dataFuncs[i](sim),nokeys[i],backslashes[i],isotropies[i])
          elif spatialTypes[i] == SpatialTypes.RADIAL:
            file = sim.config('OutDir')+'/input_coef_montjoie/'+fileNames[i]+'.don'
            self.writeIndexRadial(coefNames[i],file,dataFuncs[i],sim,nokeys[i],backslashes[i],isotropies[i])
          elif spatialTypes[i] == SpatialTypes.NODAL:
            file = sim.config('OutDir')+'/input_coef_montjoie/'+fileNames[i]+'.elb'
            self.writeIndexNodal(coefNames[i],file,dataFuncs[i],sim,nokeys[i],backslashes[i],isotropies[i])

    def writeIndexRadial(self,coeffName,fileName,dataFunc,sim,nokey=False,backslash=False,isotrope=True):

        if not os.path.exists(fileName):
          data   = dataFunc(sim)
          radius = sim.bgm.getRadius()
          NP.savetxt(fileName,NP.array((radius,data)).T) 

        self.writeRadialLine(coeffName,sim,fileName,nokey,backslash,isotrope)

    def writeIndexNodal(self,coeffName,fileName,dataFunc,sim,nokey=False,backslash=False,isotrope=True):

        if not os.path.exists(fileName):
          data        = NP.nan_to_num(dataFunc(sim))
          data        = data + NP.zeros(data.shape,dtype=complex128)
          # Read number of elements from nodal points file
          meshFile    = sim.bgm.nodalPoints.getFileName(sim.freq)
          meshFile    = open(meshFile,'r')
          Nb_elements = int(meshFile.readline().split()[0])

          with open(fileName,'wb') as F:
            # File structure: data_type(int), nb_elements(int), 
            # offsets (ints, here number of points per element * element id),
            # then data(written as complex)
            type = NP.array(1,dtype=int32)
            type.tofile(F)
            nbel = NP.array(Nb_elements,dtype=int32)
            nbel.tofile(F)
            # Offsets
            order   = int(sim.config('OrderDiscretization','10'))
            offsets = (order+1)**2*NP.arange(Nb_elements+1,dtype=int32)
            offsets.tofile(F)
            # Build data array. Data is computed on the list of points at the beginning
            # of the nodal points file. But some are shared by several elements.
            # We have to go through the list of points per element.
            # - 1. Get the first line listing the points
            for i in range(len(data)):
              line = meshFile.readline() 

            # - 2. Read two lines. The first is order(elt)**2+1,
            #      the second is the list of points
            for i in range(Nb_elements):
              line = meshFile.readline()
              line = meshFile.readline()
              listpts = NP.array(line.split()).astype(int)
              dataL   = data[listpts]
              # 3. Save the corresponding array in ***.elb
              dataL.tofile(F)

        self.writeNodalLine(coeffName,sim,fileName,nokey,backslash,isotrope)

# ========================
# HELMHOLTZ (rhoc^2 divXi)

class EquationMJHelmholtz(EquationMJ):

    def __init__(self,bgm):
        self.rhoMJtype   = bgm.getTypeForCoeffs(False,'rho','c')
        self.sigmaMJtype = bgm.getTypeForCoeffs(False,'rho','c','damping')
        self.muMJtype    = bgm.getTypeForCoeffs(False,'rho')
        if hasattr(bgm,'flow'):
          self.mMJtype  = bgm.getTypeForCoeffs(False,'rho','c','M')

    def rhoMJ(self,sim):


        # points is either nodal points or None
        points = sim.bgm.getPointsForCoeffs('rho','c')
        rho    = sim.bgm.rho(nodalPoints=points)
        c      = sim.bgm.c  (nodalPoints=points)

        return 1.e0/(rho*c*c)

    def sigmaMJ(self,sim):

        points = sim.bgm.getPointsForCoeffs('rho','c','damping')
        rho    = sim.bgm.rho    (nodalPoints=points)
        c      = sim.bgm.c      (nodalPoints=points)
        gamma  = sim.bgm.damping(freq=sim.freq,nodalPoints=points,rads=True)


        return 2.e0*gamma*RSUN/(rho*c*c)

    def muMJ(self,sim):

        points = sim.bgm.getPointsForCoeffs('rho')
        rho    = sim.bgm.rho(nodalPoints=points)

        return 1.e0/rho

    def mrMJ(self,sim):

        points = sim.bgm.getPointsForCoeffs('rho','c','M')
        rho    = sim.bgm.rho(nodalPoints=points)
        c      = sim.bgm.c  (nodalPoints=points)
        mr     = sim.bgm.flow(sim.bgm.rho,nodalPoints=points)[0]

        return mr/(rho*c*c)

    def mtMJ(self,sim):

        points = sim.bgm.getPointsForCoeffs('rho','c','M')
        rho    = sim.bgm.rho(nodalPoints=points)
        c      = sim.bgm.c  (nodalPoints=points)
        mt     = sim.bgm.flow(sim.bgm.rho,nodalPoints=points)[1]

        return mt/(rho*c*c)

    def mzMJ(self,sim):

        points = sim.bgm.getPointsForCoeffs('rho','c','M')
        rho    = sim.bgm.rho(nodalPoints=points)
        c      = sim.bgm.c  (nodalPoints=points)
        mz     = sim.bgm.flow(sim.bgm.rho,nodalPoints=points)[2]

        return mz/(rho*c*c)

    def writeInputFiles(self,sim):

        indexFilesDir = sim.config('OutDir')+'/input_coef_montjoie'
        mkdir_p(indexFilesDir)

        freqSuffix = ''
        if sim.bgm.damping.dependsUponFrequency:
          freqSuffix = sim.freqSuffix

        coefNames    = ['rho','sigma','mu']
        fileNames    = ['rho','sigma'+freqSuffix,'mu']
        spatialTypes = [self.rhoMJtype,self.sigmaMJtype,self.muMJtype]
        dataFuncs    = [self.rhoMJ,self.sigmaMJ,self.muMJ]
        nokeys       = [False]*3
        backslashes  = [False]*3
        isotropies   = ['ISOTROPE']*3

        sim.coeffsLinesToWrite = []

        if hasattr(sim.bgm,'flow'):

          sim.addFlowTermToLinesToWrite()
          coefNames    += ['M','M','M']
          fileNames    += ['Mr','Mt','Mz']
          spatialTypes += [self.mMJtype]*3
          dataFuncs    += [self.mrMJ,self.mtMJ,self.mzMJ]
          nokeys       += [False,True,True]
          backslashes  += [True,True,False]
          isotropies   += ['ISOTROPE']*3

        self.writeIndexFiles(coefNames,fileNames,spatialTypes,dataFuncs,sim,nokeys,backslashes,isotropies)

# =========================
# HELIO_HELMHOLTZ (c divXi)

class EquationMJHelioHelmholtz(EquationMJ):

    def __init__(self,bgm):
        self.rhoMJtype   = SpatialTypes.UNIFORM
        self.sigmaMJtype = bgm.getTypeForCoeffs(False,'damping')
        self.muMJtype    = bgm.getTypeForCoeffs(False,'rho')
        self.alphaMJtype = bgm.getTypeForCoeffs(False,'rho','c')
        self.betaMJtype  = bgm.getTypeForCoeffs(False,'c')
        if hasattr(bgm,'flow'):
          self.mMJtype  = bgm.getTypeForCoeffs(False,'M')

    def rhoMJ(self,sim):

        return 1.e0

    def sigmaMJ(self,sim):

        points = sim.bgm.getPointsForCoeffs('damping')
        gamma  = sim.bgm.damping(freq=sim.freq,nodalPoints=points,rads=True)
        return 2.e0*gamma*RSUN

    def muMJ(self,sim):

        points = sim.bgm.getPointsForCoeffs('rho')
        rho    = sim.bgm.rho(nodalPoints=points)
        return 1.e0/rho

    def alphaMJ(self,sim):

        points = sim.bgm.getPointsForCoeffs('rho','c')
        rho    = sim.bgm.rho(nodalPoints=points)
        c      = sim.bgm.c  (nodalPoints=points)
        return rho*c

    def betaMJ(self,sim):

        points = sim.bgm.getPointsForCoeffs('c')
        c      = sim.bgm.c(nodalPoints=points)
        return c

    def mrMJ(self,sim):

        points = sim.bgm.getPointsForCoeffs('M')
        mr     = sim.bgm.flow(sim.bgm.rho,nodalPoints=points)[0]
        return mr

    def mtMJ(self,sim):

        points = sim.bgm.getPointsForCoeffs('M')
        mt     = sim.bgm.flow(sim.bgm.rho,nodalPoints=points)[1]
        return mt

    def mzMJ(self,sim):

        points = sim.bgm.getPointsForCoeffs('M')
        mz     = sim.bgm.flow(sim.bgm.rho,nodalPoints=points)[2]
        return mz

    def writeInputFiles(self,sim):

        indexFilesDir = sim.config('OutDir')+'/input_coef_montjoie'
        mkdir_p(indexFilesDir)

        freqSuffix = ''
        if sim.bgm.damping.dependsUponFrequency:
          freqSuffix = sim.freqSuffix

        coefNames    = ['rho','sigma','mu','alpha','beta']
        fileNames    = ['rho','sigma'+freqSuffix,'mu','alpha','beta']
        spatialTypes = [self.rhoMJtype,self.sigmaMJtype,self.muMJtype,self.alphaMJtype,self.betaMJtype]
        dataFuncs    = [self.rhoMJ,self.sigmaMJ,self.muMJ,self.alphaMJ,self.betaMJ]
        nokeys       = [False,False,False,False,False]
        backslashes  = [False,False,False,False,False]
        isotropies   = ['ISOTROPE']*5

        sim.coeffsLinesToWrite = []

        if hasattr(sim.bgm,'flow'):

          sim.addFlowTermToLinesToWrite()
          coefNames    += ['M','M','M']
          fileNames    += ['Mr','Mt','Mz']
          spatialTypes += [self.mMJtype]*3
          dataFuncs    += [self.mrMJ,self.mtMJ,self.mzMJ]
          nokeys       += [False,True,True]
          backslashes  += [True,True,False]
          isotropies   += ['ISOTROPE']*3

        self.writeIndexFiles(coefNames,fileNames,spatialTypes,dataFuncs,sim,nokeys,backslashes,isotropies)

# ===============================================================================
# HELMHOLTZ_GAMMA_LAPLACE (rhoc^2 divXi + term in div((mu+kappa(gamma)) grad psi)

class EquationMJHelmholtzGammaLaplace(EquationMJHelmholtz):

    def __init__(self,bgm):
        self.kRMJtype     = bgm.getTypeForCoeffs(False,'kappaR')
        self.kThetaMJtype = bgm.getTypeForCoeffs(False,'kappaTheta')
        self.kPhiMJtype   = bgm.getTypeForCoeffs(False,'kappaPhi')
        EquationMJHelmholtz.__init__(self,bgm)

    def kappaRMJ(self,sim):

        points = sim.bgm.getPointsForCoeffs('kappaR')
        return 2*(2*NP.pi*sim.freq)*sim.bgm.kappa(0,nodalPoints=points)

    def kappaThetaMJ(self,sim):

        points = sim.bgm.getPointsForCoeffs('kappaTheta')
        return 2*(2*NP.pi*sim.freq)*sim.bgm.kappa(1,nodalPoints=points)

    def kappaPhiMJ(self,sim):

        points = sim.bgm.getPointsForCoeffs('kappaPhi')
        return 2*(2*NP.pi*sim.freq)*sim.bgm.kappa(2,nodalPoints=points)

    def writeInputFiles(self,sim):

        indexFilesDir = sim.config('OutDir')+'/input_coef_montjoie'
        mkdir_p(indexFilesDir)

        freqSuffix = ''
        if sim.bgm.damping.dependsUponFrequency:
          freqSuffix = sim.freqSuffix

        coefNames    = ['rho','sigma','mu','kappa','','']
        fileNames    = ['rho','sigma'+freqSuffix,'mu',\
                        'kappaR','kappaTheta','kappaPhi']
        spatialTypes = [self.rhoMJtype,self.sigmaMJtype,self.muMJtype,\
                        self.kRMJtype,self.kThetaMJtype,self.kPhiMJtype]
        dataFuncs    = [self.rhoMJ,self.sigmaMJ,self.muMJ,\
                        self.kappaRMJ,self.kappaThetaMJ,self.kappaPhiMJ]
        nokeys       = [False,False,False,False,True,True]
        backslashes  = [False,False,False,True,True,False]
        isotropies   = ['ISOTROPE']*3+['ORTHOTROPE','','']

        sim.coeffsLinesToWrite = []

        if hasattr(sim.bgm,'flow'):

          sim.addFlowTermToLinesToWrite()
          coefNames    += ['M','M','M']
          fileNames    += ['Mr','Mt','Mz']
          spatialTypes += [self.mMJtype]*3
          dataFuncs    += [self.mrMJ,self.mtMJ,self.mzMJ]
          nokeys       += [False,True,True]
          backslashes  += [True,True,False]
          isotropies   += ['ISOTROPE']*3

        self.writeIndexFiles(coefNames,fileNames,spatialTypes,dataFuncs,sim,nokeys,backslashes,isotropies)

# ============
# GALBRUN (Xi)

class EquationMJGalbrun(EquationMJ):

    def __init__(self,bgm):
        self.rho0MJtype  = bgm.getTypeForCoeffs(False,'rho')
        self.c0MJtype    = bgm.getTypeForCoeffs(False,'c')
        self.p0MJtype    = bgm.getTypeForCoeffs(False,'p')
        self.sigmaMJtype = bgm.getTypeForCoeffs(False,'damping')
        if hasattr(bgm,'flow'):
          self.mMJtype  = bgm.getTypeForCoeffs(False,'M')

    def rho0MJ(self,sim):

        points = sim.bgm.getPointsForCoeffs('rho')
        rho    = sim.bgm.rho(nodalPoints=points)
        return rho

    def c0MJ(self,sim):

        points = sim.bgm.getPointsForCoeffs('c')
        c      = sim.bgm.c(nodalPoints=points)
        return c

    def p0MJ(self,sim):

        points = sim.bgm.getPointsForCoeffs('p')
        p      = sim.bgm.p(nodalPoints=points)
        return p

    def sigmaMJ(self,sim):

        points = sim.bgm.getPointsForCoeffs('damping')
        gamma  = sim.bgm.damping(freq=sim.freq,nodalPoints=points,rads=True)
        return 2.e0*gamma*RSUN

    def mrMJ(self,sim):

        points = sim.bgm.getPointsForCoeffs('M')
        mr     = sim.bgm.flow(sim.bgm.rho,nodalPoints=points)[0]
        return mr

    def mtMJ(self,sim):

        points = sim.bgm.getPointsForCoeffs('M')
        mt     = sim.bgm.flow(sim.bgm.rho,nodalPoints=points)[1]
        return mt

    def mzMJ(self,sim):

        points = sim.bgm.getPointsForCoeffs('M')
        mz     = sim.bgm.flow(sim.bgm.rho,nodalPoints=points)[2]
        return mz

    def writeInputFiles(self,sim):

        indexFilesDir = sim.config('OutDir')+'/input_coef_montjoie'
        mkdir_p(indexFilesDir)

        freqSuffix = ''
        if sim.bgm.damping.dependsUponFrequency:
          freqSuffix = sim.freqSuffix

        coefNames    = ['rho0','c0','p0','sigma']
        fileNames    = ['rho0','c0','p0','sigma'+freqSuffix]
        spatialTypes = [self.rho0MJtype,self.c0MJtype,self.p0MJtype,self.sigmaMJtype]
        dataFuncs    = [self.rho0MJ,self.c0MJ,self.p0MJ,self.sigmaMJ]
        nokeys       = [False,False,False,False]
        backslashes  = [False,False,False,False]
        isotropies   = ['ISOTROPE']*4

        sim.coeffsLinesToWrite = []

        if hasattr(sim.bgm,'flow'):

          sim.addFlowTermToLinesToWrite()
          coefNames    += ['M','M','M']
          fileNames    += ['Mr','Mt','Mz']
          spatialTypes += [self.mMJtype]*3
          dataFuncs    += [self.mrMJ,self.mtMJ,self.mzMJ]
          nokeys       += [False,True,True]
          backslashes  += [True,True,False]
          isotropies   += ['ISOTROPE']*3

        self.writeIndexFiles(coefNames,fileNames,spatialTypes,dataFuncs,sim,nokeys,backslashes,isotropies)

#=================================================================================================
# 1D Equations

# ==========
# Base Class

class EquationMJ1D:
    ''' contains all common writeIndex routines '''

    def __init__(self):
        pass

# ========================
# HELMHOLTZ (rhoc^2 divXi)

class EquationMJHelmholtz1D(EquationMJ1D):

    def __init__(self,bgm):
        self.rhoMJtype   = bgm.getTypeForCoeffs(False,'rho'    )
        self.cMJtype     = bgm.getTypeForCoeffs(False,'c'      )
        self.gammaMJtype = bgm.getTypeForCoeffs(False,'damping')

    def rhoMJ(self,sim):

        return sim.bgm.rho()

    def rhoPdivrhoMJ(self,sim):

        if self.rhoMJtype == SpatialTypes.UNIFORM:
          return 0.e0
        return sim.bgm.rho.getGradient()[0]/sim.bgm.rho()*RSUN

    def cMJ(self,sim):

        return sim.bgm.c()

    def gammaMJ(self,sim):
        # print sim.bgm.damping.L_DEP
        if sim.bgm.damping.typeFreq == DampingTypes.L_DEP:
          indexFilesDir = sim.config('OutDir')+'/input_coef_montjoie'
          if sim.bgm.damping.dependsUponFrequency:
              freqSuffix = sim.freqSuffix
              freqFolder = sim.freqFolder + '/'
              indexFilesDir = indexFilesDir + '/GammaFiles/' + freqFolder
              mkdir_p(indexFilesDir)

          freqSuffix = sim.freqSuffix

          damping = RSUN*sim.bgm.damping(freq=sim.freq)*2*NP.pi

          NP.savetxt(indexFilesDir + '/gamma%s.don' % freqSuffix,damping)
          return RSUN*1e-5*2*NP.pi
        else:
          return RSUN*sim.bgm.damping(freq=sim.freq)*2*NP.pi

    def writeInputFiles(self,sim):

        indexFilesDir = sim.config('OutDir')+'/input_coef_montjoie'
        mkdir_p(indexFilesDir)
        mkdir_p(indexFilesDir + '/GammaFiles/')

        freqSuffix = ''
        freqFolder = ''
        gammaFileName = 'gamma'
        if sim.bgm.damping.dependsUponFrequency:
          freqSuffix = sim.freqSuffix
          freqFolder = sim.freqFolder + '/'
          mkdir_p(indexFilesDir + '/GammaFiles/' + freqFolder)
          gammaFileName = 'GammaFiles/' + freqFolder + 'gamma' + freqSuffix


        coefNames    = ['rho','rhopdivrho','c','gamma']
        fileNames    = ['rho','rhopdivrho','c',gammaFileName]
        spatialTypes = [self.rhoMJtype,self.rhoMJtype,self.cMJtype,self.gammaMJtype]
        dataFuncs    = [self.rhoMJ,self.rhoPdivrhoMJ,self.cMJ,self.gammaMJ]

        propertyLine = ""

        if sim.bgm.damping.DampingRW:
            spatialTypes[-1] = 1


        for i in range(len(coefNames)):
          if spatialTypes[i] == SpatialTypes.RADIAL :
            fileName = sim.config('OutDir')+'/input_coef_montjoie/'+fileNames[i]+'.don'
            if not os.path.exists(fileName):
              data   = dataFuncs[i](sim)
              radius = sim.bgm.getRadius()
              NP.savetxt(fileName,NP.array((radius,data)).T) 
            if i!= 0:
              propertyLine += " SPLINE " + fileName
          else:
            if i!= 0:
              propertyLine += " " + str(dataFuncs[i](sim))


        # Get number of mesh parts
        meshLine = sim.config('MeshOptions').upper().split()
        if meshLine[0] == 'LAYERED':
          nSegs = meshLine.index('MANUAL') - meshLine.index('LAYERED') - 2
        else:
          nSegs = 1
        sim.coeffsLinesToWrite           = []
        sim.coeffsLinesToWriteConvention = []
        for i in range(nSegs):
          sim.coeffsLinesToWrite.append("MateriauDielec = "+str(i+1)+" "+propertyLine+"\n")
          if self.rhoMJtype == SpatialTypes.RADIAL:
            fileName = sim.config('OutDir')+'/input_coef_montjoie/'+fileNames[0]+'.don'
            sim.coeffsLinesToWriteConvention.append("ConventionPhysicalIndex = "+"Helio " + str(i+1) + " SPLINE " + fileName +"\n")
          else:
            sim.coeffsLinesToWriteConvention.append("ConventionPhysicalIndex = "+"Helio " + str(i+1) + " " + str(dataFuncs[0](sim)) + "\n")

        if sim.bgm.damping.typeFreq == DampingTypes.L_DEP:

          MateriauDielecFolder =  indexFilesDir + '/MateriauDielecFiles/%s' % sim.freqFolder
          mkdir_p(MateriauDielecFolder)
          fileNAME = '%s/MateriauDielec%s.ini' % (MateriauDielecFolder,freqSuffix)
          sim.MateriauDielecPath = fileNAME
          F = open(fileNAME,'w')
          F.write('%d \n' % nSegs)
          gammaFile = indexFilesDir + '/GammaFiles/' + freqFolder+'/gamma%s.don' % freqSuffix
          GAMMA = NP.genfromtxt(gammaFile)
          for ii in range(len(GAMMA)):
            gAMMA = GAMMA[ii]
            MatDieLine = propertyLine.split()
            MatDieLine[-1] = str(gAMMA)
            MatDieLine = ' '.join(tuple(MatDieLine))
            for i in range(nSegs):
              F.write(str(i+1)+" "+MatDieLine+"\n")
          # remove the gamma_f... file as it is not necessary for montjoie
          os.system('rm -rf %s' % gammaFile)



          F.close()

