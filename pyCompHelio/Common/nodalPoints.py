import numpy as NP
import os
import shutil
from . import *
from matplotlib.pyplot import *
from .. Parameters.source import getSource

class nodalPoints:
    ''' stores the nodal points for a given mesh '''

    def __init__(self,config):
        
        self.config       = config
        self.multimesh    = self.config('MultiMesh','NO')!='NO'
        self.fileName     = ''
        self.subSystem    = getClusterName()
    def getFileMesh(self,frequency=None):
        ''' returns the appropriate mesh filename, if multiple meshes were created 
            for several frequencies '''

        fileMesh = self.config('FileMesh')
        if (frequency is not None) and self.multimesh:
          try:
            frequenciesMeshes = self.config('MultiMesh').split()[1:]
          except:
            raise Exception('Unable to read frequencies for which the meshes were generated')
       
          flist = NP.zeros((len(frequenciesMeshes),)) 
          for i in range(len(frequenciesMeshes)):
            flist[i] = float(frequenciesMeshes[i])

          fmesh    = frequenciesMeshes[NP.argmin(abs(frequency*1000-flist))]
          fileMesh = fileMesh + "_" + fmesh + "mHz.mesh"
        
        return removeDoubleSlashes(fileMesh)

    def setFileName(self,frequency=None):
        ''' sets the current nodal points file name '''
        fileMesh      = cutExtension(self.getFileMesh(frequency))
        self.fileName = self.config('OutDir','.')+'/nodal_points_'+fileMesh+'.dat' 
        self.fileName = removeDoubleSlashes(self.fileName)

    def getFileName(self,frequency=None):
        ''' sets the current nodal points file name '''
        fileMesh = cutExtension(self.getFileMesh(frequency))
        return removeDoubleSlashes(self.config('OutDir','.')+'/nodal_points_'+fileMesh+'.dat')

    def getOutputFileMesh(self,frequency):
        fileMesh = self.getFileMesh(frequency)
        return self.config('OutDir','.') + '/test_' + fileMesh

    def checkMesh(self,frequency):
        ''' check if current nodal points are computed for the given frequency,
            update if necessary '''

        if self.fileName != self.getFileName(frequency):
          self.computePoints(frequency)

    def computePoints(self,frequency=None):
        ''' calls write_nodal_points.x from Montjoie to write a nodal points file
            then read this file to store information (if write_nodal_points was called,
            it is because we need it further! '''
    
        typeElement      = 'TRIANGLE_RADAU'
        fileMesh         = self.getFileMesh(frequency)
        meshPath         = self.config('MeshPath',pathToMPS()+'/data/meshes')
        order            = self.config('OrderDiscretization')
        order            = self.config('OrderDiscretizationCoef',order)
        refinementVertex = self.config('SourceRefinement','5 2')

        typeCurve        = self.config('TypeCurve','4 CIRCLE')
        writeNodalInput  = self.config('OutDir','.') + '/write_nodal.ini'
        self.setFileName(frequency)
        testMeshOutput   = self.config('OutDir','.') + '/test_' + fileMesh
        writeNodalFile   = open(writeNodalInput,'w')
          
        writeNodalFile.write  ('TypeElement = %s\n'         % typeElement     )
        writeNodalFile.write  ('MeshPath = %s/\n'           % meshPath        )
        writeNodalFile.write  ('FileMesh = %s\n'            % fileMesh        )
        writeNodalFile.write  ('OrderDiscretization = %d\n' % (int(order))    )
        writeNodalFile.write  ('PrintLevel = 0\n'                             )

        unidim = 'EDGE' in self.config('TypeElement','TRIANGLE_LOBATTO')
        srcLocs,srcTypes = getSource(self.config,relocate=self.config('MoveSourceToVertex','YES').upper() == 'YES',returnMeshStr=True)
        coordsX = [] 
        coordsZ = [] 
        for src in srcLocs:
          if (src[0] not in coordsX and src[1] not in coordsZ):
            if unidim:
              if refinementVertex!='OFF':
                writeNodalFile.write('RefinementVertex = %s %s\n' \
                                   % (src,refinementVertex))
            else:
              if self.config('MoveSourceToVertex','YES').upper() == 'NO':
                if self.config('AddVertexAtSource','YES').upper() != 'NO':
                  writeNodalFile.write('AddVertex = %s %s\n' \
                                   % (src[0],src[1]))
              elif refinementVertex!='OFF':
                writeNodalFile.write('RefinementVertex = %s %s %s\n' \
                                   % (src[0],src[1],refinementVertex))
            coordsX.append(src[0])
            coordsZ.append(src[1])

        if typeCurve:
          writeNodalFile.write('TypeCurve = %s\n'           % typeCurve       )
        writeNodalFile.close()

        print ('- Computing nodal points for mesh', fileMesh, ':')
        print ('  ~ Running write_nodal.x')

        if self.subSystem.upper() == 'TORQUE':
          Path = pathToMPS() + '/bin/torque/'
        elif self.subSystem.upper() == 'CONDOR':
          Path = pathToMPS() + '/bin/HelioCluster/'
        elif self.subSystem.upper() == 'SLURM':
          Path = pathToMPS() + '/bin/DalmaCluster/'

        command = Path + '/preprocessing/write_nodal.x %s %s %s 1>/dev/null\n'%(writeNodalInput,self.fileName,testMeshOutput)
        os.system(command)
        #os.remove(writeNodalInput)
        #shutil.move('test.mesh', testMeshOutput)
        #shutil.move('nodal_points.dat', self.fileName)
        # shutil.copy('test.mesh', testMeshOutput)
        # shutil.copy('nodal_points.dat', self.fileName)

        # Read in nodal points
        # print(self.fileName)
        with open(self.fileName) as NPF:
          line        = NPF.readline().split()
          self.N      = int(line[1])
          print ('  ~ Reading in ' + str(self.N) + ' points')
          self.points = NP.zeros((self.N,2))
          for i in range(self.N):
            self.points[i,:] = NPF.readline().split()

            
    def getCartesianCoords(self):
        ''' returns nodal points '''
        if not hasattr(self,'points'):
          #FIXME Frequency dependance
          self.computePoints()
        return self.points[:,0],self.points[:,1]

    def getPolarCoords(self):
        ''' returns nodal points '''
        if not hasattr(self,'points'):
          #FIXME Frequency dependance
          self.computePoints()
        r = NP.sqrt(self.points[:,0]**2+self.points[:,1]**2)
        NP.seterr(all='ignore')
        theta = NP.arccos(NP.divide(self.points[:,1],r))

        return r,theta 

    def plot(self,data,stride=1,**kwargs):
      
      # Check data size
      if data.ndim != 1 or data.size != self.N:
        raise Exception('Incompatible shapes for plotting on nodal points. nodal points: '+str(self.N)+' data: '+str(data.size))

      figure(figsize=(5,10))
      scatter(self.points[::stride,0],self.points[::stride,1],c=data[::stride],**kwargs)
      xlim([0,1])
      ylim([-1,1])
      savefig('checkFlow.png')


