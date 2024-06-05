import numpy as NP
from ..Common     import *
from ..Parameters import *
from ..Background import flow
import time

class Green(object):

    ''' Class containing information on how a Green's function can be loaded/computed.
        If need be/if possible, the actual values of G are stored.
    '''

    def __init__ (self,params,loadGrad = False,\
                  observable = TypeOfObservable.rhoc2DivXi,\
                  G = None, loadAll = True,\
                  onTheFly = None, onTheFlyDel = True, onTheFlyPrev = None,\
                  nSrc = 0, parallelM = 1, MFFT = False, computeLModes = None, \
                  axisL = -1,degree = None,pgBar=True,reverseFlow=False,\
                  ComputeMissing = False,onTheFlyOptions = None):

        ''' Initialization of a Green's function.
               params : type of output, geometry and time parameters.
             loadGrad : store/load gradient from MJ as well
                    G : a green's function can be instantiated with data.
                        if getGrad G must be a list [G,dG]
              loadAll : if True, stores the values in the class attribute data_
                        upon loading/computation
             onTheFly : if an init file is given computation is performed
                        with updated mode and frequency
          onTheFlyDel : delete or not the results of on the fly computation
                        results can be kept if the same computation is done for multiple
                        sources or rotations
         onTheFlyPrev : if onTheFly, onTheFlyPrev will read results from another Green's function
                        set with onTheFlyDel to False
                 nSrc : if load from Montjoie, load Green's function 
                        corresponding to source index nSrc in the list 
                        of sources given in the init file.
            parallelM : for each frequency, parallelize loading in M as well
                        (int, nbProcs per freq)
                 MFFT : perform FFT in phi direction in case of several modes
        computeLModes : array of Ls to project the Green's function on
        reverseFlow   : Green's function with -u in the equation. It should contain a configuration file 
                        with -u if one wants to load directly the data. 
                        Otherwise, it will be computed on the fly.
        '''

        self.params_       = params
        self.observable_   = observable
        self.nSrc_         = nSrc

        self.loadGrad_     = loadGrad
        self.parallelM_    = parallelM
        self.MFFT_         = MFFT
        self.loadAll_      = loadAll
        self.onTheFly_     = onTheFly
        self.onTheFlyDel_  = onTheFlyDel
        self.onTheFlyPrev_ = onTheFlyPrev
        if onTheFly:
          self.loadAll_ = False
        self.computeLModes_ = computeLModes
        self.axisL_         = axisL
        self.iLFreq_        = -1
        self.unidim_        = 'EDGE' in self.params_.config_('TypeElement','TRIANGLE_LOBATTO')
        self.pgBar_         = pgBar
        self.onTheFlyOptions_ = onTheFlyOptions
        if degree is not None:
          self.SeperateDegrees_ = True
        else:
          self.SeperateDegrees_ = False
        self = self.getFlowType()
        self.EllCoeffs_ = self.params_.config_('SaveEllCoeffs',0) and self.params_.unidim_

        # Search the output directory and compute the missing files
        if ComputeMissing:
          Nmissing = self.ComputeMissingGreens()
          return


        # Check if the Green's function was already computed
        # and store it if necessary        
        if self.loadAll_:

          # load in ell coeffs if need be
          if self.EllCoeffs_:
            G = loadGreenEllCoeff(params,nSrc = nSrc,observable = observable,pgBar = pgBar,Grad = loadGrad)

          # G is not given : load from Montjoie
          if G is None:
            if self.isComputedMJ(reverseFlow=reverseFlow):
              G = loadGreen(params,getGrad=loadGrad,observable=observable,\
                            getSrc=nSrc,parallelM=parallelM,MFFT=MFFT,\
                            degree=degree, pgBar=pgBar)
              if reverseFlow: # reverseFlow should contain the name of the configuration file with -u
                paramsR = parameters(reverseFlow, self.params_.typeOfOutput_)
                Gr = loadGreen(paramsR,getGrad=loadGrad,\
                               observable=observable, getSrc=nSrc, \
                               parallelM=parallelM, \
                               MFFT=MFFT,degree=degree,pgBar=pgBar)

            else:
              raise Exception("Missing or too many files. Green's function can not be loaded")

          if loadGrad:
            self.set(G[0])
            self.setGrad(G[1])
            if reverseFlow:
              self.setReverse(Gr[0])
              self.setGradReverse(Gr[1])
          else:
            self.set(G)
            if reverseFlow:
              self.setReverse(Gr)

          if computeLModes is not None:
            self.GL_ = projectOnLegendre(self.data_,computeLModes,axisTheta=axisL,normalized=True)  
            if loadGrad:
              self.drGL_ = projectOnLegendre(self.gradData_[0],computeLModes,axisTheta=axisL,normalized=True)

            if reverseFlow:
              self.GL_ = projectOnLegendre(self.dataR_,computeLModes,axisTheta=axisL,normalized=True)  
              if loadGrad:
                self.drGL_ = projectOnLegendre(self.gradDataR_[0],computeLModes,axisTheta=axisL,normalized=True)

    # ==================================================================
    
    def isComputedMJ(self,iM=None,iF=None,reverseFlow=False):

       ''' Determines if MJ wrote solution files.
           By default, checks if the number of files match.
           If iM and iF are provided: check these specific files.
       '''

       prefix   = getFileNamePrefix(self.params_,self.nSrc_)
       outDir   = self.params_.config_('OutDir')
       Nm       = getModes(self.params_.config_)[2]
       if self.params_.config_('ForceAllMComputations',0):
        Nm = 1
       Nf       = getFrequencies(self.params_.config_)[2]
       if self.unidim_:
         Nl       = int(self.params_.config_('MaximumDegree'))+1

       if reverseFlow and self.needSimulationReverseFlow()[0]:
         factor = 2
       else:
         factor = 1 
       if iM is None and iF is None:
         suffix = '*'
         nFiles = Nm*Nf*factor
       elif iM is None:
         suffix = '_m*_f%d' % iF
         nFiles = Nm*factor
       elif iF is None:
         suffix = '_m%d_f*' % iM
         nFiles = Nf*factor
       else:
         suffix = '_m%d_f%d' % (iM,iF)
         nFiles = factor
       suffix += '_U0.dat'
       extraString = ''

       if self.unidim_:
        if self.SeperateDegrees_:
          suffix = '_f*_L*'
          nFiles = Nf*Nl*factor
        else:
          suffix = '_f*_U0.dat'
          nFiles = Nf*factor
          extraString = "| sed '/_L[0-9]*\.dat/d' "
 
       #moreThanSuffix = '_U0'

       fileName = outDir+'/f*/m*/.'#+prefix+suffix


       subP = subprocess.Popen('find %s -type f %s | grep "_U0" | wc -l'\
                               %(fileName,extraString),\
                               shell=True,stdout=subprocess.PIPE,\
                               stderr=subprocess.PIPE)

       out,err = subP.communicate()

       return (len(err) == 0 and int(out) == nFiles)

    # ==================================================================

    def set(self,G):
      self.data_ = G

    def setGrad(self,dG):
      self.gradData_ = dG

    def setReverse(self,Gr):
      self.dataR_ = Gr

    def setGradReverse(self,dGr):
      self.gradDataR_ = dGr

    # ==================================================================

    def __call__(self,freq=None,ifreq=None,coords=None,icoords=None,grad=False,lFilter = None,reverseFlow=False,singleM=False):
      return self.get(freq,ifreq,coords,icoords,grad,lFilter,reverseFlow,singleM)

    def get(self,freq=None,ifreq=None,coords=None,icoords=None,grad=False,lFilter = None,reverseFlow=False,singleM=False):


      ''' Returns the values of the Green's function
          for all or specified frequencies.
          If the values are not stored, load or compute on the fly.
      '''

      # Set frequencies to load
      if freq is not None:
        # Single frequency
        if isinstance(freq,float):
          ifreq = [NP.argmin(abs(freq-self.params_.time_.omega_/(2.e0*pi)))]
        else:
          ifreq = []
          for f in freq:
            ifreq.append(NP.argmin(abs(f-self.params_.time_.omega_/(2.e0*pi))))
      elif ifreq is not None:
        if isinstance(ifreq,int):
          ifreq = [ifreq]
 
      # Initialize all possible Green's function to return to None
      G = None; dG = None; Gr = None; dGr = None
      GMm = None; dGMm = None; GrMm = None; dGrMm = None 
      toReturn = [ None ] * 8

      # Solution was already stored
      if hasattr(self,'data_'):
        # print ifreq
        if ifreq is None:
          if not grad:
            toReturn[0] = self.data_
          else:
            toReturn[0]  = self.data_
            toReturn[1] = self.gradData_
        else:
          if not grad:
            toReturn[0] = self.data_[...,ifreq]
          else:
            toReturn[0] = self.data_[...,ifreq]
            toReturn[1] = self.gradData_[...,ifreq]

      # On the fly loading or computing
      else:
        if ifreq is None:
          ifreq = list(range(getFrequencies(self.params_.config_)[2]))


        for iF in ifreq:
          # Check if the file already exists
          
          if self.isComputedMJ(iF, reverseFlow=reverseFlow) and not self.onTheFly_:
            Gfreq = self.loadAllGreen(iF,self.params_.config_,grad)

          # Compute on the fly
          elif self.onTheFly_:
            Gfreq = self.computeOnTheFly(iF,grad,reverseFlow,singleM)
          else:
            raise Exception('File for frequency #%d not found.'%iF)

          for i in range(len(toReturn)):
            if Gfreq[i] is not None:
              if toReturn[i] is None: # first frequency
                toReturn[i] = [Gfreq[i]]
              else:
                toReturn[i].append(Gfreq[i])


        # Put the frequency at the end
        for i in range(len(toReturn)):
          if toReturn[i] is not None:
            toReturn[i] = NP.array(toReturn[i])
            toReturn[i] = NP.rollaxis(toReturn[i],0, toReturn[i].ndim)

      # Remove the last dimension if there is only one frequency
      for i in range(len(toReturn)):
        if toReturn[i] is not None:
          toReturn[i] = NP.array(toReturn[i])
          if toReturn[i].shape[-1] == 1:
            toReturn[i] = toReturn[i][...,0]

      # Filter Data and Grad
      if lFilter is not None:
        if ifreq is None:
          OMEGA = None
        else:
          OMEGA = self.params_.time_.omega_[ifreq]
        G = lFilter.apply(G,self.params_.geom_.theta(),omega = OMEGA,\
                          axisTheta=self.params_.geom_.axisTheta(),pgBar = self.pgBar_)
        if grad:
          dG = lFilter.apply(dG,self.params_.geom_.theta(),omega = OMEGA,\
                          axisTheta=self.params_.geom_.axisTheta()+1,pgBar = self.pgBar_)

      # Return value at a specific point
      if coords is None and icoords is None:
        return toReturn

      else:
        if coords is not None:
          icoords = []
          for i in range(len(coords)):
            icoords.append(NP.argmin(abs(self.params_.geom_.coords_[i]-coords[i])))

          for i in range(len(toReturn)):
            if toReturn[i] is not None:
              if i%2 == 0: # G
                toReturn[i] = toReturn[i][icoords][0]
              else: # dG
                for j in range(len(toReturn[i])):
                  toReturn[i][j] = toReturn[i][j,icoords][0]

          return toReturn


    # ==================================================================

    def loadAllGreen(self,iF,config=None,grad=False,configFileReverseFlow='', configFileReverseM='', configFileReverseMFlow=''):
        Gfreq = [ None ] * 8
        if config is None:
          config = self.params_.config_

        configFiles = [config.fileName_, configFileReverseFlow, configFileReverseM, configFileReverseMFlow]
        # print(configFiles)
        modes = getModes(config)[0]
        for i in [0]:#range(len(configFiles)):
          if configFiles[i] == '':
            # The Green's function can be guessed from another simulation
            if self.flow_:
              if i==1:
                # Get G_m(-u) from G_m(u)
                reverseFlow, reverseM = self.needSimulationReverseFlow(modes)
                GMutmp = self.getGReverseFlowFromG(Gfreq[0],reverseM,iF)
                Gfreq[4] = GMutmp[0]; Gfreq[6] = GMutmp[1]
              elif i==2:
                # Get G_{-m}(u) and G_{-m}(-u) from G_m(u) and G_m(-u)
                GMmtmp = self.getGReverseMFromG(Gfreq[0],Gfreq[2],modes[0],iF)
                Gfreq[4] = GMmtmp[0]; Gfreq[6] = GMmtmp[1]
                if grad:
                  dGMmtmp =self.getGReverseMFromG(Gfreq[1],Gfreq[3],modes[0],iF)
                  Gfreq[5] = dGMmtmp[0]; Gfreq[7] = dGMmtmp[1]
                break
              else:
                raise Exception('A config file for -m and -u should be given') 

          else:
            # load the Green's function from config file
            params = parameters(configFiles[i], self.params_.typeOfOutput_)
            if self.EllCoeffs_:
              Gtmp = loadGreenEllCoeff(params,nSrc = self.nSrc_,iF = iF,\
                                      observable = self.observable_,\
                                      pgBar = False,Grad = grad)
            else:
              Gtmp = loadGreenFreqInd(params,iF,getGrad=grad,\
                                      observable=self.observable_,\
                                      getSrc=self.nSrc_,\
                                      parallelM=self.parallelM_,\
                                      MFFT=self.MFFT_)
            if grad:
              Gfreq[2*i] = Gtmp[0]; Gfreq[2*i+1] = Gtmp[1]
            else:
              Gfreq[2*i] = Gtmp

        return Gfreq



    # ==================================================================
    def getGrad(self,freq=None,ifreq=None,coords=None,icoords=None,reverseFlow=False):
      
      ''' Same as get.
          Except that the Gradient is computed from G 
          if not provided by Montjoie.
      '''

      if self.loadGrad_:
        return self.get(freq,ifreq,coords,icoords,grad=True,reverseFlow=reverseFlow)
      else:
        G    = self.get(freq,ifreq,reverseFlow=reverseFlow)
        if reverseFlow:
          Gr = G[1]
          G  = G[0]
        dims = list(G.shape)
        dims.insert(0,3)
        if ( freq is not None and len( freq)==1)\
        or (ifreq is not None and len(ifreq)==1):
          dG = self.params_.geom_.sphericalGrad(G)
          if reverseFlow:
            dGr = self.params_.geom_.sphericalGrad(Gr)
        else:
          dG  = []
          for iw in range(G.shape[-1]):
            dG.append(self.params_.geom_.sphericalGrad(G[...,iw]))
          dG = NP.array(dG)

          if reverseFlow:
            dGr = []
            for iw in range(Gr.shape[-1]):
              dGr.append(self.params_.geom_.sphericalGrad(Gr[...,iw]))
            dGr = NP.array(dGr)


        if coords is None and icoords is None:
          if reverseFlow:
            return G,dG,Gr,dGr
          else:
            return G,dG
        else:
          if coords is not None:
            icoords = []
            for i in range(len(coords)):
              icoords.append(NP.argmin(abs(self.params_.geom_.coords_[i]-coords[i]))) 

          if reverseFlow:
            return G[icoords],dG[:,icoords],Gr[icoords],dGr[:,icoords]
          else:
            return G[icoords],dG[:,icoords]

    # ==================================================================

    def computeOnTheFly(self,iF,grad=False,reverseFlow=False,singleM=False,Load=True):

      ''' Runs a Montjoie computation, based on the initFile
          stored in self.onTheFly_
      '''

      outDir = self.params_.config_('OutDir')
      # Initialize all possible Green's function to return to None
      G = None; dG = None; Gr = None; dGr = None
      GMm = None; dGMm = None; GrMm = None; dGrMm = None 
      fileIDr = ''; fileIDMm = ''; fileIDrMm=''

      if not self.onTheFlyPrev_:
        # save config in a temporary init file
        # name is randomly generated to avoid writing in the same file
        freq   = self.params_.time_.omega_[iF]/(2.e0*NP.pi)
        NP.random.seed(iF)
        ID     = int(NP.random.rand(1)*1.e32)
        config = myConfigParser(self.onTheFly_)
        config.set('Frequencies','SINGLE %1.16e' % freq)
        config.set('InteractiveMode','YES')
        config.set('OutDir','%s/%d/' % (outDir,ID))
        if grad:
          config.set('ComputeGradient','YES')
        if self.onTheFlyOptions_ is not None:
          for i in range(len(self.onTheFlyOptions_)):
            OPTION = self.onTheFlyOptions_[i]
            config.set('%s' % OPTION[0],'%s' % OPTION[1])

        fileID = '%s/tmp_%d.init' % (outDir,ID)
        mkdir_p(outDir)
        config.save(fileID)

        if singleM:
          configFile = self.writeConfigM(config, singleM)
          os.system('mv %s %s' % (configFile, fileID))
        config = myConfigParser(fileID)

        # Creates a second config file for the reverse flow if necessary
        if self.flow_ and self.needSimulationReverseFlow(singleM)[0] and reverseFlow:
          configR = myConfigParser(fileID)
          fileIDr = self.writeConfigReverseFlow(configR,singleM)


        if self.flow_ and self.needSimulationReverseFlow(singleM)[1]:
          configMm = myConfigParser(fileID)
          fileIDMm = self.writeConfigM(config,-singleM)
          if reverseFlow and self.needSimulationReverseFlow(singleM)[0]:
            configRMm = myConfigParser(fileIDr)
            fileIDrMm = self.writeConfigM(configRMm,-singleM)

        print("\n=> On the fly RUN_MONTJOIE")

        if DEBUG:
          print("Folder ID %d" %ID)
          print("Config:")
          print(list(config.items()))


        command = 'cd %s/pyCompHelio/RunMontjoie; ./runMontjoie.py %s %s %s %s' % (pathToMPS(),fileID, fileIDr,fileIDMm,fileIDrMm)
        print(command)
        os.system(command)

      else:
        # Use a green's function previously computed on the fly
        try:
          ID = self.onTheFlyPrev_.onTheFlyID_
        except:
          raise Exception("Previous on the fly green's function has not generated an ID yet")

        self.onTheFlyID_ = ID
        fileID           = '%s/tmp_%d.init' % (outDir,ID)
        if self.flow_ and self.needSimulationReverseFlow(singleM)[0] and reverseFlow:
          fileIDr = '%s/%s/rFlow/tmp_%d.init' % (outDir,ID,ID)

        if self.flow_ and self.needSimulationReverseM() and reverseFlow:
          fileIDMm = '%s/m%s/tmp_%d.init' % (outDir,singleM,ID)
          fileIDrMm = '%s/m%s/tmp_%d.init' % (outDir,-singleM,ID)
      # ----------------------------------------------------------------
      # Load computed green's function
      if Load:

        if self.params_.config_('SaveEllCoeffs',0) and self.params_.unidim_:
          Gs1D = loadGreenEllCoeff(parameters(fileID,self.params_.typeOfOutput_),\
                                    nSrc = self.nSrc_ ,observable = self.observable_,\
                                    pgBar = False,Grad = grad)
          Gfreq = [None] * 8
          if grad:
            Gfreq[0] = Gs1D[0]
            Gfreq[1] = Gs1D[1]
          else:
            Gfreq[0] = Gs1D
        else:
          Gfreq = self.loadAllGreen(0,myConfigParser(fileID),grad,fileIDr,fileIDMm, fileIDrMm)

        if self.computeLModes_ is not None:
          print("Project solution on L modes")
          if reverseFlow:
            toProject = Gr
          else:
            toProject = G
          if grad:
            if reverseFlow:
              dtoProject = dGr
            else:
              dtoProject = dG
          self.GL_     = projectOnLegendre(toProject,self.computeLModes_,axisTheta=self.axisL_,normalized=True)
          if grad:
            self.drGL_ = projectOnLegendre(dtoProject[0],self.computeLModes_,axisTheta=self.axisL_,normalized=True)
          self.iLFreq_ = iF
   
        # Clean
        if self.onTheFlyDel_:
          remove(fileID)
          if reverseFlow: 
            remove(fileIDr)
          remove('%s/%d'%(outDir,ID))
        else:
          self.onTheFlyID_ = ID

        return Gfreq
      else:
        return outDir + '/%d' % ID

    # ==================================================================
   
    def getSourceLocation(self,coords="cartesian"):

      srcLoc = getSource(self.params_.config_)[0][0]

      if coords=="cartesian":
        return srcLoc
      elif coords=="spherical":
        return cartesianToSpherical([srcLoc])[0]

    def ComputeMissingGreens(self):
      config = self.params_.config_
      outDir = config('OutDir')
      hn     = getHostname()  
      binDir = pathToMPS() + 'bin/'
      Memory = 1024 *1024 * int(config('SizePerJob','8'))
      if 'helio' in hn:
        binDir = binDir+'HelioCluster/'
      limit  = self.params_.time_.limit_
      TypeEquation = self.params_.config_('TypeEquation').split()[0]
      if self.unidim_:
        if   TypeEquation in ['HELMHOLTZ']:
          MJ_exe = 'helm_radial'
        else:
          raise Exception('Not implemented')
      else:
        if   TypeEquation in ['HELMHOLTZ']:
          MJ_exe = 'helmholtz_axi'
        elif TypeEquation in ['HELMHOLTZ_GAMMA_LAPLACE']:
          MJ_exe = 'HELIO_helmholtz_axi_GL'
        elif TypeEquation in ['HELIO_HELMHOLTZ']:
          MJ_exe = 'HELIO_helmholtz_axi'
        elif TypeEquation in ['HELIO_HELMHOLTZ_V2']:
          MJ_exe = 'HELIO_helmholtz_axi_V2'
        elif TypeEquation in ['GALBRUN']:
          MJ_exe = 'galbrun_axi'

      # Only serial is implemented
      MJ_exe = binDir +'/solvers/'+ MJ_exe + '_serial.x'
      mkdir_p(outDir + '/Logs/')
      mkdir_p(outDir + '/ResubmitFiles/')
      os.system('rm -rf ' + outDir + '/Logs/*')
      os.system('rm -rf ' + outDir + '/ResubmitFiles/*')

      pb = progressBar(limit,'serial')
      NoMissing  = 0
      ind_Missing = []
      for i in range(limit):
        submitFile = outDir + '/ResubmitFiles/RESUBMIT_%i.sub' % i
        
        FILE = GetFileName(self,iM=0,iF=i)[0]
        if not os.path.isfile(FILE):
          NoMissing = NoMissing + 1
          ind_Missing.append(i)
          f = open(submitFile,'w')
          if 1:#'seismo1' in hn:
            f.write("Universe       = vanilla \n")
            f.write("Executable     = %s \n" % MJ_exe)
            f.write("arguments     = %s/config_%i.ini %s/listSources.dat\n" % (outDir,i,outDir))
            f.write("output         = %s/Logs/output_%i.out\n" % (outDir,i))
            f.write("error          = %s/Logs/error_%i.out\n" % (outDir,i))
            f.write("log            = %s/Logs/log_%i.out\n" % (outDir,i))
            f.write("Image_Size     = %i\n" % Memory)
            f.write("Queue\n")
            f.close()
            os.system("condor_submit %s" % submitFile)

          #else:# 'helio' in hn:
          #  f.write('#!/bin/bash\n')
          #  f.write('#PBS -N %s/RESUBMIT_%i.sub\n' % (outDir,i))
           # # ETA : 10 minutes
          #  duration = int(10)
          #  f.write('#PBS -l nodes=1:ppn=1,walltime=%d:%d:00\n' % (duration/60,duration%60))
          #  f.write('#PBS -q helioq\n')
          #  f.write('#PBS -o %s/Logs/$PBS_JOBID.out\n' % outDir)
          #  f.write('#PBS -e %s/Logs/$PBS_JOBID.err\n' % outDir)
          #  f.write('echo -e "Job started on" `date` "\\n\\n\\n"\n')
          #  f.write('cd %s/solvers\n' % binDir)
          #  if self.params_.bgm_.damping.typeFreq == DampingTypes.L_DEP:
          #    oneD_L = '%s/input_coef_montjoie/MateriauDielec_f%s.ini' % (outDir,i)
          #  else:
          #    oneD_L = ''
          #  f.write('./%s %s/config_%d.ini %s/listSources.dat %s\n'%(MJ_exe,outDir,i,outDir,oneD_L))
          #  f.write('echo -e "\\n\\n\\nJob finished on" `date`\n')
          #  f.close()
          #  os.system('qsub %s 1>/dev/null' % submitFile)

          # else:
            # raise Exception('Cluster unknown')

          
        NP.savetxt('%s/ResubmitFiles/Missing_ind.txt' % outDir,ind_Missing,fmt='%i')
        pb.update()

      del pb

      print('%i Files missing' % NoMissing)

    def needSimulationReverseFlow(self,singleM=False):
      '''Check if the Green's function with reverse flow can be obtained from the one with the flow or if a new simulation is necessary.'''
      if singleM:
        modes = [singleM]
      else:
        modes = self.params_.getModes()[0]

      reverseFlow = True
      reverseM = False

      flow =  self.params_.config_('Flow', 'CONSTANT 0 0 0').split()

      if flow[0] == 'CONSTANT' and float(flow[1]) == 0 and float(flow[2]) == 0:
        if evalFloat(flow[3]) != 0:
          # u_\phi component is non-zero, G_m(-u) = G_{-m}(u)
          reverseM = True        
        reverseFlow = False

      if flow[0] == 'DIFFERENTIAL_ROTATION' or flow[0] == 'NODAL_LONGI':
        # only u_\phi component is non-zero, G_m(-u) = G_{-m}(u)
        reverseM = True
        reverseFlow = False 

      return reverseFlow, reverseM

    def getFlowType(self):
      bgm = self.params_.bgm_
      self.flow_      = False
      self.meridFlow_ = False
      self.rotFlow_   = False
      if hasattr(bgm,'flow'):
        self.flow_    = True
        self.meridFlow_ = bgm.flow.type in [FlowTypes.SUPERGRANULE,\
                                            FlowTypes.MERIDIONAL_CIRCULATION,\
                                            FlowTypes.NODAL_MERID] 
        self.rotFlow_   = bgm.flow.type in [FlowTypes.DIFFERENTIAL_ROTATION,\
                                            FlowTypes.NODAL_LONGI] 
      return self

    def needSimulationReverseM(self):
      '''If the flow if purely rotational and meridional G_{-m} can be guessed from G{m} and the function returns false. Otherwise returns True.''' 
      if self.flow_:
        if self.meridFlow_ and self.rotFlow_:
          return True
        else:
          return False
      else:
        return False

    def writeConfigReverseFlow(self, oldConfig, singleM):
      '''Flip the sign of the flow in the config file and change the filenames of the outputs. Returns the new config.'''
      configR = myConfigParser(oldConfig.fileName_)
      newFlow = Flow.writeReverseFlow(oldConfig)
      configR.set('Flow', newFlow) 
      if singleM:
        configR.set('Modes', 'SINGLE %i' % singleM)
      for outputType in ['FileOutputCircle','FileOutputDisk','FileOutputPlane','FileOutputSphere']:
        if oldConfig.has_option(oldConfig.sections()[0], outputType):
          outputCrt = oldConfig(outputType).split()
          outputCrt[0] += '_rFlow'
          configR.set(outputType, ' '.join(outputCrt))
      configFileR = self.getNewConfigNameReverseFlow(oldConfig)
      newOutDir = configR('OutDir') + '/rFlow/'
      configR.set('OutDir', newOutDir)
      configR.save(configFileR)
      return configFileR

    def getNewConfigNameReverseFlow(self,oldConfig):
      dir = oldConfig('OutDir')
      dir += '/rFlow/'
      mkdir_p(dir) 
      return dir + os.path.basename(oldConfig.fileName_)

    def writeConfigM(self, oldConfig, singleM):
      '''Change the mode m in the config file and change the filenames of the outputs. Returns the new config.'''
      configR = myConfigParser(oldConfig.fileName_)
      oldM = configR('Modes')[0][0]
      configR.set('Modes', 'SINGLE %i' % singleM)
      for outputType in ['FileOutputCircle','FileOutputDisk','FileOutputPlane','FileOutputSphere']:
        if oldConfig.has_option(oldConfig.sections()[0], outputType):
          outputCrt = oldConfig(outputType).split()
          if '_m' in outputCrt[0]: # replace the old m by the new one
            outputCrt[0] = outputCrt[0].replace('_m%s' % oldM, '_m%s' % singleM)
          else:
            outputCrt[0] += '_m%i' % singleM
          configR.set(outputType, ' '.join(outputCrt))
      configFileR = self.getNewConfigNameM(oldConfig, singleM)
      newOutDir = configR('OutDir') + '/m%s/' % singleM
      configR.set('OutDir', newOutDir)
      configR.save(configFileR)
      return configFileR

    def getNewConfigNameM(self,oldConfig,singleM):
      dir = oldConfig('OutDir')
      dir += '/m%s/' % singleM
      mkdir_p(dir) 
      return dir + os.path.basename(oldConfig.fileName_)

    def getGReverseFlowFromG(self,G, reverseM = False,iF = None):
      '''Get G(-u) (or dG) from G(u) when it is possible depending on the type of flow.'''
      if reverseM:
        # G_m(-u) = G_{-m}(u)
        modes = self.params_.getModes()[0]
        # if modes[0] != -modes[-1]:
          # raise Exception('The Green function G(-u) can be guessed from G(u) only if the modes are symmetric (from -Mmax to Mmax)')

        if self.MFFT_:
          # revert the transform in phi to get G_m(u), flip the m 
          # and transform back
          Gm = NP.fft.fft(G,axis=-1)
          Gm = Gm[...,::-1]
          Gr = NP.fft.ifft(Gm,axis=-1)
        else:
          # The last column of the Green's function is the m component, 
          # revert the direction of the modes to obtain G(-u).
          if len(modes) == 1 and modes[0] != 0:
            print(iF)
            print(modes[0])
            # Gr = self.computeOnTheFly(iF,singleM=-modes[0])
            print('reverse m done')
          else:
            Gr = G[...,::-1]

      else:
        # G_m(-u) = G_m(u)
        Gr = G
      return Gr


    def getGReverseMFromG(self,G,Gr,singleM,iF=None):
      '''Get G_-m(u) and G_-m(-u) (or dG) from G_m(u) and G_m(-u) when it is possible depending on the type of flow.'''
      phi  = cartesianToSpherical(getSource(self.params_.config_)[0][0])[-1]
      fac = NP.exp(-2.j*singleM*phi)
      if self.rotFlow_:
        GMm = Gr * fac
        GrMm = G * fac
      elif self.meridFlow_:
        GMm = G * fac
        GrMm = Gr * fac
      else:
        raise Exception('Simulation with the reverse mode is required')
      return GMm,GrMm


    def computeGreen(self,freq=None,ifreq=None,coords=None,icoords=None,grad=False,lFilter = None,reverseFlow=False,singleM=False,\
                      nbCores=1, Memory=8*1024*1024,walltime=12,nNodes=1,H5py=False,localCompute=False):
      if freq is not None:
        if  hasattr(freq , '__len__'):
          nbFreqs = len(freq)
        else:
          nbFreqs = 1
          freq = [freq]
        freq = NP.array(freq)
      elif ifreq is not None:
        if  hasattr(ifreq , '__len__'):
          nbFreqs = len(ifreq)
        else:
          nbFreqs = 1
          ifreq = [ifreq]
        ifreq = NP.array(ifreq)
      else:
        raise Exception('freq or ifreq must be defined to compute the Green function')
      if localCompute:
        # if nbCores == 1:
        #   G = greenParallel(self,freq,ifreq,coords,icoords,grad,lFilter,reverseFlow,singleM)
        # else:
        G = reduce(greenParallel, (self,freq,ifreq,coords,icoords,grad,lFilter,reverseFlow,singleM),nbFreqs,nbCores,None,True)
      else:
        G = reduceOnCluster(greenParallel, (self,freq,ifreq,coords,icoords,grad,lFilter,reverseFlow,singleM), nbFreqs, nbCores, Memory, None,None,"errorReduceOnCluster.log",True,walltime,nNodes,1,H5py)


      if singleM or reverseFlow:
        # Just return G for the moment. Reshape to do later
        return G
      else:
        Gdata = NP.zeros(([len(G[0])] + list(G[0][0].shape)),dtype='complex')
        for i in range(len(G[0])):
          Gdata[i,...] = G[0][i]
        if grad:
          Ggrad = NP.zeros(( [len(G[0])] + [3] + list(G[0][0].shape)),dtype='complex')
          for i in range(len(G[0])):
            Ggrad[i,...] = G[1][i]

      # if localCompute:
      #   Gdata = NP.moveaxis(Gdata,-1,0)
      #   Ggrad = NP.moveaxis(Ggrad,(-1,1),(0,1))

      if grad:
        return Gdata,Ggrad
      else:
        return Gdata

def greenParallel(self,freq=None,ifreq=None,coords=None,icoords=None,grad=False,lFilter = None,reverseFlow=False,singleM=False):
  return self(freq,ifreq,coords,icoords,grad,lFilter,reverseFlow,singleM)


def GetFileName(self,iM=None,iF = None):
   prefix   = getFileNamePrefix(self.params_,self.nSrc_)
   outDir   = self.params_.config_('OutDir')
   Nm       = getModes(self.params_.config_)[2]
   Nf       = getFrequencies(self.params_.config_)[2]
   if self.unidim_:
     Nl       = int(self.params_.config_('MaximumDegree'))+1
   if iM is None and iF is None:
     suffix = '*'
     nFiles = Nm*Nf
   elif iM is None:
     suffix = '_m*_f%d' % iF
     nFiles = Nm
   elif iF is None:
     suffix = '_m%d_f*' % iM
     nFiles = Nf
   else:
     suffix = '_m%d_f%d' % (iM,iF)
     nFiles = 1
   extraString = ''

   if self.unidim_:
    if self.SeperateDegrees_:
      suffix = '_f*_L*'
      nFiles = Nf*Nl
    else:
      if iF is None:
        suffix = '_f*'
      else:
        suffix = '_f%s' % iF
      nFiles = Nf
      extraString = "| sed '/_L[0-9]*\.dat/d' "

   moreThanSuffix = '_U0'

   return ['%s/%s%s%s.dat' % (outDir,prefix,suffix,moreThanSuffix),nFiles,extraString]

class rotatedGreen(object):
    ''' 3D Green's function computed from a polar1D m=0 Green's function using 
        spherical harmonics rotation formula

        Since we're dealing with 3D arrays, the data is NEVER stored in the class instance.
    '''

    def __init__ (self,params3D,G2D,thetaR,phiR,loadGrad=False,Ls=NP.arange(600)):

        ''' Initialization of a 3D Rotated Green's function from a m=0 Green's function.
            Parameters of loading, onTheFly, nSrc, etc, are determined by the original
            Green's function
        '''

        if params3D.typeOfOutput_ != TypeOfOutput.Spherical3D:
          raise Exception("The given parameters are not compatible with a rotated Green's. Please provide a Spherical3D geometry.")
        self.params_       = params3D
        if G2D.params_.typeOfOutput_ != TypeOfOutput.Polar2D:
          raise Exception("The provided Green's function must have a Polar2D output type.")
        if G2D.params_.geom_.Nr() != params3D.geom_.Nr():
          raise Exception("Geometries of 2D and 3D Green's function must have the same number of radiuses.")
        self.G2D_          = G2D
        self.thetaR_       = thetaR
        self.phiR_         = phiR
        self.Ls_           = Ls
        self.observable_   = G2D.observable_ # Will be accessed by Kernel routines   
        self.iLFreq_       = -1
     
    def __call__(self,freq=None,ifreq=None,coords=None,icoords=None,returnOriginal=False):
      return self.get(freq,ifreq,coords,icoords,grad=False,returnOriginal=returnOriginal)
    
    def get(self,freq=None,ifreq=None,coords=None,icoords=None,grad=False,returnOriginal=False):

      ''' Returns the values of the Green's function
          for all or specified frequencies.
          If the values are not stored, load or compute on the fly.
      '''

      # Set frequencies to load
      if freq is not None:
        # Single frequency
        if isinstance(freq,float):
          ifreq = [NP.argmin(abs(freq-self.params_.time_.omega_/(2.e0*pi)))]
        else:
          ifreq = []
          for f in freq:
            ifreq.append(NP.argmin(abs(f-self.params_.time_.omega_/(2.e0*pi))))
      elif ifreq is not None:
        if isinstance(ifreq,int):
          ifreq = [ifreq]

      # Get polar data
      
      if not (hasattr(self.G2D_,"GL_") and self.G2D_.iLFreq_ == ifreq):
        G2D = self.G2D_.get(freq,ifreq,coords,icoords,grad)
      GL  = self.G2D_.GL_
      drGL = None
      if grad:
        drGL = self.G2D_.drGL_

      if GL.ndim==3:
        Nw = GL.shape[-1]
        if Nw == 1:
          GL = GL[...,0]
      else:
        Nw = 1

      # Single frequency
      if Nw == 1:
        print("Rotating data")
        res = rotateGreenData(self,GL,grad,drGL)
      # Many frequencies
      else:
        if self.G2D_.params_.nbProc_ > 1 and Nw >10:
          print("Rotating m0 Green's function in parallel")
          res = reduce(rotateGreenData,(self,GL,grad,drGL),Nw,self.G2D_.params_.nbProc_,progressBar=True)
        else:
          print("Rotating m0 Green's function")
          dims = self.params_.N_
          dims.append(Nw)
          res  = NP.zeros(dims,dtype=G2D.dtype)
          PB = progressBar(Nw,'serial')
          for iw in range(Nw):
            res[...,iw] = rotateGreenData(self,GL[...,iw],grad,drGL)
            PB.update()
          del PB
            
      if returnOriginal:
        return G2D,res
      else:
        return res

    def getGrad(self,freq=None,ifreq=None,coords=None,icoords=None):
      return self.get(freq,ifreq,coords,icoords,grad=True)
    
def rotateGreenData(self,UL,grad=False,drUL=None):
    ''' transforms a U(r,th) array into RU(r,th,phi), 
        stores the L transform in G 
    '''

    # Compute cos(distance to new pole)
    N   = self.params_.geom_.N_
    th  = self.params_.geom_.theta()
    phi = self.params_.geom_.phi()
    CG  = (NP.cos(th)*NP.cos(self.thetaR_))[:,NP.newaxis]\
        + (NP.sin(th)*NP.sin(self.thetaR_))[:,NP.newaxis]\
        * NP.cos(phi-self.phiR_)[NP.newaxis,:]
    Ls  = self.G2D_.computeLModes_

    # Ravel for sum legendre...
    if not grad:
      leg = Legendre(CG.ravel(),Ls,normalized=True)
      res = leg.reconstructFromBasis(UL,axis=-1,x=CG.ravel(),sumDerivatives=True)
      #res = sumLegendre(UL,Ls,CG.ravel(),axisL=-1,normalized=True)
      return res.reshape(N)
    else:

      # R component: compute derivatives of UL coefficients
      r  = self.params_.geom_.r()
      dr = FDM_Compact(r)
      if drUL is None:
        drUL = NP.zeros(UL.shape,dtype=UL.dtype)
        for l in range(len(Ls)):
          drUL[:,l] = dr(UL[:,l])

      # Compute sum of Legendre polynomials and derivatives
      leg = Legendre(CG.ravel(),Ls,normalized=True)
      sumPl,sumDPl = leg.reconstructFromBasis([UL,drUL],axis=-1,x=CG.ravel(),sumDerivatives=True)
      #sumPl,sumDPl = sumLegendre([UL,drUL],Ls,CG.ravel(),axisL=-1,normalized=True,sumDerivatives=True)
      res   = sumPl[0].reshape(N)
      drRes = sumPl[1].reshape(N)
  
      # Theta component
      dCG  = -(NP.sin(th)*NP.cos(self.thetaR_))[:,NP.newaxis]\
           +  (NP.cos(th)*NP.sin(self.thetaR_))[:,NP.newaxis]\
           *  NP.cos(phi-self.phiR_)[NP.newaxis,:]
      dthRes = sumDPl[0].reshape(N)*dCG[NP.newaxis,:,:]

      # Phi component
      dCG  = (NP.cos(th)*NP.cos(self.thetaR_))[:,NP.newaxis]\
           - (NP.sin(th)*NP.sin(self.thetaR_))[:,NP.newaxis]\
           * NP.sin(phi-self.phiR_)[NP.newaxis,:]
      dphiRes = sumDPl[0].reshape(N)*dCG[NP.newaxis,:,:]
      
    if not grad:
      return res
    else:
      return res,NP.array([drRes,dthRes,dphiRes])


def getGreenCluster(GreensClass,ifreq=0):
    return GreensClass.get(ifreq = ifreq)[0]
