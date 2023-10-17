from __future__ import print_function
import itertools as IT
import numpy     as NP
import getpass
import os.path
import sys
import subprocess
import shutil
if sys.version_info >= (3.,0.):
  import _thread
else:
  import thread
import time
import h5py

# from multiprocessing import Pool,Lock,Value
from math            import floor,ceil
from .        import *
from .misc           import *

import copyreg
import types
# import multiprocessing
import multiprocess as multiprocessing
from multiprocess import Pool,Lock,Value
from scipy.sparse import hstack,vstack


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copyreg.pickle(types.MethodType, _pickle_method)

# self explanatory
class progressBar(object):

  def __init__(self,N,stype='serial',bonus=None,w=50):

    if stype == 'parallel':
      self.stype    = 'parallel'
      self.progress = Value('i',0)
    else:
      self.stype    = 'serial'
      self.progress = 0.e0
    self.end      = N
    self.lock     = Lock()

    self.w  = w
    str      = '  0% |'+(w*' ')+'|'
    if bonus is not None:
      self.bonus = bonus
      str = str + ' ' + bonus
    else:
      self.bonus = ''
    self.pbl = len(str)
    print( str,end='\r')

  def update(self,w=50):

    if self.stype=='parallel':
      self.lock.acquire()
      self.progress.value += 1
      val = self.progress.value
    else:
      self.progress += 1
      val = self.progress
    percent = int(floor(val/(1.0*self.end)*100))
    prctstr = '%3d%%' % percent
    nequal  = max(percent*self.w//100-1,0) 
 
    print('\r'+prctstr + ' |' + (nequal* '=') + '|' + ((self.w-nequal-1)*' ') + '|' + self.bonus,end='\r' )
    sys.stdout.flush()

    if self.stype=='parallel':
      self.lock.release()
 
  def __del__(self):

    #print (self.pbl*'\r',)
    #print (3*self.pbl*' ',)
    #print (3*self.pbl*'\r',)
    print()
    sys.stdout.flush()

###############################################################################
# Routines to unroll the list of arguments (pool cannot pass several arguments)

def applyPar(args):
  return apppply(*args)

def runningPar(args):
  return running(*args)

def runningSumPar(args):
  return runningSum(*args)

def runningMinPar(args):
  return runningMin(*args)

def runningMaxPar(args):
  return runningMax(*args)

def runningAvePar(args):
  return runningAve(*args)

###############################################################################
# Process or reduction routines

def apppply(funcname,args):

  if args[0][0]:
    PG = progressBar(len(args),'serial',bonus=' (parallel)')

  for i in range(len(args)):
    funcname(*(args[i][1:]))
    if args[i][0]:
      PG.update()          
 
  if args[i][0]:
    del PG

def running(funcname,args):
  res = []
  if args[0][0]:
    PB = progressBar(len(args),'serial',bonus=' (parallel)')
  for i in range(len(args)):
    res.append(funcname(*(args[i][1:])))
    if args[i][0]:
      PB.update()          
  if args[i][0]:
    del PB
  return res

def runningSum(funcname,args):
  ''' compute the sum from start to end, applying func_name to whatever args'''
  res = 0
  if args[0][0]:
    PG = progressBar(len(args),'serial',bonus=' (parallel)')
  for i in range(len(args)):
    res += funcname(*(args[i][1:]))
    if args[i][0]:
      PG.update()          
  if args[i][0]:
    del PG
  return res

def runningAve(funcname,args):
  ''' compute the sum from start to end, applying func_name to whatever args'''
  res = 0
  if args[0][0]:
    PG = progressBar(len(args),'serial',bonus=' (parallel)')
  for i in range(len(args)):
    res += args[i][1]*funcname(*(args[i][2:]))
    if args[i][0]:
      PG.update()          
  if args[i][0]:
    del PG
  return res

def runningMin(funcname,args):
  res = funcname(*(args[0][1:]))
  if args[0][0]:
    PG = progressBar(len(args),'serial',bonus=' (parallel)')
    PG.update()
  for i in range(1,len(args)):
    res = NP.minimum(res,funcname(*(args[i][1:])))
    if args[i][0]:
      PG.update()          
  if args[i][0]:
    del PG
  return res

def runningMax(funcname,args):
  res = funcname(*(args[0][1:]))
  if args[0][0]:
    PG = progressBar(len(args),'serial',bonus=' (parallel)')
    PG.update()
  for i in range(1,len(args)):
    res = NP.maximum(res,funcname(*(args[i][1:])))
    if args[i][0]:
      PG.update()          
  if args[i][0]:
    del PG
  return res

########################################################################
# Reduction preparation routines

def buildArgTable(args,N,nbproc,progressBar,weights=None):

  starts = partition(N,nbproc)
  # prepare list of arguments
  arg_table = []
  if weights is not None:
    w = NP.genfromtxt(weights)

  for i in range(nbproc):
    arg_table_proc = []    
    for j in range(starts[i],starts[i+1]):

      list_args = []
      # Precise in arguments if progress bar is needed
      if progressBar and i==0:
        list_args.append(True)
      else:
        list_args.append(False)

      # Precise the weights for a weighted average
      if weights is not None:
        list_args.append(w[j])
      for arg in list(args):
        # Detect if the argument is a numpy array with last dimension N
        try:
          if (arg.shape[-1]==N):
            if arg.ndim == 1:
              list_args.append(arg[j])
            else:
              list_args.append(arg[...,j])
          else:
            list_args.append(arg)
        except: 
          list_args.append(arg)
      arg_table_proc.append(tuple(list_args))

    arg_table.append(arg_table_proc)
  return arg_table

def partition(N,Nprocs):

  division = N / float(Nprocs)
  starts   = []
  for i in range(Nprocs):
    starts.append(int(round(division * i)))
  starts.append(min(int(round(division*(i+1))),N))
  return starts

########################################################################
# Top reduction routines

def reduce(funcname,args,N,nbproc,type=None,progressBar=False,SparseStack=None):
  ''' Applies the type of reduction on a pool map of nbprocs
      N : size of array on which we perform computation using the function func_name
      type can be 'SUM' 'MAX' 'MIN' 'AVE'
  '''

  if (type is not None and type[:3] == 'AVE'):
    weights = type[3:]
  else:
    weights = None
  argTable = buildArgTable(args,N,nbproc,progressBar,weights)



  pool = Pool(nbproc)
  print(pool)

  if (type==None):
    resPerProc = pool.map(runningPar,zip(IT.repeat(funcname),argTable))
  elif (type=='SUM'):
    resPerProc = pool.map(runningSumPar,zip(IT.repeat(funcname),argTable))
  elif (type=='MIN'):
    resPerProc = pool.map(runningMinPar,zip(IT.repeat(funcname),argTable))
  elif (type=='MAX'):
    resPerProc = pool.map(runningMaxPar,zip(IT.repeat(funcname),argTable))
  elif (type[:3]=='AVE'):
    resPerProc = pool.map(runningAvePar,zip(IT.repeat(funcname),argTable))

  # Final reduction
  if (type==None):
    if SparseStack is None:
      res2 = []
      for i in range(nbproc):
        for j in range(len(resPerProc[i])):
          res2.append(NP.array(resPerProc[i][j]))
      D   = NP.array(res2)
      res = NP.rollaxis(D,0,D.ndim)
    else:
      res = []
      for i in range(nbproc):
        for j in range(len(resPerProc[i])):
          res.append(resPerProc[i][j])
      if SparseStack.upper() == 'VSTACK':
        return vstack(res)
      elif SparseStack.upper() == 'HSTACK':
        return hstack(res)        

  elif (type=='SUM' or type[:3] == 'AVE'):
    res = NP.sum(resPerProc,axis=0)
  elif (type=='MIN'):
    res = NP.min(resPerProc,axis=0)
  elif (type=='MAX'):
    res = NP.max(resPerProc,axis=0)
  elif (type=='NO'):
    # nothing to return
    print ('ending parallel computation')

  pool.close()
  return NP.asarray(res)

def apply(funcname,args,N,nbproc,progressBar=False):
  '''same as reduce, without return '''

  argTable = buildArgTable(args,N,nbproc,progressBar)

  pool = Pool(nbproc)
  pool.apply_async(applyPar,zip(IT.repeat(funcname),argTable))
  pool.close()

########################################################################
# Reductions on clusters

def reduceOnCluster(func,args,nTask,nProc,imageSize,rType=None,\
                    additionalCommands=None,\
                    errorLogFileName="errorReduceOnCluster.log",\
                    deleteFiles=True,walltime=72,nNodes = 1,parallelRegroup=1,saveH5py=False):
  
  cluster = getClusterName()
  # Condor
  if cluster.upper() == 'CONDOR':
    return reduceOnSeismoCluster(func,args,nTask,nProc,imageSize,rType,\
                                 additionalCommands,errorLogFileName,deleteFiles,parallelRegroup,saveH5py)
  # Torque:
  elif cluster.upper() == 'TORQUE':
    return reduceOnHelioCluster(func,args,nTask,nProc,imageSize,rType,\
                                additionalCommands,errorLogFileName,\
                                deleteFiles,walltime,parallelRegroup,saveH5py)
  #Slurm
  elif cluster.upper() == 'SLURM':
    return reduceOnDalmaCluster(func,args,nTask,nProc,imageSize,rType,\
                                additionalCommands,errorLogFileName,\
                                deleteFiles,walltime,nNodes,parallelRegroup,saveH5py)

def reRunReduceOnCluster(errorLogFileName,func,args,nTask,nProc,imageSize,\
                         rType=None,additionalCommands=None,\
                         deleteFiles=True,walltime=72,nNodes=1,parallelRegroup=1,saveH5py = False):
  ''' Completes missing parts of reduceOnCluster. 
      Some arguments are not used, but are there so you can
      just copy paste the reduceOnCluster lines, and add 'reRun' and the
      errorLogFileName containing the ID of the reduction data
  '''

  cluster = getClusterName()
  # Condor
  if cluster.upper() == 'CONDOR':
    return reRunReduceOnSeismoCluster(errorLogFileName,func,args,nTask,nProc,\
                                      imageSize,rType,additionalCommands,deleteFiles,parallelRegroup,saveH5py)
  # Torque:
  elif cluster.upper() == 'TORQUE':
    return reRunReduceOnHelioCluster(errorLogFileName,func,args,nTask,nProc,imageSize,\
                                     rType,additionalCommands,deleteFiles,walltime,parallelRegroup,saveH5py)
  # Slurm
  elif cluster.upper() == 'SLURM':
    print(nNodes)
    return reRunReduceOnDalmaCluster(errorLogFileName,func,args,nTask,nProc,imageSize,\
                                     rType,additionalCommands,deleteFiles,walltime,nNodes,parallelRegroup,saveH5py)

########################################################################

def loadPartialReduction(outDir,iStart,iEnd,saveH5py):
  if saveH5py:
    h5f = h5py.File('%s/res_%d_%d.h5'%(outDir,iStart,iEnd))
    res = h5f['res'].value
    h5f.close()
  else:
    res =  NP.load('%s/res_%d_%d.npy'%(outDir,iStart,iEnd),allow_pickle=True)
  return res

def reducePartialReduction(ID,nTask,nProc,rType,deleteFiles=True,parallelRegroup=1,cluster = 'condor',saveH5py=False):
  ''' regroups partial results files '''
  
  if cluster.upper() == 'SLURM':
    rDir   = '/scratch/' + getpass.getuser() + '/reduceData/%d' %ID
  else:
    rDir   = '/scratch/seismo/' + getpass.getuser() + '/reduceData/%d' %ID

  starts = partition(nTask,nProc)

  print ("\n Regrouping results")

  if parallelRegroup <= 1:
 
    PB = progressBar(nProc,'serial')
    if saveH5py:
      h5f = h5py.File('%s/res_%d_%d.h5'%(rDir,starts[0],starts[1]),'r')
      res = NP.array(h5f['res'])
      h5f.close()
    else:
      res    = NP.load('%s/res_%d_%d.npy'%(rDir,starts[0],starts[1]),allow_pickle=True)
    PB.update()  

    for i in range(1,nProc):
      if saveH5py:
        h5f = h5py.File('%s/res_%d_%d.h5'%(rDir,starts[i],starts[i+1]),'r')
        resTmp = NP.array(h5f['res'])
        h5f.close()
      else:
        resTmp = NP.load('%s/res_%d_%d.npy'%(rDir,starts[i],starts[i+1]),allow_pickle=True)
      if rType is None:
        res = NP.concatenate((res,resTmp))
      elif rType == 'SUM' or rType[:3] == 'AVE':
        res += resTmp
      elif rType == 'MIN':
        res = NP.minimum(res,resTmp)
      elif rType == 'MAX':
        res = NP.maximum(res,resTmp)
      PB.update()

    del PB

  else:
    ## Special reduce
    #res = reduce(loadPartialReduction,(rDir,NP.array(starts[:nProc]),NP.array(starts[1:])),nProc,int(floor(parallelRegroup)),rType,progressBar=True)
    # Protection against greedy or careless people
    hn = getHostname()
    nRegroup = int(floor(parallelRegroup))
    # Condor
    if 'seismo1' in hn:
      Nmax = 24
    elif 'helio' in hn and ('helios' not in hn):
      Nmax = 28
    else:
      Nmax = 28
    if nRegroup > Nmax//2:
      print((bColors.warning() + ' You asked for too many processors on an interactive machine.'))
      print(('The number of processors will be brought back to half of the machine (%i)' % (int(Nmax)//2)))
      nRegroup = Nmax//2

    argTable = buildArgTable((rDir,NP.array(starts[:nProc]),NP.array(starts[1:]),saveH5py),nProc,nRegroup,progressBar)

    pool = Pool(nRegroup)
    if (rType==None):
      resPerProc = pool.map(runningPar,zip(IT.repeat(loadPartialReduction),argTable))
    elif (rType=='SUM' or rType[:3] == 'AVE'):
      resPerProc = pool.map(runningSumPar,zip(IT.repeat(loadPartialReduction),argTable))
    elif (rType=='MIN'):
      resPerProc = pool.map(runningMinPar,zip(IT.repeat(loadPartialReduction),argTable))
    elif (rType=='MAX'):
      resPerProc = pool.map(runningMaxPar,zip(IT.repeat(loadPartialReduction),argTable))

    # Final reduction
    if (rType is None):
      res2 = []
      for i in range(nRegroup):
        for j in range(len(resPerProc[i])):
          for k in range(len(resPerProc[i][j])):
            res2.append(NP.array(resPerProc[i][j][k]))
      res = NP.array(res2)
    elif (rType=='SUM' or rType[:3] == 'AVE'):
      res = NP.sum(resPerProc,axis=0)
    elif (rType=='MIN'):
      res = NP.min(resPerProc,axis=0)
    elif (rType=='MAX'):
      res = NP.max(resPerProc,axis=0)

    pool.close()
    # End parallel regroup

  res = NP.array(res)

  if deleteFiles:
    shutil.rmtree(rDir,ignore_errors=True)

  if rType is None:
    return NP.rollaxis(res,0,res.ndim)
  else:
    return res


########################################################################
# Condor specific

def reduceOnSeismoCluster(func,args,nTask,nProc,imageSize,rType=None,\
                          additionalCommands=None,\
                          errorLogFileName="errorReduceOnCluster.log",
                          deleteFiles=True,parallelRegroup=1,saveH5py=False):

  ID   = int(NP.random.rand(1)*1.e32) 
  rDir = '/scratch/seismo/' + getpass.getuser() + '/reduceData/%d' %ID
  if not deleteFiles:
    with open('reduceID','w') as elf:
      elf.write('%d\n' % ID)
  if deleteFiles and os.path.exists(rDir):
    shutil.rmtree(rDir,ignore_errors = True)
  mkdir_p(rDir)

  # Save function
  argsFile = rDir + '/arguments.pkl'
  funcFile = rDir + '/function.pkl'

  saveObject(args,argsFile)
  saveObject(func,funcFile)

  # Write condor submission file
  subFileName  = rDir + '/toSubmit.sub'  
  execFileName = rDir + '/exeLoadAnaconda.sh'

  writeExecutableReduceOnSeismoCluster(execFileName)

  with open(subFileName,'w') as CSF:
    CSF.write('Universe   = Vanilla\n')
    CSF.write('Executable = %s\n' % execFileName)
    CSF.write('Arguments  = %s %s $(Process) %d %d %s %s %i\n'\
           % (funcFile,argsFile,nProc,nTask,rType,rDir,saveH5py))
    CSF.write('image_size = %d\n' % imageSize)
    CSF.write('get_env    = True\n')
    mkdir_p(rDir+'/condorLogs')
    CSF.write('output     = %s/condorLogs/cond$(Process).out\n' % rDir)
    CSF.write('error      = %s/condorLogs/cond$(Process).err\n' % rDir)
    CSF.write('log        = %s/condorLogs/cond$(Process).log\n' % rDir)
    CSF.write('queue %d\n' % nProc)
    if additionalCommands:
      CSF.write('%s \n' % additionalCommands)

  # Submit
  subP    = subprocess.Popen('condor_submit %s' % subFileName,shell=True,\
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  out,err = subP.communicate()

  if err:
    print ("Job couldn't be submitted (probably disconnected scheduler).")
    return

  # Wait for all jobs to finish
  jobID    = out.split()[-1].split('.')[0]
  print(("Submitted job #%s to the cluster. Now waiting for the jobs to complete." % jobID))
  nok,nnotok = waitCondorJobs(ID,nProc,jobID)

  # Check missing files
  if nnotok != 0:
    print(("\nWARNING: %d jobs abnormally terminated! Result won't be computed\n" % nnotok))
    # Generate a log file which can be used with reRunReduceOnCluster
    with open(errorLogFileName,'w') as elf:
      elf.write('%d\n'%ID)
      print(("Reduction ID was stored in %s." % errorLogFileName))
    return
  else:
    return reducePartialReduction(ID,nTask,nProc,rType,deleteFiles,parallelRegroup,saveH5py=saveH5py)

def writeExecutableReduceOnSeismoCluster(execFileName):
  ''' Writes executable to load modules and call the generic python script '''

  with open(execFileName,'w') as ELA:
    ELA.write('#!/bin/bash\n')
    ELA.write('. $MODULESHOME/init/bash\n')
    ELA.write('module load anaconda/2.2.0\n')
    ELA.write('cd %s/pyCompHelio/Common\n' % pathToMPS())
    ELA.write('./reduceOnCluster.py $1 $2 $3 $4 $5 $6 $7 $8\n')
  os.system('chmod +x %s'%execFileName)

def waitCondorJobs(ID,nProc,jobID,strJobList=None):
  ''' Checks log files of jobs to extract the word "termination".
      When all jobs are done (good or bad), the script can go on.
  '''

  rDir     = '/scratch/seismo/' + getpass.getuser() + '/reduceData/%d' %ID
  finished = False
  pings    = 0
  if strJobList is None:
    nJobs      = nProc
    strJobList = "0"
    for i in range(1,nProc):
      strJobList += " %d"%i
    folder = "condorLogs"
  else:
    nJobs  = len(strJobList.split())
    folder = "condorLogsReRun"
    jobID  = " ".join(jobID)

  while not finished:

    if pings == 0:
      print('zzZZzz (pings every 10 sec)')
    if pings == 12:
      print('\n Entering deep slumber (pings every 1 min)')
    if pings == 69:
      print('\n Comatose (pings every 10 min)')

    if pings < 12:
      time.sleep(10)
      sys.stdout.write('.')
    elif pings < 69:
      time.sleep(60)
      sys.stdout.write('.')
    else:
      time.sleep(600)
      sys.stdout.write('.')
    sys.stdout.flush()
    # subP = subprocess.Popen('for i in %s; do cat %s/%s/cond$i.log | egrep \'termination\'; done | wc -l'\
                            # % (strJobList,rDir,folder),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if nJobs == 0:
      out = '1'
    else:
      subP = subprocess.Popen('condor_history %s | wc -l' % jobID,\
                              shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
      out,err  = subP.communicate()

    try:
      nfnshd   = int(out.split()[0]) -1# subtract one for the column names
      finished = (nfnshd == nJobs)
    except:
      pass
    pings   += 1
  
  # Safety period to ensure all files are written
  time.sleep(20)

  # Count succesfull runs
  subP = subprocess.Popen('for i in %s; do cat %s/%s/cond$i.log | egrep \'termination\' | awk \'{print $6}\' | cut -f 1 -d \')\' ; done'\
                          % (strJobList,rDir,folder),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  # subP = subprocess.Popen('condor_history %s | grep "C " | wc -l' % jobID,\
                           # shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   
  out,err  = subP.communicate()
  statuses = NP.array(out.split()).astype(int)
  nok      = sum(statuses==0)
  nnotok   = nJobs-nok

  return nok,nnotok

def reRunReduceOnSeismoCluster(errorLogFileName,func,args,nTask,nProc,imageSize,\
                               rType=None,additionalCommands=None,\
                               deleteFiles=True,parallelRegroup=1,saveH5py=False):
  ''' From an ID, checks missing files and rerun the corresponding jobs '''

  try:
    elf = open(errorLogFileName)
    ID  = int(elf.read())
  except:
    raise Exception('Unable to read log file %s' % errorLogFileName)

  rDir         = '/scratch/seismo/' + getpass.getuser() + '/reduceData/%d' %ID
  argsFile     = rDir + '/arguments.pkl'
  funcFile     = rDir + '/function.pkl'
  execFileName = rDir + '/exeLoadAnaconda.sh'
  saveObject(args,argsFile)
  saveObject(func,funcFile)
  
  # Detect missing results
  iFiles = []
  starts = partition(nTask,nProc) 
  for i in range(nProc):
    if not os.path.exists('%s/res_%d_%d.npy'%(rDir,starts[i],starts[i+1])):
      iFiles.append(i)

  print(("iFiles", iFiles))

  # Submit on condor for missing files
  jobIDs = []
  for iFile in iFiles:
    subFileName  = rDir + '/toSubmit%d.sub' % iFile 
    with open(subFileName,'w') as CSF:
      CSF.write('Universe   = Vanilla\n')
      CSF.write('Executable = %s\n' % execFileName)
      CSF.write('Arguments  = %s %s %d %d %d %s %s %i\n'\
             % (funcFile,argsFile,iFile,nProc,nTask,rType,rDir,saveH5py))
      CSF.write('image_size = %d\n' % imageSize)
      CSF.write('get_env    = True\n')
      shutil.rmtree(rDir+'/condorLogsReRun',ignore_errors=True)
      mkdir_p(rDir+'/condorLogsReRun')
      CSF.write('output     = %s/condorLogsReRun/cond%d.out\n' % (rDir,iFile))
      CSF.write('error      = %s/condorLogsReRun/cond%d.err\n' % (rDir,iFile))
      CSF.write('log        = %s/condorLogsReRun/cond%d.log\n' % (rDir,iFile))
      CSF.write('queue\n')
      if additionalCommands:
        CSF.write('%s \n' % additionalCommands)

    # Submit
    subP    = subprocess.Popen('condor_submit %s' % subFileName,shell=True,\
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    out,err = subP.communicate()
    print((out,err))
    jobIDs.append(out.split()[-1].split('.')[0])


  # Wait for jobs to finish
  print(("Submitted job ",jobIDs,"to the cluster. Now waiting for the jobs to complete."))
  strJobList = ""
  for iFile in iFiles:
    strJobList += "%d " % iFile
  nok,nnotok = waitCondorJobs(ID,nProc,jobIDs,strJobList)

  # Check missing files
  if nnotok != 0:
    print(("\nWARNING: %d jobs abnormally terminated! Result won't be computed\n" % nnotok))
    return
  else:
    return reducePartialReduction(ID,nTask,nProc,rType,deleteFiles,parallelRegroup,saveH5py=saveH5py)

########################################################################
# Slurm (Abu Dhabi) specific

def reduceOnDalmaCluster(func,args,nTask,nProc,imageSize,rType=None,\
                         additionalCommands=None,\
                         errorLogFileName="errorReduceOnCluster.log",\
                         deleteFiles=True,walltime=72,nNodes=1,parallelRegroup=1,saveH5py=False):

  ID   = int(NP.random.rand(1)*1.e32) 
  rDir = '/scratch/' + getpass.getuser() + '/reduceData/%d' %ID
  if not deleteFiles:
    with open('reduceID','w') as elf:
      elf.write('%d\n' % ID)
  if deleteFiles and os.path.exists(rDir):
    shutil.rmtree(rDir,ignore_errors = True)
  mkdir_p(rDir)
  mkdir_p(rDir+'/jobLogs')

  # Save function
  argsFile = rDir + '/arguments.pkl'
  funcFile = rDir + '/function.pkl'

  saveObject(args,argsFile)
  saveObject(func,funcFile)

  # jobIDs = []
  # for iJob in range(nProc):
  #   subFileName  = rDir + '/toSubmit%s.sub' % iJob
  #   with open(subFileName,'w') as QSF:
  #     QSF.write('#!/bin/bash\n')
  #     QSF.write('#SBATCH -p serial\n')
  #     QSF.write('#SBATCH --ntasks=1\n')
  #     QSF.write('#SBATCH --mem=%d\n' % (imageSize/1024))
  #     QSF.write('#SBATCH --time=%d:00:00\n' % walltime)
  #     QSF.write('#SBATCH -o %s/jobLogs/job_%%J.out\n' % (rDir))
  #     QSF.write('#SBATCH -e %s/jobLogs/job_%%J.err\n' % (rDir))
  #     QSF.write('#SBATCH -C avx2 \n')
  #     #QSF.write('. $MODULESHOME/init/bash\n')
  #     # module load command outputs everything in stderr (><)
  #     QSF.write('compnam=gcc \ncompver=4.9.3 \ncomp=${compnam}/${compver} \narch=avx2 \n')
  #     QSF.write('module purge 2>/dev/null\n')
  #     QSF.write('module load all $comp metis/${arch}/5.1.0 mumps/${arch}/5.0.1 lapack arpack/96 gsl/${arch}/2.1 fftw3/${arch}/3.3.4 superlu/${arch}/5.0 blas/${arch}/3.6.0 cblas/${arch}/3.6.0 python/2.7.11 anaconda/2-4.1.1 2>/dev/null\n')
  #     QSF.write('cd %s/pyCompHelio/Common\n' % pathToMPS())
  #     QSF.write('source /home/%s/.bashrc\n' % getpass.getuser())
  #     QSF.write('python -W ignore reduceOnCluster.py %s %s %d %d %d %s %s %i\n'%\
  #               (funcFile,argsFile,iJob,nProc,nTask,rType,rDir,saveH5py))
  #     if additionalCommands is not None:
  #       QSF.write('%s \n' % additionalCommands)

  #   # Submit
  #   subP    = subprocess.Popen('sbatch %s' % subFileName,shell=True,\
  #                              stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  #   out,err = subP.communicate()

  #   if err:
  #     print ("Job couldn't be submitted.")
  #     print()
  #     print (err)
  #     return
  #   # Get jobId
  #   jobIDs.append(out.split()[-1])
  # print([nTask,nNodes])

  nCpuPerTask       = NP.ceil(28/(126*1024*1024/imageSize)).astype(int)
  nSlurmTasks       = min(nTask,NP.floor(28.*nNodes/nCpuPerTask).astype(int))
  nNodes_act        = min(NP.ceil(nSlurmTasks*nCpuPerTask/28).astype(int),nNodes)
  # print(['nNodes',nNodes_act])

  # nJobsPerSlurmTask = nTask / nNodes / 28.

  # print(nCpuPerTask)
  # print(nSlurmTasks)
  # abort


  jobIDs = []
  subFileName  = '%s/CompHelioWork/Slurm_Submissions/toSubmit_%i.sub' % (pathToMPS(),ID)
  with open(subFileName,'w') as QSF:
    for iJob in range(nSlurmTasks):
      QSF.write('cd %s/CompHelioWork/Slurm_Submissions/; python -W ignore %s/pyCompHelio/Common/reduceOnCluster.py %s %s %d %d %d %s %s %i\n'%\
                (pathToMPS(),pathToMPS(),funcFile,argsFile,iJob,nSlurmTasks,nTask,rType,rDir,saveH5py))
    if additionalCommands is not None:
      QSF.write('%s \n' % additionalCommands)

  # Submit
  # abort
  
  subP    = subprocess.Popen('slurm_parallel_ja_submit_MJ.sh -t %02d:00:00 -N %i -c %i -L %s %s' % \
                               (walltime,nNodes_act,nCpuPerTask,rDir + '/jobLogs/',subFileName),shell=True,\
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  print('slurm_parallel_ja_submit_MJ.sh -t %02d:00:00 -N %i -c %i -L %s %s' % \
                               (walltime,nNodes,nCpuPerTask,rDir + '/jobLogs/',subFileName))
  out,err = subP.communicate()
  # out = out.decode('utf-8')
  # err = err


  if len(err.decode('utf-8').split('\n'))>3:
    print ("Job couldn't be submitted.")
    print()
    print (err.decode('utf-8'))
    return
  # Get jobId
  for nNode in range(1,nNodes_act+1):
    jobIDs.append(bytes(out.split()[-1].decode('utf-8') + '_%i' % (nNode),'utf-8'))
  # for nNode in range(nNodes):
    # jobIDs.append(out.split()[-1] + '_%i' (nNode+1))

  # Wait for all jobs to finish
  print (text_special("Submitted jobs to the cluster.\nSlurm jobID: %s \nNow waiting for completion." % out.split()[-1].decode('utf-8'),'g'))
  nok,nnotok = waitSlurmJobs(ID,jobIDs)

  # Check missing files
  if nnotok != 0:
    print(text_special(("\nWARNING: %d jobs abnormally terminated! Result won't be computed\n" % nnotok),'r'))
    # Generate a log file which can be used with reRunReduceOnCluster
    with open(errorLogFileName,'w') as elf:
      elf.write('%d\n'%ID)
      print(text_special("Reduction ID was stored in %s." % errorLogFileName,'y'))
    return
  else:
    return reducePartialReduction(ID,nTask,nSlurmTasks,rType,deleteFiles,parallelRegroup,cluster = 'SLURM',saveH5py=saveH5py)

def waitSlurmJobs(ID,jobIDs):

  rDir     = '/scratch/' + getpass.getuser() + '/reduceData/%d' %ID
  finished = False
  pings    = 0
  nJobs    = len(jobIDs)

  # Build output files list that should be present after jobs completion
  strJobList = ""
  strErrList = ""
  for jobID in jobIDs:
    strJobList = strJobList + " " + jobID.decode("utf-8")
    strErrList = strErrList + " " + rDir + "/jobLogs/job_%s.err" % jobID.decode("utf-8")
  # print(strJobList)

  while not finished:
    if pings == 0:
      print('zzZZzz (10s ping for 2 min)')
    if pings == 12:
      print('\n Entering deep slumber (1min ping for 1 hour)')
    if pings == 72:
      print('\n Comatose (10min ping)')

    if pings < 12:
      time.sleep(10)
      # print('.',end='\r')
      print('.', end='', flush=True)
    elif pings < 72:
      time.sleep(60)
      # print('.',end='\r')
      print('.', end='', flush=True)
    else:
      time.sleep(600)
      # print('.',end='\r')
      print('.', end='', flush=True)

    subP = subprocess.Popen('for i in %s; do sacct -j $i | egrep \'sbatch\' | egrep \'COMPLETE\'; done | wc -l'\
                            % strJobList,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    out,err  = subP.communicate()
    # print(out)

    try:
      nfnshd   = int(out.split()[0])
      finished = (nfnshd == nJobs)
    except:
      pass
    pings   += 1
  print()
  # Safety wait
  time.sleep(10)

  # Count succesfull runs
  # print(("for i in %s; do wc -l $i | awk '{print $1}'; done" % strJobList))
  # subP = subprocess.Popen("for i in %s; do wc -l $i | awk '{print $1}'; done" % strErrList,\
                          # shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  subP = subprocess.Popen("for i in %s; do cat $i | grep -v '===' | wc -l; done" % strErrList,\
                          shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)

  # abort
   
  out,err  = subP.communicate()
  out      = out.decode('utf-8');err = err.decode('utf-8')
  statuses = NP.array(out.split('\n')[:-1]).astype(int)
  nok      = sum(statuses==0)
  nnotok   = nJobs-nok

  return nok,nnotok

def reRunReduceOnDalmaCluster(errorLogFileName,func,args,nTask,nProc,imageSize,\
                              rType=None,additionalCommands=None,\
                              deleteFiles=True,walltime=72,nNodes=1,parallelRegroup=1,saveH5py=False):

  ''' From an ID, checks missing files and rerun the corresponding jobs '''

  try:
    elf = open(errorLogFileName)
    ID  = int(elf.read())
  except:
    raise Exception('Unable to read log file %s' % errorLogFileName)

  rDir         = '/scratch/' + getpass.getuser() + '/reduceData/%d' %ID
  argsFile     = rDir + '/arguments.pkl'
  funcFile     = rDir + '/function.pkl'
  saveObject(args,argsFile)
  saveObject(func,funcFile)

  nCpuPerTask       = NP.ceil(28/(126*1024*1024/imageSize)).astype(int)
  nSlurmTasks       = min(nTask,NP.floor(28.*nNodes/nCpuPerTask).astype(int))
  print([nCpuPerTask,nSlurmTasks,nNodes])
  
  # Detect missing results
  iFiles = []
  starts = partition(nTask,nSlurmTasks) 
  for i in range(nSlurmTasks):
    if not os.path.exists('%s/res_%d_%d.npy'%(rDir,starts[i],starts[i+1])):
      iFiles.append(i)

  print(("iFiles", iFiles))

  # Submit on condor for missing files
  jobIDs = []
  # for iFile in iFiles:
  subFileName  = '%s/CompHelioWork/Slurm_Submissions/toSubmit_%i.sub' % (pathToMPS(),ID)
  reSubFileName = subFileName[:-4] + '_reSub.sub'
  resubCmds    = NP.genfromtxt(subFileName,dtype=str)
  NP.savetxt(reSubFileName,resubCmds[iFiles],fmt = '%s')
  
  # Submit
  subP    = subprocess.Popen('slurm_parallel_ja_submit_MJ.sh -t %02d:00:00 -N %i -c %i -L %s %s' % \
                               (walltime,NP.ceil(len(iFiles)/(nSlurmTasks/nNodes)).astype(int),nCpuPerTask,rDir + '/jobLogs/',reSubFileName),shell=True,\
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  out,err = subP.communicate()
  # out = out.decode('utf-8')
  # err = err


  if len(err.decode('utf-8').split('\n'))>3:
    print ("Job couldn't be submitted.")
    print()
    print (err.decode('utf-8'))
    return
  for nNode in range(1,NP.ceil(len(iFiles)/(nSlurmTasks/nNodes)).astype(int)+1):
    jobIDs.append(bytes(out.split()[-1].decode('utf-8') + '_%i' % (nNode),'utf-8'))
  # print(jobIDs)

  # Wait for jobs to finish
  print ("Submitted jobs to the cluster. Now waiting for completion.")
  nok,nnotok = waitSlurmJobs(ID,jobIDs)

  # Check missing files
  if nnotok != 0:
    print(("\nWARNING: %d jobs abnormally terminated! Result won't be computed\n" % nnotok))
    return
  else:
    return reducePartialReduction(ID,nTask,nSlurmTasks,rType,deleteFiles,parallelRegroup,cluster = 'SLURM',saveH5py=saveH5py)

########################################################################

# Torque specific

def reduceOnHelioCluster(func,args,nTask,nProc,imageSize,rType=None,\
                         additionalCommands=None,\
                         errorLogFileName="errorReduceOnCluster.log",\
                         deleteFiles=True,walltime=72,parallelRegroup=1,saveH5py=False):

  ID   = int(NP.random.rand(1)*1.e32) 
  rDir = '/scratch/seismo/' + getpass.getuser() + '/reduceData/%d' %ID
  if not deleteFiles:
    with open('reduceID','w') as elf:
      elf.write('%d\n' % ID)
  if deleteFiles and os.path.exists(rDir):
    shutil.rmtree(rDir,ignore_errors = True)
  mkdir_p(rDir)
  mkdir_p(rDir+'/jobLogs')

  # Save function
  argsFile = rDir + '/arguments.pkl'
  funcFile = rDir + '/function.pkl'

  saveObject(args,argsFile)
  saveObject(func,funcFile)

  jobIDs = []
  for iJob in range(nProc):
    subFileName  = rDir + '/toSubmit%s.sub' % iJob
    with open(subFileName,'w') as QSF:
      QSF.write('#!/bin/bash\n')
      QSF.write('#PBS -N %s%d\n'% (func.__name__,iJob))
      QSF.write('#PBS -l nodes=1:ppn=1,walltime=%d:00:00\n' % walltime)
      QSF.write('#PBS -q helioq\n')
      QSF.write('#PBS -o %s/jobLogs/job$PBS_JOBID.out\n' % rDir)
      QSF.write('#PBS -e %s/jobLogs/job$PBS_JOBID.err\n' % rDir)
      QSF.write('. $MODULESHOME/init/bash\n')
      # module load command outputs everything in stderr (><)
      QSF.write('module load anaconda/2.2.0 2>/dev/null\n')
      QSF.write('cd %s/pyCompHelio/Common\n' % pathToMPS())
      QSF.write('python -W ignore reduceOnCluster.py %s %s %d %d %d %s %s %i\n'%\
                (funcFile,argsFile,iJob,nProc,nTask,rType,rDir,saveH5py))
      if additionalCommands is not None:
        QSF.write('%s \n' % additionalCommands)

    # Submit
    subP    = subprocess.Popen('qsub %s' % subFileName,shell=True,\
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    out,err = subP.communicate()

    if err:
      print ("Job couldn't be submitted.")
      print()
      print (err)
      return
    # Get jobId
    jobIDs.append(out[:-1])

  # Wait for all jobs to finish
  print ("Submitted jobs to the cluster. Now waiting for completion.")
  nok,nnotok = waitTorqueJobs(ID,jobIDs)

  # Check missing files
  if nnotok != 0:
    print(("\nWARNING: %d jobs abnormally terminated! Result won't be computed\n" % nnotok))
    # Generate a log file which can be used with reRunReduceOnCluster
    with open(errorLogFileName,'w') as elf:
      elf.write('%d\n'%ID)
      print(("Reduction ID was stored in %s." % errorLogFileName))
    return
  else:
    return reducePartialReduction(ID,nTask,nProc,rType,deleteFiles,parallelRegroup,saveH5py=saveH5py)

def waitTorqueJobs(ID,jobIDs):

  rDir     = '/scratch/seismo/' + getpass.getuser() + '/reduceData/%d' %ID
  finished = False
  pings    = 0
  nJobs    = len(jobIDs)

  # Build output files list that should be present after jobs completion
  strJobList = ""
  for jobID in jobIDs:
    strJobList = strJobList + " " + rDir + "/jobLogs/job" + jobID + ".err"

  while not finished:
    if pings == 0:
      print('zzZZzz')
    if pings == 12:
      print('\n Entering deep slumber')
    if pings == 69:
      print('\n Comatose')

    if pings < 12:
      time.sleep(10)
      print('.',end='\r')
    elif pings < 69:
      time.sleep(60)
      print('.',end='\r')
    else:
      time.sleep(600)
      print('.',end='\r')

    subP = subprocess.Popen('ls %s | wc -l' % strJobList,shell=True,\
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    out,err  = subP.communicate()

    try:
      nfnshd   = int(out.split()[0])
      finished = (nfnshd == nJobs)
    except:
      pass
    pings   += 1
  print()
  # Safety minute
  time.sleep(20)

  # Count succesfull runs
  subP = subprocess.Popen("for i in %s; do wc -l $i | awk '{print $1}'; done" % strJobList,\
                          shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   
  out,err  = subP.communicate()
  statuses = NP.array(out.split('\n')[:-1]).astype(int)
  nok      = sum(statuses==0)
  nnotok   = nJobs-nok

  return nok,nnotok

def reRunReduceOnHelioCluster(errorLogFileName,func,args,nTask,nProc,imageSize,\
                              rType=None,debug=False,additionalCommands=None,\
                              deleteFiles=True,walltime=72,parallelRegroup=1,saveH5py=False):
  ''' From an ID, checks missing files and rerun the corresponding jobs '''

  try:
    elf = open(errorLogFileName)
    ID  = int(elf.read())
  except:
    raise Exception('Unable to read log file %s' % errorLogFileName)

  rDir         = '/scratch/seismo/' + getpass.getuser() + '/reduceData/%d' %ID
  argsFile     = rDir + '/arguments.pkl'
  funcFile     = rDir + '/function.pkl'
  saveObject(args,argsFile)
  saveObject(func,funcFile)
  
  # Detect missing results
  iFiles = []
  starts = partition(nTask,nProc) 
  for i in range(nbProc):
    if not os.path.exists('%s/res_%d_%d.npy'%(rDir,starts[i],starts[i+1])):
      iFiles.append(i)

  print(("iFiles", iFiles))

  # Submit on condor for missing files
  jobIDs = []
  for iFile in iFiles:
    subFileName  = rDir + '/toSubmit%d.sub' % iFile 
    # Submit
    subP    = subprocess.Popen('qsub %s' % subFileName,shell=True,\
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    out,err = subP.communicate()
    if err:
      print ("Job couldn't be submitted.")
      return
    jobIDs.append(out[:-1])

  # Wait for jobs to finish
  print ("Submitted jobs to the cluster. Now waiting for completion.")
  nok,nnotok = waitTorqueJobs(ID,jobIDs)

  # Check missing files
  if nnotok != 0:
    print(("\nWARNING: %d jobs abnormally terminated! Result won't be computed\n" % nnotok))
    return
  else:
    return reducePartialReduction(ID,nTask,nProc,rType,deleteFiles,parallelRegroup,saveH5py=saveH5py)

########################################################################
# Routine to find out what submission system your machine uses
def getClusterName():
  hn = getHostname()
  if 'seismo' in hn or 'helio' in hn:
    return 'condor'
  elif '.fast' in hn:
    return 'slurm'
  elif 'something for torque' in hn:
    return 'torque'
  else:
    raise Exception('Cluster no recognized check getClusterName in Common/parallelTools.py')





# Function in order to parallelize a class method
def parallelize_classObject(classMethodObject,*args):
  # classMethodObject = class.method
  res = classMethodObject(*args)
  return res



###############################3
# test function for testing reduce on cluster
def test_func(a,x):
  error = NP.random.randint(0,2)
  return 'a'
  if error:
    raise Exception('failed')
  else:
    return NP.array([1])
