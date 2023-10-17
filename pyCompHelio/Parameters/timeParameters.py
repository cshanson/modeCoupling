import numpy as NP

from ..Common import *

class timeParameters(object):
  ''' class containing time parameters loaded from config file:
      frequencies, time steps, etc.
  '''

  def __init__(self,config=None,time=None,nbT=None,sampling=1):
      ''' config is a myConfigParser instance '''

      self.sampling_ = sampling
      if time is not None and nbT is not None:
        self.setObservationalTime(time/sampling,nbT/sampling)
      elif config:
        if config('Frequencies',0):
          Fs,self.Nt_,self.limit_ = getFrequencies(config)
          Fs           = Fs[::sampling]
          self.Nt_    //= sampling
          self.limit_ //= sampling
          self.omega_  = Fs*2*NP.pi
          try:
            self.homega_ = self.omega[1]-self.omega[0]
          except:
            self.homega_ = 1.

          fType = config('Frequencies').split()[0]
          if (fType == "SINGLE"):
            self.Nt_ = 1
            self.ht_ = 1.e0
          elif (fType != "RANGE"):
            T = 1./(NP.abs(Fs[1]-Fs[0]))
            self.setObservationalTime(T/sampling,self.Nt_)
            self.limit_ = self.Nt_//2+1


        elif config('Time',0):
          options = config('Time').split()
          try:
            T  = evalFloat(options[0])
            dt = evalFloat(options[1])
            Nt = int(T/dt)
            self.setObservationalTime(T/sampling,Nt/sampling)
            self.limit_ = Nt
          except:
            raise Exception('Could not read time parameters from config file.')
        else:
          raise Exception('Could not read time parameters from config file.')
      else:
        raise Exception('No time parameters or given config file to initialize time parameters structure')

  def setObservationalTime(self,T,Nt):
      ''' Initializes an object time parameter using the total observation
          time T and the number of points in the time domain nbT. 
          Create vectors containing the time and frequency points 
          and initialize frequency and time sampling.
      '''

      self.T_      = T
      self.Nt_     = Nt
      self.homega_ = 2.e0*NP.pi/T

      if self.Nt_ % 2 == 0:
        # even number of points, t goes from (-Nt/2 to Nt/2-1)*ht
        self.ht_    = T/float(Nt)
        self.t_     = NP.linspace(-Nt/2,Nt/2-1,Nt)*self.ht_
        self.omega_ = NP.linspace(-Nt/2,Nt/2-1,Nt)*self.homega_
        # Reorder for FFTs
        self.omega_ = NP.fft.ifftshift(self.omega_)
        self.t_     = NP.fft.ifftshift(self.t_)

      else:
        self.ht_    = T/float(Nt)
        self.t_     = NP.linspace(-(Nt-1)/2,(Nt-1)/2,Nt)*self.ht_
        self.omega_ = NP.linspace(-(Nt-1)/2,(Nt-1)/2,Nt)*self.homega_
        self.omega_ = NP.fft.ifftshift(self.omega_)
        self.t_     = NP.fft.ifftshift(self.t_)

  def changeTimeStep(self,newTimeStep):
      ''' Create a new time object from a previous one by changing 
          the time step. Returns the new object and the number of points 
          necessary for the padding if a fft has to be performed.
      '''

      ratio   = int(self.ht_/newTimeStep)
      timePts =  ratio*self.Nt_
      if timePts%2 != self.Nt_%2: # to ensure that nbPad is an integer
        timePts = timePts+1
      newTime = timeParameters(time=self.T_,nbT=timePts)
      nbPad   = int((timePts-self.Nt_)/2)
      return newTime,nbPad

  def addSymmetricPart(self, field):
      ''' Adds to a causal field with last component omega 
          the part with negative frequencies. Given field[...,w], 
          add conj(field[...,-w])
      '''
      field_rev = field[:,::-1]
      # Shift due to the zero frequency
      if (self.Nt_ % 2 == 0):
        field_rev = NP.roll(field_rev,1,axis=-1)      
      field += NP.conj(field_rev)
      return field

  def useCausality(self):
      return self.Nt_ != 1 and self.limit_ == int((self.Nt_+1)/2)

  def getFrequencies(self):
     return getFrequencies(self.config_)[0]

  def getNumberFrequencies(self):
     return getFrequencies(self.config_)[1]

  def getLimitFrequencies(self):
     return getFrequencies(self.config_)[2]

  def addSymmetricPart(self,data):
    ''' Adds to a causal field with last component omega 
        the part with negative frequencies. 
        Given field[...,w], returns field[...,w] + conj(field[...,-w])
    '''

    field_rev = field[:,::-1]
    # Shift due to the zero frequency
    if (self.Nt_%2==0):
      field_rev = NP.roll(field_rev, 1, axis=-1) 
    return field + NP.conj(field_rev)

###############################################################################
  
def getFrequencies(config):
  ''' Returns information on the frequencies from a config file
      (outside of time Parameters class so it can be used by
      runMontjoie without complications)
      Limit is the number of frequencies to be processed
      (if we take or not symmetry into account)
  '''

  # Default values
  type  = 'SINGLE'
  val1  = 1.e-3
  limit = 0

  options  = config('Frequencies','SINGLE 0.').split()
  if options:
    try:
      type = options[0]
      val1 = evalFloat(options[1])

      if type == 'SINGLE':
        Fs = [val1]
        Nf = 1

      elif type == 'RANGE':
        fmin = val1
        fmax = evalFloat(options[2])
        hf   = evalFloat(options[3])
        if isinstance(hf,int):
          # number of frequencies
          Fs   = NP.linspace(fmin,fmax,hf)
        else:
          # frequency resolution
          Fs   = NP.arange(fmin,fmax,hf)
        Nf   = len(Fs)

      # POSITIVE ONLY OR ALL
      else:

        dt = evalFloat(options[2])
        T  = val1
        Nt = int(T/dt)
        if (Nt%2 != 1):
          T  = T+dt
          Nt = T/dt
          print(('Warning : T has been set to T+dt so that the number of time points Nt = ', Nt,  'is odd'))

        Nthalf  = int((Nt-1)/2)
        hf      = float(1./T)                        # frequency resolution
        intfreq = NP.array(list(range(-Nthalf, Nthalf+1))) # in Hz
        intfreq = NP.roll(intfreq, Nthalf+1);        # put the 0 frequency first
        Fs      = intfreq*hf                         # in Hz

        if (type == "ALL"):
          Nf = Nt
        if (type == "POSITIVE_ONLY"):
          limit = int((Nt+1)/2)
          Nf    = Nt

    except:
      raise Exception('Impossible to read frequency parameters.')

    if limit == 0:
      limit = len(Fs)

  return NP.array(Fs),Nf,limit

