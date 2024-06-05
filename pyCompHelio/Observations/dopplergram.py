from ..Common     import *
from ..Parameters import *
import os
import datetime
import numpy             as NP
import matplotlib.pyplot as PLOT

class dopplergram:
    ''' COMMENTS please '''

    def __init__(self,seriesPrefix,dateIni,nDays,nbDopplersPerDay,params):
      self.sPrefix_         = seriesPrefix
      self.dateIni_         = dateIni
      self.nDays_           = nDays
      self.nDopplersPerDay_ = nDopplersPerDay
      self.params_          = params

    def setSeriesDirectories(self):
      ''' Finds the location of the raw datacubes from the DRMS 
          depending on the serie prefix 
          and dates given in the constructor
      '''

      self.sDirectories_ = []
      os.system('show_info -p ds=%s n=%i > tmp.txt' %(self.sPrefix_,self.nDays_))
      with open('tmp.txt') as fh:
        for line in fh.readlines():
          self.sDirectories_.append(line.rstrip('\n'))
      remove('tmp.txt')

    def setOutputFileNames(self,prefix):
      self.names_ = self.generateFileNames(prefix)

    def generateFileNames(self,prefix):
      ''' Writes the name of the files that will contain the filtered data. 
          It contains a prefix, then the mode filter, the date 
          and the number of the dopplergram during this day, 
          for example phikw_f_mode_2010.10.27_3of3.mat
      '''
      names_ = []
      date1 = datetime.datetime.strptime(self.dateIni_,"%Y-%m-%d")
      for i in range(self.nDays_):
        date   = date1+datetime.timedelta(days=i)
        data   = data.strftime("%Y.%m.%d")
        dnames = []
        for j in range(self.nDopplersPerDay_):
          dnames.append('%s_%s_%iof%i.npy'%(prefix,date,j+1,self.nbDopplersPerDay_))
        names.append(dnames)
      return names

    def setOutputDirectory(self, directory):
      self.directory_ = directory

    def setFilterOmega(self,omegaMin,omegaMax):
      self.omegaMin_ = omegaMin
      self.omegaMax_ = omegaMax
  
    def setFilterK(self,kMin=None,kMax=None):
      if kMin is None:
         self.kMin_ = self.omegaMin_**2/GSUN
      else:
        self.kMin_ = kMin
      if kMax is None:
         self.kMax_ = self.omegaMax_**2/GSUN
      else:
        self.kMax_ = kMax
  
    def setModeFilter(self,typeOfModeFilter):
      self.typeOfModeFilter_ = typeOfModeFilter
  
    def createFilteredObservations(self):
      ''' Reads the cubes contained in seriesDirectories (given by the DRMS) 
          and apply the different filters.
          Writes the results in the (k,w) space in names_
      '''

      # Shortcuts
      geom  = self.params_.geom_
      time  = self.params_.time_
      kx    = NP.fft.ifftshift(geom.k_[0] )
      ky    = NP.fft.ifftshift(geom.k_[1] )
      omega = NP.fft.ifftshift(time.omega_)
      if hasattr(self,'kMin_'):
        # create the filter in the k-space if necessary
        Fk      = NP.ones(tuple(geom.N_[1::-1]))
        k1g,k2g = NP.meshgrid(kx,ky)
        kk      = NP.hypot(k1g,k2g)
        Fk[kk<self.kMin_] = 0.e0
        Fk[kk>self.kMax_] = 0.e0
  
      if hasattr(self,'typeOfModeFilter_'):
        # Create the mode filter if necessary
        F = modeFilter.get3DFilter(self.typeOfModeFilter_,self.params_,\
                                   self.omegaMin_,self.omegaMax_)
  
      for i in range(self.nDays_):

        # Test if all the files of this day already exist 
        # to avoid loading the dopplergram in this case
        alreadyCreated = True
        for j in range(self.nDopplersPerDay_):
          fullname        = '%s%s.npy' %(self.directory_,self.names_[i][j])
          alreadyCreated *= os.path.isfile(fullname)
  
        if not alreadyCreated:

          doppler = fits.open('%s/MTcube.fits' %self.sDirectories_[i])
          doppler = doppler[0].data
          # reorder the data in ky,kx,omega
          doppler = NP.transpose(doppler,(1,2,0)) 

          for j in range(self.nbDopplersPerDay_):

            # Test if the files already exist
            fullname = '%s%s' % (self.directory_,self.names_[i][j])
            if not os.path.isfile(fullname):

              # cut the dopplergram in nbDopplersPerDay_
              dopplergramCrt = doppler[:,:,j*time.Nt_:(j+1)*time.Nt_]
              # Go to (k,w) space and apply the different filters
              dopplergramkw = NP.fft.fftshift(solarFFT.FFTn(NP.fft.ifftshift(dopplergramCrt),time,geom))  
  
              if hasattr(self,'omegaMin_'):
                dopplergramkw[:,:,abs(omega)<self.omegaMin_] = 0
                dopplergramkw[:,:,abs(omega)>self.omegaMax_] = 0 
  
              if hasattr(self,'kMin_'):
                dopplergramkw = dopplergramkw*Fk[:,:,NP.newaxis]
  
              if hasattr(self,'typeOfModeFilter_'):
                dopplergramkw = dopplergramkw*F
  
              # save with 0 frequency at the beginning
              dopplergramkw = NP.fft.ifftshift(dopplergramkw)
              NP.save(fullname,dopplergramkw)
