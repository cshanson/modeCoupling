# Purpose of this routine is just to specify the directories we will use

# Path to routines, specify the path the to modeCoupling directory
pathToRoutines = '~/modeCoupling/'
import sys
sys.path.insert(0,pathToRoutines)
print("Importing pythonRoutines")
import numpy.fft as fft


# Where most the data we use or will create are stored
# Change as you see fit
DATADIR = '/home/hanson/Desktop/github_testing/DATA/'
