#!/scratch/ch3246/project/PythonEnv/miniconda3/envs/dalma-python3-CPU/bin/python

pathToRoutines = '/home/ch3246/mps_montjoie/'
import sys
sys.path.insert(0,pathToRoutines)
print("Importing pythonRoutines")
import numpy as NP
from pyCompHelio import *
from matplotlib.pyplot import *
from astropy.io import fits
from scipy.sparse import coo_matrix,csr_matrix,hstack,vstack
plt.ion()



x = NP.linspace(-100,100,201)
y = NP.linspace(-100,100,201)

xg,yg = NP.meshgrid(x,y,indexing='ij')
dist = NP.sqrt(xg**2 + yg**2)

model = NP.exp(-(dist**2) / (2*20**2)) * NP.sin(dist/2)

plt.figure()
plt.pcolormesh(xg,yg,model)

plt.figure()
plt.pcolormesh(fft.fftshift(fft.fftn(model)).real)


ABSX = dist.ravel()
absX_bins = NP.histogram(ABSX,bins=100)[1]

model_azi = NP.zeros((len(absX_bins)-1));

DAT       = model.ravel()
for binInd in range(len(absX_bins)-1):
	inds = (ABSX > absX_bins[binInd])*(ABSX < absX_bins[binInd+1])
	model_azi    [binInd] = NP.nanmean(DAT       [inds])

plt.figure()
plt.plot(absX_bins[:-1] + NP.diff(absX_bins)[0]/2,model_azi)

plt.figure()
plt.plot(fft.rfft(model_azi).real)
plt.plot(fft.rfft(model_azi).imag)