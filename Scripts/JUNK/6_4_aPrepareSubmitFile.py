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

plt.close('all')

QX = NP.arange(-30,31)
QY = NP.arange(-30,31)

# Submit using
# slurm_parallel_ja_submit_MJ.sh -t 48:00:00 -N 16 -c 5 -L slurm_files/JOBARRAY job_SOLAcoeffs.sh


# ---------------Load spacing in k ----------------
with h5py.File('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/Bcoeffs_AVG' + '/Bcoeffs_n%i_np%i%s.h5' % (0,0,''),'r') as h5f:
	dkx,dky = [NP.array(h5f[x]) for x in ['dkx','dky']]


qxgrid,qygrid = NP.meshgrid(QX,QY,indexing='ij')
absq = NP.sqrt((qxgrid*dkx*RSUN)**2 + (qygrid*dky*RSUN)**2)

mask = NP.where(absq < 300,True,False).ravel()

qx_job = qxgrid.ravel()[mask]#[:9]
qy_job = qygrid.ravel()[mask]#[:9]

with open('job_SOLAcoeffs.sh','w') as file1:
	for ii in range(len(qx_job)):
		file1.write(os.getcwd() + '/6_4_Inversion_SOLA_coeffs.py %i %i\n' % (qx_job[ii],qy_job[ii]))
