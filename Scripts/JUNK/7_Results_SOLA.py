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

FILTER = False
nPad = 70
subSample = 2

OUTDIR = '/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/SOLA_Coeffs/'

QX = NP.arange(-30,31)
QY = NP.arange(-30,31)
SIGMA = 0.

DATN = '/scratch/ch3246/OBSDATA/modeCouple/Cartesian/SG_INVERSION/%s[%i][090][090.0][+00.0][+00.0]/' % ('mTrack_modeCoupling_3d_30deg',2200)
with h5py.File(DATN + '/%s/Bcoeffs/Bcoeffs_n%i_np%i_%i_%i%s.h5' % ('1Day',0,0,1920,1920*2,DATN.split('/')[-2].split('g')[-1]),'r') as h5f:
	nPad,xgrid,ygrid,dkx,dky = [NP.array(h5f[x]) for x in ['nPad','xgrid','ygrid','dkx','dky']]
xgrid = xgrid[:,0];ygrid = ygrid[0,:]
if subSample > 1:
	xgrid = subSampleVector(xgrid,1/subSample)
	ygrid = subSampleVector(ygrid,1/subSample)

with NP.load(OUTDIR + '/QX_%i/QY_%i/SOLA_coeffs_SIGMA%i_wider.npz' % (0,0,0)) as npydict:
	TargetDepths = npydict['TargetDepths']


if 'POL' not in locals():
	qxgrid,qygrid = NP.meshgrid(QX,QY,indexing='ij')
	absq = NP.sqrt((qxgrid*dkx*RSUN)**2 + (qygrid*dky*RSUN)**2)
	qxgrid = qxgrid.ravel();qygrid = qygrid.ravel();absq = absq.ravel()

	POL = NP.zeros((len(QX),len(QY),1,len(TargetDepths)),complex)
	PG = progressBar(len(qxgrid),'serial')
	for ii in range(len(qxgrid)):
		if absq[ii] >= 300:
			PG.update()
			continue

		with NP.load(OUTDIR + '/QX_%i/QY_%i/SOLA_coeffs_SIGMA%i_wider.npz' % (qxgrid[ii],qygrid[ii],SIGMA)) as npydict:
			coeffs = npydict['coeffs']
			Bcoeffs = npydict['Bcoefs']
			# sols    = npydict['sols']
			ns = npydict['ns']

		POL[NP.argmin(abs(QX - qxgrid[ii])),NP.argmin(abs(QY - qygrid[ii])),0,:] = NP.dot(coeffs,Bcoeffs)
		# POL[NP.argmin(abs(QX - qxgrid[ii])),NP.argmin(abs(QY - qygrid[ii])),0,:] = sols

		PG.update()
	del PG

	if subSample > 1:
		nPadN = nPad*subSample +len(QX)//subSample
	else:
		nPadN = nPad

	dz = FDM_Compact(TargetDepths)

	ur    = fft.ifftn(fft.ifftshift(NP.pad(POL,   ((nPadN,nPadN),(nPadN,nPadN),(0,0),(0,0)),constant_values=0),axes=(0,1)),axes=(0,1),norm='forward').real
	udivh = -NP.gradient(ur,TargetDepths,axis=-1)


z0ind = NP.argmin(abs(TargetDepths*1e-6 + 1))
z5ind = NP.argmin(abs(TargetDepths*1e-6 + 5))

fig,ax = plt.subplots(1,2,sharey=True,sharex=True,figsize=array([8.56, 4.8 ]))

DAT = ur[:,:,0,z5ind]
VMAX = NP.amax(abs(DAT))
im1 = ax[0].pcolormesh(xgrid,ygrid,DAT.T,vmin=-VMAX,vmax=VMAX,cmap='jet')
fig.colorbar(im1, ax=ax[0], orientation='vertical')
ax[0].set_title(r'$u_r$ at 5Mm')

DAT = udivh[:,:,0,z0ind]
VMAX = NP.amax(abs(DAT))
im2 = ax[1].pcolormesh(xgrid,ygrid,DAT.T,vmin=-VMAX,vmax=VMAX,cmap='jet')
fig.colorbar(im2, ax=ax[1], orientation='vertical')
ax[0].set_xlim(-45,45)
ax[0].set_ylim(-45,45)
ax[1].set_title(r'$\nabla_h\cdot u$ at Surface')


ax[0].set_ylabel(r'$y$ [Mm]',fontsize=15)
for ii in range(2):
	ax[ii].set_xlabel(r'$x$ [Mm]',fontsize=15)

plt.tight_layout()


# #---------------------------------------------------------------
# #					Average in Q
# #---------------------------------------------------------------

# ABSQ = NP.sqrt((qxgrid*dkx*RSUN)**2+(qygrid*dky*RSUN)**2)
# ABSQ = ABSQ.ravel()

# absQ_bins = NP.histogram(ABSQ,bins=30)[1]
# absQ_bins = absQ_bins[NP.argmin(abs(absQ_bins-15)):NP.argmin(abs(absQ_bins-415))]


# ur_q_avg = NP.zeros((len(xFinal),len(absQ_bins)-1));
# udivh_q_avg = NP.zeros((len(xFinal),len(absQ_bins)-1));

# for ii in range(len(xFinal)):
# 	DATur       = abs(ur_q[ALPHAInd,:,:,0,ii].ravel())**2
# 	DATudivh    = abs(udivh_q[ALPHAInd,:,:,0,ii].ravel())**2
# 	for binInd in range(len(absQ_bins)-1):
# 		inds = (ABSQ > absQ_bins[binInd])*(ABSQ < absQ_bins[binInd+1])
# 		ur_q_avg    [ii,binInd] = NP.nanmean(DATur       [inds])
# 		udivh_q_avg [ii,binInd] = NP.nanmean(DATudivh      [inds])


# fig,ax = plt.subplots(2,1,sharey=True,sharex=True,figsize=NP.array([8.52, 8.6 ]))
# ax20 = ax[0].twinx()
# ax21 = ax[1].twinx()
# ax[0].pcolormesh(xFinal*1e-6,absQ_bins[:-1]+NP.diff(absQ_bins)[0]/2,ur_q_avg.T)
# ax[0].contour(xFinal*1e-6,absQ_bins[:-1]+NP.diff(absQ_bins)[0]/2,ur_q_avg.T,cmap='jet')
# ax[1].pcolormesh(xFinal*1e-6,absQ_bins[:-1]+NP.diff(absQ_bins)[0]/2,udivh_q_avg.T)
# ax[1].contour(xFinal*1e-6,absQ_bins[:-1]+NP.diff(absQ_bins)[0]/2,udivh_q_avg.T,cmap='jet')
# ax[0].set_ylim([50,300])

# for axt in [ax20,ax21]:
# 	axt.set_ylim([50,300])
# 	axt.set_yticks(NP.linspace(50,300,9))
# 	axt.set_yticklabels(NP.around(2*NP.pi/(NP.linspace(50,300,9)/RSUN)*1e-6,1))
# 	axt.set_ylabel(r'$\lambda_h$ [Mm]',fontsize=15)
# ax[1].set_xlabel('Height [Mm]',fontsize=15)
# for ii in range(2):
# 	ax[ii].set_ylabel(r'$q$R$_\odot$',fontsize=15)
# 	ax[ii].tick_params(labelsize=12)
# fig.tight_layout()


#----------------------------------------------------------------
#                    Azimuthal average
#-----------------------------------------------------------------

xGrid,yGrid = NP.meshgrid(xgrid,ygrid,indexing='ij')
ABSX = NP.sqrt((xGrid)**2+(yGrid)**2)
ABSX = ABSX.ravel()

absX_bins = NP.histogram(ABSX,bins=100*subSample)[1]

ur_azi_avg = NP.zeros((len(TargetDepths),len(absX_bins)-1));
udivh_azi_avg = NP.zeros((len(TargetDepths),len(absX_bins)-1));

for ii in range(len(TargetDepths)):
	DATur       = ur[:,:,0,ii].ravel()
	DATudivh    = udivh[:,:,0,ii].ravel()
	for binInd in range(len(absX_bins)-1):
		inds = (ABSX > absX_bins[binInd])*(ABSX < absX_bins[binInd+1])
		ur_azi_avg    [ii,binInd] = NP.nanmean(DATur       [inds])
		udivh_azi_avg [ii,binInd] = NP.nanmean(DATudivh      [inds])


plt.figure()
plt.axhline(y=0,color='k')
plt.plot(TargetDepths*1e-6,ur_azi_avg[:,0],'b',lw = 2,label = r'$u_r$')
plt.plot(TargetDepths*1e-6,udivh_azi_avg[:,0]*1e7,'r',lw=2,label = r'$\nabla_h\cdot \mathbf{u} \times 10^7$')
plt.legend()
plt.xlabel('Height [Mm]',fontsize=15)
# plt.title(r'$\alpha = %1.2e$' % alpha[ALPHAInd]))
plt.tight_layout()

NP.save('data/aziFlows_SOLA.npy',[TargetDepths*1e-6,ur_azi_avg[:,0]])

fig,ax = plt.subplots(1,2,sharey=True,sharex=True,figsize=array([8.56, 4.8 ]))
VMAXur = NP.amax(ur_azi_avg)
VMAXudivh = NP.amax(udivh_azi_avg)

im1 = ax[0].pcolormesh(absX_bins[:-1] + NP.diff(absX_bins)[0]/2,TargetDepths*1e-6,ur_azi_avg,vmin=-VMAXur,vmax=VMAXur,cmap='jet')
im2 = ax[1].pcolormesh(absX_bins[:-1] + NP.diff(absX_bins)[0]/2,TargetDepths*1e-6,udivh_azi_avg,vmin=-VMAXudivh,vmax=VMAXudivh,cmap='jet')

ax[0].set_xlim([0,45])
ax[0].set_ylabel('Height [Mm]',fontsize=15)
for ii in range(2):
	ax[ii].set_xlabel('Distance [Mm]',fontsize=15)
	ax[ii].set_title([r'$u_z$',r'$\nabla_h\cdot u$'][ii])

fig.colorbar(im1, ax=ax[0], orientation='vertical')
fig.colorbar(im2, ax=ax[1], orientation='vertical')
plt.tight_layout()


import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("InversionResults%s_SOLA.pdf" % (['','_FILT%i' % int(FILTER)][int(FILTER is not False)]))
for fig in range(1, figure().number): ## will open an empty extra figure :(
    pdf.savefig( fig )
pdf.close()

plt.close('all')

