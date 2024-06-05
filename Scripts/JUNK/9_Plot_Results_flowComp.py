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


DERIV = 1


BASIS = 'Bspline'

if BASIS == 'Bspline':
	BasisStr = '_Bspline'
else:
	BasisStr = ''

LM = 2

print(text_special('Plotting for %s' %['CHEBYSHEV','BSPLINE'][int(BASIS=='Bspline')],'g',True,True))

#--------------------------------------------
if 'POWS' not in locals():
	POWS = [];noisePropts = []
	for DERIV in [0,1]:
		DICT = NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/Inv_Results_test_v2_L%i%s.npz' % (LM,BasisStr))
		DICTn=NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/NoiseProp_test_v2_L%i%s.npz' % (LM,BasisStr))
		Basis1D    = NP.load('/scratch/ch3246/Private/Mode_Coupling/SG_Inversions/Basis%s.npy' % (BasisStr),allow_pickle=True)[0]

		QX,QY,SIGMA,dkx,dky,dw = [DICT[x] for x in ['QX','QY','SIGMA','dkx','dky','dw']]
		res_kw      = DICT['res']
		noiseProp   = DICTn['res']
		alphaGrid   = DICT['alphaGrid']
		guessAlpha  = NP.argmin(abs(alphaGrid - DICT['guessAlpha']))
		arrayLabel = DICT['arrayLabel']
		noisePropt = NP.zeros((2,)+res_kw.shape[-4:-1] + (len(Basis1D.x_),))
		for qx_ind in range(len(QX)):
			for qy_ind in range(len(QY)):
				for sigma_ind in range(len(SIGMA)): 
					tmpP = noiseProp[qx_ind,qy_ind,sigma_ind][:Basis1D.nbBasisFunctions_,:Basis1D.nbBasisFunctions_]
					tmpT = noiseProp[qx_ind,qy_ind,sigma_ind][Basis1D.nbBasisFunctions_:,Basis1D.nbBasisFunctions_:]
					noisePropt[0,qx_ind,qy_ind,sigma_ind] = NP.diag(Basis1D.reconstructFromBasis(Basis1D.reconstructFromBasis(tmpP,axis=-1,derivative=DERIV),derivative=DERIV,axis=-2))
					noisePropt[1,qx_ind,qy_ind,sigma_ind] = NP.diag(Basis1D.reconstructFromBasis(Basis1D.reconstructFromBasis(tmpT,axis=-1,derivative=DERIV),derivative=DERIV,axis=-2))


		POWt = NP.nanmean(abs(Basis1D.reconstructFromBasis(res_kw[:,:,guessAlpha].real,axis=-1,derivative=DERIV)+1.j*Basis1D.reconstructFromBasis(res_kw[:,:,guessAlpha].imag,axis=-1,derivative=DERIV))**2,axis=0)
		POWS.append(POWt)
		noisePropts.append(noisePropt)



for II in range(2):
	POW = POWS[II]
	noisePropt = noisePropts[II]
	QXIND = 20
	QYIND = 15

	plt.figure(1)
	linefmt = ['-','--']
	for ii in range(2):
		plt.plot(Basis1D.x_*1e-6,[POW,noisePropt][ii][0,QXIND,QYIND,0]/NP.amax(POW[0,QXIND,QYIND,0]),color='C%i' % (2*II),linestyle=linefmt[ii],label='%s %s' % ([r'$u_r$',r'$\nabla_h\cdot u$'][II],['Results','Noise Model'][ii]))
	plt.ylabel('Normalized Power')
	plt.xlabel('Height [Mm]')
	plt.legend()
	plt.title(r"[$q_x$R$_\odot$,$q_x$R$_\odot$]=[%1.2f,%1.2f]" % (QX[QXIND]*RSUN*dkx,QY[QYIND]*RSUN*dky))

	qx_grid = QX*dkx*RSUN
	qy_grid = QY*dky*RSUN
	depthInd = NP.argmin(abs(Basis1D.x_*1e-6))

	ABSQ = NP.sqrt((qx_grid[:,None])**2+(qy_grid[None,:])**2)
	ABSQ = ABSQ.ravel()

	# fig,ax = plt.subplots(1+len(Ngrid),1,sharex=True)
	absk_bins = NP.histogram(ABSQ,bins=25)[1]
	absk_bins = absk_bins[NP.argmin(abs(absk_bins-35)):NP.argmin(abs(absk_bins-415))]
	# ax[0].set_ylabel(r'N samples')
	# ax[0].set_title(r'$\sigma=%1.2f\mu$Hz' % (sigma_grid[0]))

	# mask = NP.where(qx_grid < 0,NP.nan,1)[:,None] *NP.ones(len(qy_grid))[None,:]
	# mask = mask.ravel()


	power_kk = NP.zeros((2,len(absk_bins),POW.shape[-1]))
	power_noise   = copy.copy(power_kk)
	std_kk   = copy.copy(power_kk)

	power_noise_nn = []
	for flowComp in range(2):
			for dd in range(POW.shape[-1]):
				DAT = POW[flowComp,:,:,0,dd].reshape(-1)
				DATn = noisePropt[flowComp,:,:,0,dd].reshape(-1)
				# DAT_noise = NP.nanmean(POW_noise[...,indl:indu],axis=-1)[nn].reshape(-1)
				for binInd in range(len(absk_bins)-1):
					inds = (ABSQ > absk_bins[binInd])*(ABSQ < absk_bins[binInd+1])
					power_kk[flowComp,binInd,dd] = NP.nanmean(DAT[inds])
					power_noise[flowComp,binInd,dd] = NP.nanmean(DATn[inds])
					std_kk[flowComp,:,dd] = NP.nanstd(DAT[inds])#/NP.sqrt(sum(inds))
				# power_nn.append(power_kk)
				# power_noise_nn.append(power_noise)
				# std_nn.append(std_kk)
	# power_nn = NP.array(power_nn);power_noise_nn = NP.array(power_noise_nn)
	# std_nn = NP.array(std_nn)

	plt.figure(2)
	ABSKI=6
	linefmt = ['-','--']
	for ii in range(2):
		plt.plot(Basis1D.x_*1e-6,[power_kk,power_noise][ii][0,ABSKI,:]/NP.amax(power_kk[0,ABSKI,:]),color='C%i' %(2*II),linestyle=linefmt[ii],label='%s %s' % ([r'$u_r$',r'$\nabla_h\cdot u$'][II],['Results','Noise Model'][ii]))
	plt.fill_between(Basis1D.x_*1e-6,(power_kk[0,ABSKI,:]+std_kk[0,ABSKI,:])/NP.amax(power_kk[0,ABSKI,:]),(power_kk[0,ABSKI,:]-std_kk[0,ABSKI,:])/NP.amax(power_kk[0,ABSKI,:]),color='C%i' %(2*II),alpha=0.3)
	plt.ylabel('Normalized Power')
	plt.xlabel(r'Height [Mm]',fontsize=15)
	plt.title(r'$|q|$R$_\odot$=%1.2f' % absk_bins[ABSKI])
	plt.legend()
	# plt.yscale('log')

	plt.figure(3)

	linefmt = ['-','--']
	for ii in range(2):
		plt.plot(absk_bins,[power_kk,power_noise][ii][0,:,depthInd]/NP.amax(power_kk[0,:,depthInd]),color='C%i' %(2*II),linestyle=linefmt[ii],label='%s %s' % ([r'$u_r$',r'$\nabla_h\cdot u$'][II],['Results','Noise Model'][ii]))
	plt.ylabel('Normalized Power')
	plt.xlabel(r'|q|R$_\odot$',fontsize=15)
	plt.yscale('log')
	plt.legend()

	fig,ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=NP.array([9,4.8]))
	ax = [ax]
	for ii in range(1):
		ax[ii].pcolormesh(Basis1D.x_*1e-6,absk_bins,(power_kk-power_noise)[ii]/NP.amax(power_kk[ii]),vmax=0.3)
		ax[ii].contour(Basis1D.x_*1e-6,absk_bins,(power_kk-power_noise)[ii]/NP.amax(power_kk[ii]),cmap='jet',levels=NP.linspace(0.1,1,15))

		ax[ii].set_xlabel('Height [Mm]',fontsize=15)
		ax[ii].tick_params(labelsize=12)
		ax[ii].set_title([r'$u_r$',r'$\nabla_h\cdot u$'][II])
	ax[0].set_ylabel(r'|q|R$_\odot$',fontsize=15)

	fig,ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=NP.array([9,4.8]));ax = [ax]
	depthInd2 = NP.argmin(abs(Basis1D.x_*1e-6))
	ax[0].pcolormesh(QX*dkx*RSUN,QY*dky*RSUN,POW[0,:,:,0,depthInd2].T)
	# ax[1].pcolormesh(QX*dkx*RSUN,QY*dky*RSUN,POW[1,:,:,0,depthInd2].T)
	ax[0].set_ylabel(r'$q_y$R$_\odot$',fontsize=15)
	ax[0].set_xlabel(r'$q_x$R$_\odot$',fontsize=15)
	# ax[1].set_xlabel(r'$q_x$R$_\odot$',fontsize=15)
	ax[0].set_title([r'$u_r$',r'$\nabla_h\cdot u$'][II])
	# ax[1].set_title('Toroidal')

	# for ii in range(2):
	# 	ax[ii].axvline(x=0,color='w')
	# 	ax[ii].axhline(y=0,color='w')

	# plt.tight_layout()





import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("results_flowComp.pdf")
for fig in range(1, figure().number): ## will open an empty extra figure :(
    pdf.savefig( fig )
pdf.close()

plt.close('all')