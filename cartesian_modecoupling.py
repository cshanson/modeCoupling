import numpy as NP
from matplotlib.pyplot import *
import matplotlib.pylab as plt
from astropy.io import fits
from scipy.sparse import coo_matrix,csr_matrix,hstack,vstack
import pyfftw

from ..Common       import *
from ..Parameters   import *

plt.ion()

plt.close('all')


class cartesian_modeCoupling(object):

	def __init__(self,obs_fitsPath,apodization_fitsPath=None,dxFactor = 1,dyFactor = 1,dt = 45.,kMax=None,OmegaMax=None,\
					daysPad=None,timeSubSample=1,rdvFits = True,timeInds = None):

		self.obs_fitsPath_         = obs_fitsPath
		self.apodization_fitsPath_ = apodization_fitsPath

		if isinstance(obs_fitsPath,list):
			with fits.open(obs_fitsPath[0]) as hdu:
				self.padLength_ = int(daysPad*24*3600/(45*timeSubSample))
				self.Nt_ = hdu[1].header['NAXIS3']*len(obs_fitsPath)//timeSubSample+self.padLength_*(len(obs_fitsPath)-1)
				self.Nx_ = hdu[1].header['NAXIS1']
				self.Ny_ = hdu[1].header['NAXIS2']	
		else:
			with fits.open(obs_fitsPath) as hdu:
				self.Nt_ = hdu[1].header['NAXIS3']//timeSubSample
				self.Nx_ = hdu[1].header['NAXIS1']
				self.Ny_ = hdu[1].header['NAXIS2']

		if timeInds is not None:
			self.Nt_ = timeInds[1] - timeInds[0]
		self.timeInds_ = timeInds
		
		self.dt_ = dt*timeSubSample
		self.dxFactor_ = dxFactor
		self.dyFactor_ = dyFactor
		self.kMax_ = kMax
		self.OmegaMax_ = OmegaMax
		self.timeSubSample_ = timeSubSample
		self.rdvFits_ = rdvFits

		if apodization_fitsPath is None:
			# computeApod = True
			# if os.path.isfile(os.getcwd() + '/Apodize.fits'):
				# with fits.open(os.getcwd() + '/Apodize.fits') as hdu:
					# data = hdu[0].data 
				# if data.shape[0] == self.Ny_ and data.shape[1] == self.Nx_:
					# computeApod = False

			# if computeApod:
			x = NP.linspace(-1,1,self.Nx_)
			y = NP.linspace(-1,1,self.Ny_)
			xg,yg = NP.meshgrid(x,y,indexing='ij')
			apod_data = smoothRectangle(NP.sqrt(xg**2+yg**2),-0.95,0.95,0.1)
			hdu = fits.PrimaryHDU(apod_data)
			hdu.writeto(os.getcwd() + '/Apodize.fits',overwrite=True)
			self.apodization_fitsPath_ = os.getcwd() + '/Apodize.fits'
	
	def readFits(self,storeInInstance=False):
		# reads in the V_xyt cube, stores it in globals and returns dimensions
		if isinstance(self.obs_fitsPath_,list):
			for ii in range(len(self.obs_fitsPath_)):
				with fits.open(self.obs_fitsPath_[ii]) as hdu:
					HEADER = hdu[1].header
					if ii == 0:
						phi_xyt = hdu[1].data[::self.timeSubSample_]
					else:
						phi_xyt = NP.concatenate([phi_xyt,NP.zeros((self.padLength_,)+phi_xyt.shape[1:],phi_xyt.dtype)],axis=0)
						phi_xyt = NP.concatenate([phi_xyt,hdu[1].data[::self.timeSubSample_]],axis=0)


		else:
			with fits.open(self.obs_fitsPath_) as hdu:
				HEADER = hdu[1].header
				phi_xyt = hdu[1].data[::self.timeSubSample_]

		phi_xyt = NP.nan_to_num(phi_xyt)


		if storeInInstance:
			self.phi_xyt_    = phi_xyt
			self.header_xyt_ = HEADER

		return phi_xyt,HEADER

	def TEST(self,ii):
		print(ii)
		return ii

	def parallel_TEST(self,N):
		reduce(parallelize_classObject,(self.TEST,NP.arange(N)),N,8,progressBar=True)


	def computeFFT(self,fitsPathLoad = None,fitsPathSave=None,fitsPath_power = None,storeInInstance=False,removeTemporalAverage=False,FFTforloop=False):
		# reads in dopplercube with readFits, applies apodization, computes fourier transform,save spectrum

		if storeInInstance:
			global phi_kw

		if fitsPathLoad is not None and fitsPathSave is None:
			with fits.open(fitsPathLoad) as hdu:
				dat = hdu[0].data
			phi_kw = dat[0] + 1.j*dat[1]
			return phi_kw
		
		# read doppler cube
		try:
			phi_xyt = self.phi_xyt_
			header_xyt = self.header_xyt_
		except:
			phi_xyt,header_xyt = self.readFits(False)

		if self.timeInds_ is not None:
			phi_xyt = phi_xyt[self.timeInds_[0]:self.timeInds_[1]]


		if removeTemporalAverage:
			phi_xyt = phi_xyt - NP.nanmean(phi_xyt,axis=0)[None,...]

		# read apodize function
		with fits.open(self.apodization_fitsPath_) as hdu:
			apodize = NP.float32(hdu[0].data[None,...])

		phi_xyt = phi_xyt*apodize

		# compute fft
		start = time.time()
		if FFTforloop:
			phi_kw = NP.zeros(phi_xyt.shape,'complex64')
			print('Computing FFt in for loop');PG = progressBar(len(phi_xyt),'serial')
			for ii in range(len(phi_xyt)):
				# phi_kw[ii] = pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.fftn(phi_xyt[ii],norm='ortho',axes=(0,1)),axes=(0,1))/(2*NP.pi)**3
				phi_kw[ii] = NP.fft.fftshift(NP.fft.fftn(phi_xyt[ii],norm='ortho',axes=(0,1)),axes=(0,1))/(2*NP.pi)**3
				PG.update()
			del PG
		else:
			# phi_kw = pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.fftn(phi_xyt,norm='ortho',axes=(1,2)),axes=(1,2))/(2*NP.pi)**3
			phi_kw = NP.fft.fftshift(NP.fft.fftn(phi_xyt,norm='ortho',axes=(1,2)),axes=(1,2))/(2*NP.pi)**3
		del phi_xyt
		
		if self.kMax_ is not None:
			kx,ky,omega = self.computeFreq(True)
			indkeepx = abs(kx) <= self.kMax_
			indkeepy = abs(ky) <= self.kMax_
			phi_kw   = phi_kw[:,indkeepy][:,:,indkeepx]

		# phi_kw = pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.ifft(phi_kw,norm='ortho',axis=0),axes=0)#[:self.Nt_//2]
		phi_kw = NP.fft.fftshift(NP.fft.ifft(phi_kw,norm='ortho',axis=0),axes=0)#[:self.Nt_//2]
		print(phi_kw.shape)

		if self.OmegaMax_ is not None:
			kx,ky,omega = self.computeFreq(True)
			indkeepw = abs(omega) <= self.OmegaMax_
			phi_kw = phi_kw[indkeepw]

		# phi_kw = fft.fftshift(fft.fftn(fft.irfft(phi_xyt * hdu[0].data[None,...],n=powerBitLength(phi_xyt.shape[0]),norm='ortho',axis=0),s=(powerBitLength(phi_xyt.shape[1]),powerBitLength(phi_xyt.shape[2])),norm='ortho',axes=(1,2)),axes=(1,2))/(2*NP.pi)**3
		end = time.time()
		print('fftw compute time: ',end-start)


		# Save power spectrum
		if fitsPathSave is not None:
			header_kw = copy.copy(header_xyt)
			header_kw['NAXIS'] = 4
			header_kw['NAXIS3'] = len(phi_kw)
			header_kw['NAXIS4'] = 2
			hdul = fits.PrimaryHDU(NP.array([phi_kw.real,phi_kw.imag]).astype('float32'),header=header_kw)
			hdul.writeto(fitsPathSave,overwrite=True)

		if fitsPath_power is not None:
			hdul = fits.PrimaryHDU(abs(phi_kw)[::5],header=fits.open(self.obs_fitsPath_)[1].header)
			hdul.writeto(fitsPath_power,overwrite=True)

		return phi_kw

	def computeRealSpace(self):

		try:
			self.kx_
		except:
			self.computeFreq()

		tgrid = NP.linspace(0,self.Nt_*self.dt_,self.Nt_)
		xgrid = NP.linspace(0,len(self.kx_)*(NP.pi/self.kx_.max()),len(self.kx_))
		ygrid = NP.linspace(0,len(self.ky_)*(NP.pi/self.ky_.max()),len(self.ky_))

		self.xgrid_ = xgrid
		self.ygrid_ = ygrid
		self.tgrid_ = tgrid

		return xgrid,ygrid,tgrid


	def computeFreq(self,returnComplete=False):
		# computes the spatial and temporal frequencies (kx,ky,omega)
		omega = NP.fft.fftshift(NP.fft.fftfreq(self.Nt_,self.dt_)*2*NP.pi )
		sizePerPixel = 0.04/180*RSUN*NP.pi # 0.04 is the standard mTrack size
		kx    = NP.fft.fftshift(NP.fft.fftfreq(self.Nx_,sizePerPixel* self.dxFactor_) * 2 * NP.pi) # At disk center
		ky    = NP.fft.fftshift(NP.fft.fftfreq(self.Ny_,sizePerPixel* self.dyFactor_) * 2 * NP.pi) # At disk center
		# kx    = NP.fft.fftshift(NP.fft.fftfreq(self.Nx_,348000*self.dxFactor_) * 2 * NP.pi) # At disk center
		# ky    = NP.fft.fftshift(NP.fft.fftfreq(self.Nx_,348000*self.dyFactor_) * 2 * NP.pi) # At disk center
		# omega = omega[:len(omega)//2]

		if self.kMax_ is not None and returnComplete is False:
			indkeepx = abs(kx) <= self.kMax_
			indkeepy = abs(ky) <= self.kMax_
		else:
			indkeepx = NP.arange(len(kx))
			indkeepy = NP.arange(len(ky))
		if self.OmegaMax_ is not None and returnComplete is False:
			indkeepw = abs(omega) <= self.OmegaMax_
		else:
			indkeepw = NP.arange(len(omega))

		self.kx_    = kx[indkeepx]
		self.ky_    = ky[indkeepy]
		self.omega_ = omega[indkeepw]


		return self.kx_,self.ky_,self.omega_

	def R_omega_k(self,omega,omega_nk,Gamma_nk,A_nk = 1):
		# Compute the resonant response function (eq 18 Woodard 2006) or (eq 2 Hanasoge 2018)
		# Also called the "Lorentzian"

		# return A_nk/( (omega_nk - 1.j*Gamma_nk/2)**2 - omega**2 ) #/ abs(1/( (omega_nk - 1.j*Gamma_nk/2)**2 - omega_nk**2 ))
		return A_nk/( omega_nk**2 - 1.j*omega*Gamma_nk/2 - omega**2 )


	def ref_fit_params(self,absk,radial_order,usePrecompute=True,rdv_module='c',Nsamples=100):
		# if radial_order == 0:
			# return fmode_dispersion_fit_params(absk)
		if usePrecompute:
			try:
				self.RADIAL_
			except:
				self.RADIAL_ = 999

			if self.RADIAL_ != radial_order:
				if self.rdvFits_:
					self.ABSK_,self.OMEGA_,self.GAMMA_,self.AMP_ = NP.genfromtxt(os.getcwd() + '/data/modeData_rdvFit%s/modeData_n%i.dat' % (rdv_module,radial_order)).T
				else:
					self.ABSK_,self.OMEGA_,self.GAMMA_,self.AMP_ = NP.genfromtxt(os.getcwd() + '/data/modeData/modeData_n%i.dat' % radial_order).T

				self.RADIAL_ = radial_order
			# ITP_OMEGA   = scipy.interpolate.interp2d(ABSK,  ,kind='linear',bounds_error=False,fill_value=NP.nan)

			if hasattr(absk,'__len__'):
				omega = [];gamma = [];amp = []
				for ii in range(len(absk)):
					ind = NP.argmin(abs(self.ABSK_ - absk[ii]))
					omega.append(self.OMEGA_[ind])
					gamma.append(self.GAMMA_[ind])
					amp.append(self.AMP_[ind])
			else:
				ind = NP.argmin(abs(self.ABSK_ - absk))
				omega = self.OMEGA_[ind]
				gamma = self.GAMMA_[ind]
				amp   = self.AMP_[ind]

			# amp = NP.nan_to_num(absk/absk)

		else:

			if self.rdvFits_:
				# if 'out' not in locals():


				# plt.figure(1)
				# plt.figure(2)

				if 'KK' not in globals():
					global KK,OM,FW,AP,NN

					subP = subprocess.Popen('ls /scratch/ch3246/OBSDATA/modeCouple/rdv_fits%s_fd15/fit%s_files/*' % (rdv_module,rdv_module),\
					                              shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
					out,err  = subP.communicate()
					out = out.split()
					for ii in range(len(out)):
						out[ii] = out[ii].decode('utf-8')

					KK = [];OM = [];FW = [];AP = [];NN = []
					print(text_special('Computing mode parameter averages from %i ring tiles' % Nsamples,'y'))
					PG = progressBar(Nsamples,'serial')
					for ii in range(Nsamples):
						DATA = NP.genfromtxt(out[ii]).T
						inds = DATA[3] < 5500
						if rdv_module == 'f':
							[nn,ell,kk,nu,dnu,ux,dux,uy,duy,fit,amp,damp,bg,dbg,fwhm,dfwhm,delnu,dnu2,kbin,nfe,minfunc,rdchi] = DATA[:,inds]

						elif rdv_module == 'c':
							[nn,ell,kk,nu,dnu,ux,dux,uy,duy,fit,l_guess,bfgs,f,fm,amp,damp,c,D_c,fwhm,dfwhm,B1,D_B1,B2,D_B2,A1,D_A1,A2,D_A2,A3,D_A3,p,D_p,w1,D_w1,S,D_S] = DATA[:,inds]

						kk = kk*1e-6
						omega = nu*1e-6*2*NP.pi
						fwhm  = fwhm*1e-6*2*NP.pi

						KK= NP.concatenate([KK,kk],axis=0);
						OM= NP.concatenate([OM,omega],axis=0)
						FW= NP.concatenate([FW,fwhm],axis=0)
						AP= NP.concatenate([AP,NP.exp(amp)],axis=0)
						NN= NP.concatenate([NN,nn],axis=0)

						# plt.figure(1)
						# plt.plot(kk*RSUN,nu,'.r')
						# plt.axhline(y=3000)
						# plt.axvline(x=1000)

						# plt.figure(2)
						# plt.plot(nu,fwhm,'.b')

						PG.update()
					del PG

				# plt.figure(1)
				# plt.figure(2)
				# plt.figure(3)
				Ogrid = [];Fgrid = [];Agrid = []

				inds = NN == radial_order
				Ogrid = [];Fgrid = [];Agrid = []
				if rdv_module == 'f':
					Kgrid = NP.unique(KK[inds])
					
					for kk in Kgrid:
						inds2 = KK == kk
						Ogrid.append(NP.mean(OM[inds*inds2]))
						Fgrid.append(NP.mean(FW[inds*inds2]))
						Agrid.append(NP.mean(AP[inds*inds2]))
				elif rdv_module == 'c':
					Kgrid = histogram(KK[inds],bins=50)[1]
					Kgrid = Kgrid[Kgrid*RSUN>150]
					# plt.close()
					for nbin in range(len(Kgrid)-1):
						inds_bin = (KK[inds] > Kgrid[nbin])*(KK[inds] < Kgrid[nbin+1])
						if sum(inds_bin) == 0:
							Ogrid.append(NP.nan)
							Fgrid.append(NP.nan)
							Agrid.append(NP.nan)
						else:				
							Ogrid.append(NP.mean(OM[inds][inds_bin]))
							Fgrid.append(NP.mean(FW[inds][inds_bin]))
							Agrid.append(NP.mean(AP[inds][inds_bin]))	
					Kgrid = Kgrid[:-1] + NP.diff(Kgrid)[0]/2
					Ogrid = NP.array(Ogrid)
					Fgrid = NP.array(Fgrid)
					Agrid = NP.array(Agrid)		

					# inds2 = NP.argsort()
					# Kgrid = NP.linspace(150/RSUN,KK[inds].max(),300)
					# KK2 = KK[inds]
					# inds2 = NP.argsort(KK2)
					indsO = ~NP.isnan(Ogrid);indsF = ~NP.isnan(Fgrid);indsA = ~NP.isnan(Agrid)
					# if sum(indsO) < 5 or sum(indsF) < 5 or sum(indsA) < 5:
						# return NP.ones((3,len(absk))).squeeze()*NP.nan
					Ogrid = scipy.interpolate.griddata(Kgrid[indsO],Ogrid[indsO],Kgrid,'linear')
					Fgrid = scipy.interpolate.griddata(Kgrid[indsF],Fgrid[indsF],Kgrid,'linear')
					Agrid = scipy.interpolate.griddata(Kgrid[indsA],Agrid[indsA],Kgrid,'linear')

				ITP_w = scipy.interpolate.interp1d(Kgrid*RSUN,Ogrid,kind='linear',fill_value=NP.nan,bounds_error=False)
				ITP_g = scipy.interpolate.interp1d(Kgrid*RSUN,NP.log(Fgrid),kind='linear',fill_value=NP.nan,bounds_error=False)
				ITP_a = scipy.interpolate.interp1d(Kgrid*RSUN,Agrid,kind='linear',fill_value=NP.nan,bounds_error=False)

			else:

				DATA = NP.genfromtxt('/home/ch3246/mps_montjoie/data/Observations/KORZENNIK_DATA/multiplets-mdi-2001.dat')

				inds = DATA[:,0] == radial_order

				DATA = DATA[inds][:,[1,2,8,16]]
				DATA = NP.sort(DATA,axis=0)
				
				inds = (DATA[:,2] > 0)*(DATA[:,3] > 0)

				DATA = DATA[inds]

				ITP_w = scipy.interpolate.interp1d(DATA[:,0],DATA[:,1]/1e6 * 2*NP.pi,kind='linear',fill_value='extrapolate')
				ITP_g = scipy.interpolate.interp1d(DATA[:,0],NP.log(DATA[:,2]/1e6 * 2*NP.pi),kind='linear',fill_value=(NP.log(DATA[:,2]/1e6 * 2*NP.pi).min(),NP.log(DATA[:,2]/1e6 * 2*NP.pi).max()),bounds_error=False)
				ITP_a = scipy.interpolate.interp1d(DATA[:,0],DATA[:,3],kind='linear',fill_value='extrapolate')

			omega = ITP_w(absk*RSUN)
			gamma = NP.exp(ITP_g(absk*RSUN))
			amp   = ITP_a(absk*RSUN)
			# amp   = NP.nan_to_num(absk/absk)

		return NP.array([omega,gamma,amp])

	def precompute_ref_fit_params(self,radial_order,rdv_module='c',Nsamples=100):
		try:
			self.kx_
		except:
			self.computeFreq()
		kx,ky = NP.meshgrid(self.kx_,self.ky_,indexing='ij')
		abs_k = NP.unique(NP.sqrt(kx**2+ky**2))
		OMEGA,GAMMA,AMP = self.ref_fit_params(abs_k,radial_order,usePrecompute=False,rdv_module=rdv_module,Nsamples=Nsamples)

		AMP = NP.where(AMP < 0,0,AMP)

		if not self.rdvFits_:
			mkdir_p(os.getcwd() + '/data/modeData/')
			NP.savetxt(os.getcwd() + '/data/modeData/modeData_n%i.dat' % radial_order,NP.array([abs_k,OMEGA,GAMMA,AMP]).T)
		else:
			mkdir_p(os.getcwd() + '/data/modeData_rdvFit%s/' % rdv_module)
			NP.savetxt(os.getcwd() + '/data/modeData_rdvFit%s/modeData_n%i.dat' % (rdv_module,radial_order),NP.array([abs_k,OMEGA,GAMMA,AMP]).T)




	def fmode_dispersion_fit_params(self,absk):
	    nu_fmode = 1e6*NP.sqrt(274*absk)/(2*NP.pi)## In μHz
	    # nu_fmode = confmode * (absk) ## In μHz

	    gamma = 100 * (nu_fmode/3000.0)**4.4 ## From GIZON & BIRCH 2002, Eq. 57
	    amp = 1.0
	    return NP.array([nu_fmode*1e-6*2*NP.pi,gamma*1e-6*2*NP.pi,amp],dtype=object)


	def mask_cube(self,radial_order,omega='all',Ngamma=1,nu_min = 0.0025,nu_max = 0.0045,kRmin=0,kRmax=1e4,storeInInstance=False):
		kx_grid,ky_grid,omega_tmp = self.computeFreq()
		ind_wmin = NP.argmin(abs(omega_tmp - nu_min*2*NP.pi))
		ind_wmax = NP.argmin(abs(omega_tmp - nu_max*2*NP.pi))

		if omega != 'all':
			kx_grid,ky_grid = NP.meshgrid(kx_grid,ky_grid,indexing='ij')
		else:
			kx_grid,ky_grid,omega_tmp2 = NP.meshgrid(kx_grid,ky_grid,omega_tmp[ind_wmin:ind_wmax],indexing='ij')

		mask_cube = NP.zeros(kx_grid.shape).ravel()

		abs_k  = NP.sqrt(kx_grid**2 + ky_grid**2)
		mask_k = NP.where((abs_k < kRmin/RSUN) + (abs_k > kRmax/RSUN),0,1)

		abs_k_shape = abs_k.shape
		omega_grid,gamma_grid = self.ref_fit_params(abs_k.ravel(),radial_order,usePrecompute=True)[:2]

		omega_grid = omega_grid.reshape(abs_k_shape)
		gamma_grid = gamma_grid.reshape(abs_k_shape)

		# return omega_grid,gamma_grid
		gamma_grid = NP.where((omega_grid > nu_max*2*NP.pi)+(omega_grid < nu_min*2*NP.pi),-1,gamma_grid)


		# return R_omega_k(omega_grid,omega,gamma_grid) * NP.where(abs(omega_grid-omega) < Ngamma*gamma_grid,1,0)
		if omega == 'all':
			# mask,omega_out = [NP.where(abs(omega_grid-omega_tmp2) <= Ngamma*gamma_grid,1,0),omega_tmp[ind_wmin:ind_wmax]]
			inds = NP.argmin(abs(omega_grid - omega_tmp2),axis=-1).ravel()
			# print(omega_grid.shape,omega_tmp2.shape,inds.shape)
			mask = NP.zeros(omega_tmp2.shape).reshape(-1,omega_tmp2.shape[-1])
			# print(mask.shape,len(inds))
			for ii in range(len(inds)):
				mask[ii,inds[ii]] = 1
			mask = mask.reshape(omega_tmp2.shape)
			omega_out = omega_tmp[ind_wmin:ind_wmax]
			mask[:,:,0] = 0; mask[:,:,-1] = 0 # kill if inds ourside numin and numax
		else:
			mask,omega_out = [NP.where(abs(omega_grid-omega) <= Ngamma*gamma_grid,1,0),omega]

		mask *= mask_k

		if storeInInstance:
			self.mask_ = mask
			self.maskOmega_ = omega_out

		return mask,omega_out



	def compute_bcoeff_serial(self,radial_order,kx_ind,ky_ind,qx_ind_arr,qy_ind_arr,sigma_ind_arr,amp_array,nu_min=0.0025,nu_max=0.0045,\
								absq_range=NP.array([0,1e9])/RSUN,VERBOSE=False,returnNoise=False,\
								windfactor = 2,modeSpacingFactor = None,Measurement='flow',TEST=0):
		# Compute the obervable B coefficient
		# B^sigma_k_q = sum_w[H^w_k_kp_sigma* phi^w_k* phi^w+sigma_k+q] / sum_w[|H^w_k_kp_sigma|^2]
		# where H^w_k_kp_sigma = -2w(N_k |R^w_k|^2 R^w+sigma_kp  + N_kp |R^w+sigma_kp|^2 R^w_k*)

		if Measurement.upper() not in ['FLOW','SOUNDSPEED']:
			raise Exception('Measurement should be one of: [FLOW, SOUNDSPEED]\n Currently is: %s' % Measurement.upper())

		# ind : index of kx_ind_arr to use to compute kx
		# radial_order: radial order n of the mode
		# kx_ind_arr: 1-d array containing the indices of the kx_grid for k'
		# ky_ind_arr: 1-d array containing the indices of the ky_grid for k'
		# nu_min [Hz]: minimum frequency for resonant frequencies

		amp_array = NP.array(amp_array)

		if hasattr(radial_order,'__len__'):
			if len(radial_order) > 2:
				raise Exception("Too many n given: Can only compute [n,n'] coupling")
			radial_orderp = radial_order[1]
			radial_order  = radial_order[0]
		else:
			radial_orderp = radial_order

		NP.seterr(divide='ignore', invalid='ignore')

		try:
			# self.phi_kw_
			phi_kw
		except:
			raise Exception('Data cube phi_kw is not defined: please run computeFFT')

		# if modeSpacingFactor is None:
			# modeSpacingFactor = 2*windfactor


		self.computeFreq()
		absk   = NP.sqrt(self.kx_[kx_ind]**2+self.ky_[ky_ind]**2)

		bcoeff = NP.zeros((len(qx_ind_arr),len(qy_ind_arr),len(sigma_ind_arr)),complex)


		if VERBOSE:
			print(text_special('kx  = %1.4e m^(-1)' % self.kx_[kx_ind],'y'))
			print(text_special('ky  = %1.4e m^(-1)' % self.ky_[ky_ind],'y'))
			print(text_special('ell = %1.4f' % (absk*RSUN),'y'))

		dkx    = NP.diff(self.kx_)[0]
		dky    = NP.diff(self.ky_)[0]
		domega = NP.diff(self.omega_)[0]


		wac_ind = (abs(self.omega_) > nu_min*2*NP.pi)*(abs(self.omega_) < nu_max*2*NP.pi)


		MODE_w_nk,MODE_gamma_nk,N_nk = self.ref_fit_params(absk,radial_order,usePrecompute=True)

		# Determine the N_nk to be consistent with obs
		if amp_array.ndim == 3:
			# for n n' coupling
			N_nk = amp_array[0][kx_ind,ky_ind]
		else:
			# for n n coupling
			N_nk = amp_array[kx_ind,ky_ind]
		indl = NP.argmin(abs(self.omega_ - (MODE_w_nk-5*MODE_gamma_nk)))
		indr = NP.argmin(abs(self.omega_ - (MODE_w_nk+5*MODE_gamma_nk)))
		# Loren = self.R_omega_k(self.omega_[indl:indr],MODE_w_nk,MODE_gamma_nk)
		# POW  = phi_kw[indl:indr,ky_ind,kx_ind]
		# POW  = NP.conj(POW) * POW
		# N_nk = NP.real(NP.sum(POW)/NP.sum(abs(Loren*NP.conj(Loren))) )

		if TEST ==3:
			plt.figure()
			plt.plot(self.omega_[indl:indr]/(2*NP.pi)*1e6,abs(phi_kw[indl:indr,ky_ind,kx_ind])**2)
			Loren = self.R_omega_k(self.omega_[indl:indr],MODE_w_nk,MODE_gamma_nk)
			plt.plot(self.omega_[indl:indr]/(2*NP.pi)*1e6,abs(Loren*NP.conj(Loren))*N_nk)
			return 0
		# if reference model fails
		if NP.isnan(MODE_w_nk) or (MODE_w_nk < nu_min*2*NP.pi) or (MODE_w_nk > nu_max*2*NP.pi):
			if VERBOSE:
				print('MODE not found')
			# return coo_matrix(bcoeff.reshape(-1))
			return bcoeff

		# windfactor is the number of linewidths we take when summing in omega
		# omega_inds are the indices in omega in which we are within the omega window
		# min value of windfactor must be 2 to encapsulate linewidth. If not could miss part of line width due to the boolen func below
		# windfactor = 1.
		# windfactor = max([windfactor,2])

		MSF_m = [ -0.01,-0.01,-0.01,-0.012,-0.014,-0.018,-0.012,-0.01,0,0]
		MSF_c = [  13.5, 14.5,13.5 , 11.85, 11.15, 11.41, 8.47 , 7   ,2,2]

		MSF_m_np1 = [None,-0.005,-0.01,-0.01,0. ,0.,0.,0.,0.]
		MSF_c_np1 = [None,6.5   ,11.5 ,11   ,4.5,4.,3.,2.,2.]
		# MSF_m_np1 = [None,-0.005,-0.01,-0.01,0.,0.,None]
		# MSF_c_np1 = [None,6.5,11.5,11,3.5,3,None]

		MSF_m_np2 = [None,None,0.,0.,0.,0.,0.,0.]
		MSF_c_np2 = [None,None,11,9.,3.,3.,3.,2.]

		if Measurement.upper() == 'SOUNDSPEED':
			MSF_m = [ -0.01,-0.01,-0.01,-0.01 ,-0.01,-0.01,-0.01,-0.01,0,0]
			MSF_c = [13.5  ,14.  ,12.5 ,12.   ,11.  ,10.5 ,10.  ,9.5  ,2,2]			

			MSF_m_np1 = [None,-0.01,-0.01,0.   ,0. ,0.,0 ,0.,0.]
			MSF_c_np1 = [None,13   ,11   ,3.   ,2. ,2 ,2.,2.,2.]

			MSF_m_np2 = [None,None,0.,0.,0.,0.,0.,0.]
			MSF_c_np2 = [None,None,11,9.,2.,2.,2.,2.]
		if modeSpacingFactor is None:
			if [MSF_m,MSF_m_np1,MSF_c_np2][int(radial_orderp - radial_order)][radial_order] is None:
				raise Exception('Fit for Delta not done for p%i-p%i' % (radial_order,radial_orderp))
			modeSpacingFactor = [MSF_m,MSF_m_np1,MSF_m_np2][int(radial_orderp - radial_order)][radial_order]*(absk*RSUN) + [MSF_c,MSF_c_np1,MSF_c_np2][int(radial_orderp - radial_order)][radial_order]
			if radial_order == 2 and radial_orderp ==2 and absk*RSUN > 750:
				modeSpacingFactor = modeSpacingFactor - 0.5
			# if radial_order > 2:
				# modeSpacingFactor = modeSpacingFactor-2
		# print(modeSpacingFactor)


		omega_inds = NP.arange(len(self.omega_))[(abs(self.omega_) < MODE_w_nk + windfactor*MODE_gamma_nk/2)*(abs(self.omega_) > MODE_w_nk - windfactor*MODE_gamma_nk/2)*wac_ind]

		# if TEST == 1:
			# return omega_inds

		# if mode is outside the frequency limits nu_min and nu_max, return values
		if len(omega_inds) == 0:
			return bcoeff#coo_matrix(bcoeff.reshape(-1))

		if VERBOSE:
			PG = progressBar(len(qx_ind_arr),'serial')
		fails = NP.zeros(5);success = 0
		qx_out_ind = 0
		for qx_ind in qx_ind_arr:
			qy_out_ind = 0
			for qy_ind in qy_ind_arr:
				sigma_out_ind = 0
				for sigma_ind in sigma_ind_arr:
					# in case the k+q is outside of the cube continue
					if (kx_ind+qx_ind < 0) or (kx_ind+qx_ind > len(self.kx_)) or (ky_ind+qy_ind < 0) or (ky_ind+qy_ind > len(self.ky_)):
						sigma_out_ind += 1
						fails[0] +=1
						continue;

					kxp_ind = kx_ind + qx_ind
					kyp_ind = ky_ind + qy_ind

					if kxp_ind > (len(self.kx_)-1) or kyp_ind > (len(self.ky_)-1):
						fails[1] +=1
						# If reaching edge of cube
						continue;

					qx = self.kx_[kxp_ind] - self.kx_[kx_ind]
					qy = self.ky_[kyp_ind] - self.ky_[ky_ind]
					absq   = NP.sqrt(qx**2 + qy**2)

					kxp = self.kx_[kxp_ind]
					kyp = self.ky_[kyp_ind]
					abskp = NP.sqrt(kxp**2 + kyp**2)



					if absq > absq_range[1] or absq < absq_range[0] :
						sigma_out_ind += 1
						fails[2]+=1
						continue;

					# Apply checks here (e.g. outside wave length)

					# compute the mode frequency of |kp'|
					MODE_w_nkp,MODE_gamma_nkp,N_nkp = self.ref_fit_params(abskp,radial_orderp,usePrecompute=True)

					# Determine the N_nkp to be consistent with obs
					# indl  = NP.argmin(abs(self.omega_ - (MODE_w_nkp-5*MODE_gamma_nkp)))
					# indr  = NP.argmin(abs(self.omega_ - (MODE_w_nkp+5*MODE_gamma_nkp)))
					# Loren = self.R_omega_k(self.omega_[indl:indr],MODE_w_nkp,MODE_gamma_nkp)
					# POW   = phi_kw[indl:indr,kyp_ind,kxp_ind]
					# POW   = NP.conj(POW) * POW
					# N_nkp = NP.real(NP.sum(POW)/NP.sum(abs(Loren*NP.conj(Loren))) )
					# N_nkp = NP.sqrt( NP.real(NP.sum(POW)/NP.sum(abs(Loren*NP.conj(Loren))) ))
					# N_nkp = amp_array[kxp_ind,kyp_ind]
					if amp_array.ndim == 3:
						# for n n' coupling
						N_nkp = amp_array[1][kxp_ind,kyp_ind]
					else:
						# for n n coupling
						N_nkp = amp_array[kxp_ind,kyp_ind]

					# if N_nkp == 0:
						# # kp is not in the mask
						# continue; 


					if NP.isnan(MODE_w_nkp) or (MODE_w_nkp < nu_min*2*NP.pi) or (MODE_w_nkp > nu_max*2*NP.pi):
						sigma_out_ind += 1
						fails[3] += 1
						continue

					# if abs(MODE_w_nk - (MODE_w_nkp+sigma_ind*NP.diff(self.omega_)[0])) > modeSpacingFactor*MODE_gamma_nk:
					if abs(MODE_w_nk - MODE_w_nkp) > modeSpacingFactor*MODE_gamma_nk:
						sigma_out_ind += 1
						fails[4] += 1
						continue

					# if abs(MODE_w_nk - (MODE_w_nkp+sigma_ind*NP.diff(self.omega_)[0])) < 0.5*MODE_gamma_nk:
					# 	sigma_out_ind += 1
					# 	fails[4] += 1
					# 	continue
					# check if refernce model worked
					# if NP.isnan(MODE_w_nkp):
						# continue

					# find frequency window 
					omega_indsp = NP.arange(len(self.omega_))[(abs(self.omega_) < MODE_w_nkp + windfactor*MODE_gamma_nkp/2)*(abs(self.omega_) > MODE_w_nkp - windfactor*MODE_gamma_nkp/2)*wac_ind]

					# if len(omega_indsp) == 0:
						# continue


					# determine the frequencies to add up
					omega_inds_all = NP.union1d(omega_inds,omega_indsp)
					if TEST==2:
						return omega_inds_all
					omega_all = self.omega_[omega_inds_all]
					omega_all_sigma = self.omega_[omega_inds_all+sigma_ind]



					# compute phi_k and phi_k'
					# phi_k  = self.phi_kw_[omega_inds_all,kx_ind,ky_ind]
					# phi_kp = self.phi_kw_[omega_inds_all,kxp_ind,kyp_ind]
					phi_k  = phi_kw[omega_inds_all,ky_ind,kx_ind] # INDEXING IS FLIPPED DUE TO HOW FITS IS STORED [t,y,x]
					phi_kp = phi_kw[omega_inds_all+sigma_ind,kyp_ind,kxp_ind]

					# compute the R_wk and H_kkp
					R_wk  = self.R_omega_k(omega_all,MODE_w_nk,MODE_gamma_nk)
					R_wkp = self.R_omega_k(omega_all_sigma,MODE_w_nkp,MODE_gamma_nkp)

					H_kkp = (N_nk*abs(R_wk)**2*R_wkp + \
											N_nkp*abs(R_wkp)**2*NP.conj(R_wk))
					if Measurement.upper() == 'FLOW':
						H_kkp = H_kkp*(2.j)*omega_all

					if TEST ==4:
						plt.figure()
						plt.plot(self.omega_/(2*NP.pi)*1e6,abs(phi_kw[:,ky_ind,kx_ind])**2,label=r'$\phi(k,\omega)$')
						plt.plot(self.omega_/(2*NP.pi)*1e6,abs(phi_kw[:,kyp_ind,kxp_ind])**2,label=r"$\phi(k',\omega)$")
						plt.plot(self.omega_[omega_inds_all]/(2*NP.pi)*1e6,abs(phi_kw[:,kyp_ind,kxp_ind][omega_inds_all])**2,label=r'$\phi(k,\omega)$')
						plt.plot(self.omega_[omega_inds_all]/(2*NP.pi)*1e6,abs(phi_kw[:,ky_ind,kx_ind][omega_inds_all])**2,label=r"$\phi(k',\omega)$")
						# plt.plot(self/(2*NP.pi)*1e6,abs(R_wk)*N_nkp,'--')
						plt.plot(omega_all/(2*NP.pi)*1e6,abs(R_wk)**2*N_nk)
						plt.plot(omega_all/(2*NP.pi)*1e6,abs(R_wkp)**2*N_nkp)
						plt.legend()
						print('TEST 4 complete')
						return 0#[omega_all/(2*NP.pi)*1e6,Loren,R_wk]


					# compute the bcoeff
					if returnNoise:
						bcoeff[qx_out_ind,qy_out_ind,sigma_out_ind]  =  NP.sum(abs(H_kkp)**2 * N_nk*abs(R_wk)**2 * N_nkp*abs(R_wkp)**2)
						# if (abs(qx_ind)+abs(sigma_ind)+abs(qy_ind)) == 0:
							# bcoeff[qx_out_ind,qy_out_ind,sigma_out_ind]  += NP.sum(abs(H_kkp)**2 * N_nk*abs(R_wk)**2 * N_nkp*abs(R_wkp)**2)
						bcoeff[qx_out_ind,qy_out_ind,sigma_out_ind] /=  NP.sum(abs(H_kkp)**2)**2
						if TEST == 1:
							abort
					# elif returnNoise == 'obs':
						# bcoeff[qx_out_ind,qy_out_ind,sigma_out_ind]  =  NP.sum(abs(H_kkp)**2 * abs(phi_k)**2 * abs(phi_kp)**2)
						# if (abs(qx_ind)+abs(sigma_ind)+abs(qy_ind)) == 0:
							# bcoeff[qx_out_ind,qy_out_ind,sigma_out_ind]  += NP.sum(abs(H_kkp)**2 * N_nk*abs(R_wk)**2 * N_nkp*abs(R_wkp)**2)
						# bcoeff[qx_out_ind,qy_out_ind,sigma_out_ind] /=  NP.sum(abs(H_kkp)**2)**2
					else:
						bcoeff[qx_out_ind,qy_out_ind,sigma_out_ind] = NP.sum(NP.conj(H_kkp) * NP.conj(phi_k) * phi_kp) / NP.sum(abs(H_kkp)**2)

					sigma_out_ind += 1
					success += 1

				qy_out_ind += 1
			qx_out_ind += 1


			if VERBOSE:
				PG.update()
		if VERBOSE:
			del PG
			print('Fail calls: ',fails)
			print('Successes: ',success)
			# return [fails,success]

		# return coo_matrix(bcoeff.reshape(-1))
		return bcoeff

	def compute_bcoeffs_parallel(self,radial_order,k_mask,qx_ind_arr,qy_ind_arr,sigma_ind_arr,amp_array,nu_min=0.0025,nu_max=0.0045,\
								absq_range=NP.array([0,1e9])/RSUN,VERBOSE=False,nbProc=8,reorg_k=True,\
								windfactor=2,modeSpacingFactor=None,returnNoise=False,rtype=None,\
								saveDir = None,iFile=None,Measurement='flow'):

		if Measurement.upper() not in ['FLOW','SOUNDSPEED']:
			raise Exception('Measurement should be one of: [FLOW, SOUNDSPEED]\n Currently is: %s' % Measurement.upper())

		# check amp_array is loaded in
		if NP.array(amp_array).shape != k_mask.shape and NP.array(amp_array)[0].shape != k_mask.shape:
			raise Exception('Amp array must be loaded in and of the same size as mask')
		if saveDir is not None and iFile is None:
			raise Exception('saveDir is specified, please specify the Cube number: iFile')

		if hasattr(radial_order,'__len__'):
			if len(radial_order) > 2:
				raise Exception("Too many n given: Can only compute [n,n'] coupling")
			radial_orderp = radial_order[1]
			radial_order  = radial_order[0]
		else:
			radial_orderp = radial_order

		# Compute the kx and ky indices from the input mask
		kx_inds,ky_inds = NP.where(k_mask==1)

		# Compute the kernels for each kx and ky in the masked area
		res_tmp = reduce(parallelize_classObject,(self.compute_bcoeff_serial,[radial_order,radial_orderp],kx_inds,ky_inds,qx_ind_arr,qy_ind_arr,sigma_ind_arr,amp_array,nu_min,nu_max,absq_range,False,returnNoise,windfactor,modeSpacingFactor,Measurement),len(kx_inds),nbProc,type=rtype,progressBar=VERBOSE)

		if returnNoise:
			res_tmp  = solarFFT.testRealFFT(res_tmp)
			
		if not reorg_k:
			kx,ky,omega = self.computeFreq()
			if saveDir is not None:
				mkdir_p(saveDir)
				for ii in range(len(qx_ind_arr)):
					for jj in range(len(qy_ind_arr)):
						outDir = saveDir + '/Bcoeffs/qx_%i/qy_%i/' % (qx_ind_arr[ii],qy_ind_arr[jj])
						mkdir_p(outDir)
						NP.savez_compressed(outDir + '/%s_n%i_np%i.npz' % (['Bcoeff_iFile%i' % iFile,'NoiseModel'][int(returnNoise)],radial_order,radial_orderp),\
											Bcoeff = res_tmp[ii,jj],\
											QX = qx_ind_arr,QY = qy_ind_arr,SIGMA=sigma_ind_arr,\
											kx = kx[kx_inds],ky=ky[ky_inds],\
											mask = k_mask,\
											dims = ['SIGMA','k'])
				return kx[kx_inds],ky[ky_inds],'SAVED: %s' % saveDir


			if VERBOSE:
				return kx[kx_inds],ky[ky_inds],res_tmp
			else:
				return kx[kx_inds],ky[ky_inds],res_tmp
		res = NP.zeros(k_mask.shape + res_tmp.shape[:-1],complex)*NP.nan
		for ii in range(len(kx_inds)):
			res[kx_inds[ii],ky_inds[ii]] = res_tmp[...,ii]

		return res

	def compute_kernels(self,radial_order,omega,kx_ind,ky_ind,qx_ind_arr,qy_ind_arr,BasisClass,kernelType,\
						absq_range=NP.array([0,1e9])/RSUN,multiplyH=False,scaleFields=2):

		if not hasattr(kernelType,'__len__'):
			kernelType = [kernelType]
		if not isinstance(kernelType[0],str):
			raise Exception('kernelType should be string or list of strings')

		kernelTypeInds = NP.zeros(len(kernelType),dtype=int)*NP.nan
		for ii in range(len(kernelType)):
			kernelTypeInds[ii] = ['POLOIDAL','TOROIDAL','SOUNDSPEED','LORENZ','DENSITY','UX','UY','UZ'].index(kernelType[ii].upper())
		kernelTypeInds = kernelTypeInds.astype(int)

		# Kernels are independant of sigma
		sigma_ind_arr = NP.array([0])

		t1 = time.time()

		if hasattr(radial_order,'__len__'):
			if len(radial_order) != 2:
				raise Exception("Too many n given: Can only compute [n,n'] coupling")
			radial_orderp = radial_order[1]
			radial_order  = radial_order[0]

			Exception("Need to code in the eigenfunctions for n,n'")
		else:
			radial_orderp = radial_order


		kx,ky,omega_grid = self.computeFreq()


		# Load in the globals for parallel computing
		if 'ddfj' not in globals():
			print('Initializing Eigenfunctions')
			global z,rho,Hrho,drhodz,cs,eig_k,eig_w,Xi_h,Xi_z,dzXi_h,dzXi_z,ITP_Xi_h,ITP_Xi_z,ITP_dzXi_h,ITP_dzXi_z,Basis1D,fj,dfj,ddfj,rhop
			global Xi_hp,Xi_zp,dzXi_hp,dzXi_zp,ITP_Xi_hp,ITP_Xi_zp,ITP_dzXi_hp,ITP_dzXi_zp

			# # load the background and eigenfunctions
			# with fits.open('/home/ch3246/mps_montjoie/CompHelioWork/ModeCoupling/Cartesian/eigs/model.fits') as hdu:
				# z,rho = hdu[0].data[[0,3]]
			# with fits.open('/home/ch3246/mps_montjoie/CompHelioWork/ModeCoupling/Cartesian/eigs/k.fits') as hdu:
				# eig_k = hdu[0].data.squeeze()
			# with fits.open('/home/ch3246/mps_montjoie/CompHelioWork/ModeCoupling/Cartesian/eigs/omega%02d.fits' % radial_order) as hdu:
			# 	eig_w = hdu[0].data.squeeze()
			# with fits.open('/home/ch3246/mps_montjoie/CompHelioWork/ModeCoupling/Cartesian/eigs/eig%02d.fits' % radial_order) as hdu:
			# 	Xi_h,Xi_z = hdu[0].data[[0,1]]
			# with fits.open('/home/ch3246/mps_montjoie/CompHelioWork/ModeCoupling/Cartesian/eigs/omega%02d.fits' % radial_orderp) as hdu:
			# 	eig_wp = hdu[0].data.squeeze()
			# with fits.open('/home/ch3246/mps_montjoie/CompHelioWork/ModeCoupling/Cartesian/eigs/eig%02d.fits' % radial_orderp) as hdu:
			# 	Xi_hp,Xi_zp = hdu[0].data[[0,1]]
			# # convert to SI
			# eig_k *= 1e-6
			# z     *= 1e6
			# rho   *= 1e3
			# Xi_h  *= NP.sqrt(1/10.)
			# Xi_z  *= NP.sqrt(1/10.)
			# Xi_hp *= NP.sqrt(1/10.)
			# Xi_zp *= NP.sqrt(1/10.)

			with NP.load('/scratch/ch3246/OBSDATA/gyreresult/eigenfunctions_combined/eigs%02d.npz' % radial_order) as DICT:
				z     = DICT['z']
				rho   = DICT['rho']
				cs    = DICT['cs']
				eig_k = DICT['eig_k']
				Xi_h  = DICT['Xi_h']
				Xi_z  = DICT['Xi_z']

			with NP.load('/scratch/ch3246/OBSDATA/gyreresult/eigenfunctions_combined/eigs%02d.npz' % radial_orderp) as DICT:
				eig_kp = DICT['eig_k']
				Xi_hp = DICT['Xi_h']
				Xi_zp = DICT['Xi_z']


			# if not NP.prod(BasisClass.x_ == z):
				# print(text_special('BasisClass.x_ must match z \n return z and Basis1D.x_ now','r',True,True))
				# return NP.array([z,BasisClass.x_])

			# Compute deivatives of Xi
			dz = FDM_Compact(z)
			if BasisClass.__class__.__name__ == 'constantBasis1D':
				z = z[::BasisClass.subsample_]
				dz = FDM_Compact(z)
			rho = NP.exp(BasisClass.reconstructFromBasis(BasisClass.projectOnBasis(NP.log(rho))))

			drhodz = dz.Compute_derivative(rho)
			Hrho  = -1./(dz.Compute_derivative(NP.log(rho))).real

			# dzXi_h = dz.Compute_derivative(Xi_h)
			# dzXi_z = dz.Compute_derivative(Xi_z)
			# dzXi_hp = dz.Compute_derivative(Xi_hp)
			# dzXi_zp = dz.Compute_derivative(Xi_zp)
			dzXi_h = BasisClass.reconstructFromBasis(BasisClass.projectOnBasis(Xi_h,axis=-1),derivative=1,axis=1)
			dzXi_z = BasisClass.reconstructFromBasis(BasisClass.projectOnBasis(Xi_z,axis=-1),derivative=1,axis=1)
			dzXi_hp = BasisClass.reconstructFromBasis(BasisClass.projectOnBasis(Xi_hp,axis=-1),derivative=1,axis=1)
			dzXi_zp = BasisClass.reconstructFromBasis(BasisClass.projectOnBasis(Xi_zp,axis=-1),derivative=1,axis=1)
			if BasisClass.__class__.__name__ == 'constantBasis1D':
				Xi_h = BasisClass.reconstructFromBasis(BasisClass.projectOnBasis(Xi_h,axis=-1),axis=1)
				Xi_z = BasisClass.reconstructFromBasis(BasisClass.projectOnBasis(Xi_z,axis=-1),axis=1)
				Xi_hp = BasisClass.reconstructFromBasis(BasisClass.projectOnBasis(Xi_hp,axis=-1),axis=1)
				Xi_zp = BasisClass.reconstructFromBasis(BasisClass.projectOnBasis(Xi_zp,axis=-1),axis=1)
			# return z,Xi_z,dzXi_z
			# inteprolate the eigenfunctions in z and k
			ITP_Xi_h   = scipy.interpolate.interp2d(eig_k,z,Xi_h.T  ,kind='linear',bounds_error=False,fill_value=NP.nan)
			ITP_Xi_z   = scipy.interpolate.interp2d(eig_k,z,Xi_z.T  ,kind='linear',bounds_error=False,fill_value=NP.nan)
			ITP_dzXi_h = scipy.interpolate.interp2d(eig_k,z,dzXi_h.T,kind='linear',bounds_error=False,fill_value=NP.nan)
			ITP_dzXi_z = scipy.interpolate.interp2d(eig_k,z,dzXi_z.T,kind='linear',bounds_error=False,fill_value=NP.nan)

			ITP_Xi_hp   = scipy.interpolate.interp2d(eig_kp,z,Xi_hp.T  ,kind='linear',bounds_error=False,fill_value=NP.nan)
			ITP_Xi_zp   = scipy.interpolate.interp2d(eig_kp,z,Xi_zp.T  ,kind='linear',bounds_error=False,fill_value=NP.nan)
			ITP_dzXi_hp = scipy.interpolate.interp2d(eig_kp,z,dzXi_hp.T,kind='linear',bounds_error=False,fill_value=NP.nan)
			ITP_dzXi_zp = scipy.interpolate.interp2d(eig_kp,z,dzXi_zp.T,kind='linear',bounds_error=False,fill_value=NP.nan)
			# compute the basis functions and the derivatives
			fj   = NP.zeros((BasisClass.nbBasisFunctions_,len(z)))
			dfj  = NP.zeros((BasisClass.nbBasisFunctions_,len(z)))
			ddfj = NP.zeros((BasisClass.nbBasisFunctions_,len(z)))
			for ii in range(BasisClass.nbBasisFunctions_):
				fj[ii]   = BasisClass(ii)
				dfj[ii]  = BasisClass(ii,derivative=1)
				ddfj[ii] = BasisClass(ii,derivative=2)

			# Compute mass matrix for integral
			BasisClass.computeMassMatrix()
			# # rhop = BasisClass.projectOnBasis(rho)
			# # Xi_hp,Xi_zp,dzXi_hp,dzXi_zp = BasisClass.projectOnBasis(NP.array([Xi_h,Xi_z,dzXi_h,dzXi_z]),axis=-1)


		t2 = time.time()
		# Initialize kernels
		kernels = NP.zeros((len(kernelType),len(qx_ind_arr),len(qy_ind_arr),len(sigma_ind_arr),BasisClass.nbBasisFunctions_),dtype='complex64')


		# compute the k, k_hat, |k| and w_nk of the mode
		k_vec = NP.array([kx[kx_ind],ky[ky_ind],0])
		k_hat = k_vec/NP.linalg.norm(k_vec)
		abs_k = NP.linalg.norm(k_vec)

		# get the observational mode details for this |k| and n
		MODE_w_nk,MODE_gamma_nk,N_nk = self.ref_fit_params(abs_k,radial_order)
		t3 = time.time()

		# determine the eigenfunctions for k
		Xi_h_k  = ITP_Xi_h (abs_k,z).squeeze()
		Xi_z_k  = ITP_Xi_z (abs_k,z).squeeze()
		dzXi_h_k = ITP_dzXi_h(abs_k,z).squeeze()
		dzXi_z_k = ITP_dzXi_z(abs_k,z).squeeze()

		# # kxp_grid,kyp_grid = NP.meshgrid(kx,ky,indexing='ij')

		# # interp_error = 0;num_calc = 0

		

		# For loop over the given qx and qy
		qx_out_ind = 0
		for qx_ind in qx_ind_arr:
			qy_out_ind = 0
			for qy_ind in qy_ind_arr:
				sigma_out_ind = 0
				for sigma_ind in sigma_ind_arr:

					# in case the k+q is outside of the cube continue
					if (kx_ind+qx_ind < 0) or (kx_ind+qx_ind > len(kx)) or (ky_ind+qy_ind < 0) or (ky_ind+qy_ind > len(ky)):
						sigma_out_ind += 1
						continue;

					# compute |q|,q,k',|k'| and k'_hat
					# q = k' - k
					kxp_ind = kx_ind + qx_ind
					kyp_ind = ky_ind + qy_ind

					if kxp_ind > (len(self.kx_)-1) or kyp_ind > (len(self.ky_)-1):
						# if qx is outside of the grid, continue
						continue;

					qx = self.kx_[kxp_ind] - self.kx_[kx_ind]
					qy = self.ky_[kyp_ind] - self.ky_[ky_ind]
					abs_q = NP.sqrt(qx**2 + qy**2)
					q_vec = NP.array([qx,qy,0])

					kxp = self.kx_[kxp_ind]
					kyp = self.ky_[kyp_ind]
					abs_kp = NP.sqrt(kxp**2 + kyp**2)
					kp_vec = NP.array([kxp,kyp,0])
					kp_hat = kp_vec/NP.linalg.norm(kp_vec)

					if NP.sum(NP.isnan(kp_hat)):
						sigma_out_ind += 1
						continue;

					# abs_q = NP.sqrt((kx[kx_ind+qx_ind]-kx[kx_ind])**2+(ky[ky_ind+qy_ind]-ky[ky_ind])**2)
					# q_vec = NP.array([kx[kx_ind+qx_ind]-kx[kx_ind],ky[ky_ind+qy_ind]-ky[ky_ind],0])
					# kp_vec = k_vec + q_vec
					# kp_hat = kp_vec/NP.linalg.norm(kp_vec)
					# abs_kp = NP.linalg.norm(kp_vec)



					# if abs_q is outside a desired range continue
					if abs_q > absq_range[1] or abs_q < absq_range[0] :
						sigma_out_ind += 1
						continue;
					# # num_calc += 1

					# get the observational mode details for this |k'| and n'
					MODE_w_nkp,MODE_gamma_nkp,N_nkp = self.ref_fit_params(abs_kp,radial_orderp)

					# determine the eigenfunctions for k'		
					Xi_h_kp   = ITP_Xi_hp (abs_kp,z).squeeze()
					Xi_z_kp   = ITP_Xi_zp (abs_kp,z).squeeze()
					dzXi_h_kp = ITP_dzXi_hp(abs_kp,z).squeeze()
					dzXi_z_kp = ITP_dzXi_zp(abs_kp,z).squeeze()

					# If there is an interpolation error in the eigenfunctions, continue
					if NP.sum(NP.isnan([Xi_h_kp,Xi_z_kp,dzXi_h_kp,dzXi_z_kp])) != 0:
						sigma_out_ind += 1
						# # interp_error += 1
						continue;


					# # kdotkp = (kx[kx_ind]*kxp_grid[kxp_ind,kyp_ind] + ky[ky_ind]*kyp_grid[kxp_ind,kyp_ind])

					if multiplyH:
						# compute the R_wk and H_kkp
						R_wk  = self.R_omega_k(omega,MODE_w_nk,MODE_gamma_nk)
						R_wkp = self.R_omega_k(omega,MODE_w_nkp,MODE_gamma_nkp)

						H_kkp = -2.j*omega * (N_nk*abs(R_wk)**2*R_wkp + \
												N_nkp*abs(R_wkp)**2*NP.conj(R_wk))
					else:
						H_kkp = 0

					# Compute k_hat.k'_hat and k.q
					kdotkp_hat = NP.dot(k_hat,kp_hat)
					kdotq      = NP.dot(k_vec,q_vec)

					if scaleFields == 1:
						LZ = abs(z[0]-z[-1])
					elif scaleFields == 2:
						if abs_q == 0:
							LZ = 1
						else:
							LZ = 1/abs_q
					else:
						LZ = 1

					for kind in range(len(kernelType)):

						# compute integrand for P matrix
						if kernelType[kind].upper() == 'POLOIDAL':
							integrand = (abs_q**2*fj)*(kdotkp_hat*dzXi_h_k*NP.conj(Xi_h_kp) + dzXi_z_k*NP.conj(Xi_z_kp))\
											- NP.dot(k_vec,q_vec)*(dfj-fj/Hrho)*(kdotkp_hat*Xi_h_k*NP.conj(Xi_h_kp) + Xi_z_k*NP.conj(Xi_z_kp))
							integrand = integrand * rho[None,:] 
							integrand = integrand * LZ**2

						# compute the integrand for the T matrix
						elif kernelType[kind].upper() == 'TOROIDAL':			
							integrand = NP.dot(k_vec,NP.cross(q_vec,NP.array([0,0,1]))) * (fj*rho[None,:]) * (kdotkp_hat*Xi_h_k*NP.conj(Xi_h_kp) + Xi_z_k*NP.conj(Xi_z_kp))
							integrand = integrand * LZ

						# compute the integrand for the cs matrix (kernels for dc /c)
						elif kernelType[kind].upper() == 'SOUNDSPEED':
							integrand = -2 * fj * rho[None,:]* (cs[None,:]**2) * (-abs_k*Xi_h_k + dzXi_z_k) * (-abs_kp*NP.conj(Xi_h_kp) + NP.conj(dzXi_z_kp) ) 

						#--------------------------------
						# Magnetic Lorenz Stress
						elif kernelType[kind].upper() == 'LORENZ':

							integrand = (kdotkp_hat)*(dzXi_h_k*dzXi_h_kp + 0.5*(abs_k*dzXi_z_k*Xi_h_kp + abs_kp*dzXi_z_kp*Xi_h_k+abs_k*Xi_z_k*dzXi_h_kp + abs_kp*Xi_z_kp*dzXi_h_k))
							integrand+= abs_k*abs_kp*Xi_h_k*Xi_h_kp + 0.5*(abs_kp*dzXi_z_k*Xi_h_kp + abs_k*dzXi_z_kp*Xi_h_k+abs_kp*Xi_z_k*dzXi_h_kp + abs_k*Xi_z_kp*dzXi_h_k)
							integrand = fj  * integrand * (rho*cs**2) [None,:]# for kernels of va^2/cs^2
							# integrand = fj  * integrand / (4*NP.pi*1e-7)#for kernels of B^2
							
						# Density (kernels for drho /rho)
						elif kernelType[kind].upper() == 'DENSITY':
							gg = 274 # surface gravitational acceleration
							integrand = -MODE_w_nk**2 * (kdotkp_hat * Xi_h_k*NP.conj(Xi_h_kp) + Xi_z_k*NP.conj(Xi_z_kp))[None,:]
							integrand += -cs[None,:]**2 * (-abs_k*Xi_h_k + dzXi_z_k) * (-abs_kp*NP.conj(Xi_h_kp) + NP.conj(dzXi_z_kp))
							integrand += gg * Xi_z_k * (-abs_kp * NP.conj(Xi_h_kp) + NP.conj(dzXi_z_kp))
							integrand += -gg*( NP.dot(kp_vec,k_hat) * Xi_h_k*NP.conj(Xi_z_kp) + Xi_z_k*NP.conj(dzXi_z_kp))

							integrand = fj * rho[None,:] * integrand
						# ux,uy kernels
						elif kernelType[kind].upper() in ['UX','UY']:
							integrand = fj * rho * (kdotkp_hat * Xi_h_k*NP.conj(Xi_h_kp) + Xi_z_k * NP.conj(Xi_z_kp))
							if kernelType[kind].upper()  == 'UX':
								integrand = integrand * 1.j *kx[kx_ind]
							elif kernelType[kind].upper() == 'UY':
								integrand = integrand * 1.j * ky[ky_ind]

						# uz kernel
						elif kernelType[kind].upper() == 'UZ':
							integrand = fj * rho * (kdotkp_hat * dzXi_h_k*NP.conj(Xi_h_kp) + dzXi_z_k * NP.conj(Xi_z_kp))


						if BasisClass.__class__.__name__ == 'constantBasis1D':
							kernels     [kind,qx_out_ind,qy_out_ind,sigma_out_ind,:] = NP.sum(NP.dot(integrand,BasisClass.mass_),axis=1)
						else:						
							kernels     [kind,qx_out_ind,qy_out_ind,sigma_out_ind,:] = simps(integrand,x=z,axis=1)



					# if BasisClass.__class__.__name__ == 'constantBasis1D':
					# 	Poloidal_kernel     [qx_out_ind,qy_out_ind,sigma_out_ind,:] = NP.sum(NP.dot(integrand_P,BasisClass.mass_),axis=1) # not sure on negative, but only way to make sense
					# 	Toroidal_kernel     [qx_out_ind,qy_out_ind,sigma_out_ind,:] = NP.sum(NP.dot(integrand_T,BasisClass.mass_),axis=1)
					# 	Soundspeed_kernel   [qx_out_ind,qy_out_ind,sigma_out_ind,:] = NP.sum(NP.dot(integrand_C,BasisClass.mass_),axis=1)
					# 	Density_kernel      [qx_out_ind,qy_out_ind,sigma_out_ind,:] = NP.sum(NP.dot(integrand_rho,BasisClass.mass_),axis=1)
					# 	Alfvenspeedzz_kernel[qx_out_ind,qy_out_ind,sigma_out_ind,:] = NP.sum(NP.dot(integrand_Vzz,BasisClass.mass_),axis=1)
					# else:						
					# 	Poloidal_kernel     [qx_out_ind,qy_out_ind,sigma_out_ind,:] = simps(integrand_P,x=z,axis=1)
					# 	Toroidal_kernel     [qx_out_ind,qy_out_ind,sigma_out_ind,:] = simps(integrand_T,x=z,axis=1)
					# 	Soundspeed_kernel   [qx_out_ind,qy_out_ind,sigma_out_ind,:] = simps(integrand_C,x=z,axis=1)
					# 	Density_kernel      [qx_out_ind,qy_out_ind,sigma_out_ind,:] = simps(integrand_rho,x=z,axis=1)
					# 	Alfvenspeedzz_kernel[qx_out_ind,qy_out_ind,sigma_out_ind,:] = simps(integrand_Vzz,x=z,axis=1)


					# abort

					# if NP.sum(NP.isnan(Poloidal_kernel[qx_out_ind,qy_out_ind,sigma_out_ind,:])):
					# 	abort
					# if qx_out_ind == 7 and qy_out_ind == 0 and sigma_out_ind == 0:
					# 	abort2

					# Poloidal_kernel     [qx_out_ind,qy_out_ind,sigma_out_ind,:] = Poloidal_kernel     [qx_out_ind,qy_out_ind,sigma_out_ind,:]*LZ**2
					# Toroidal_kernel     [qx_out_ind,qy_out_ind,sigma_out_ind,:] = Toroidal_kernel     [qx_out_ind,qy_out_ind,sigma_out_ind,:]*LZ
					# # Toroidal_kernel  [qx_out_ind,qy_out_ind,sigma_out_ind,:] = Toroidal_kernel  [qx_out_ind,qy_out_ind,sigma_out_ind,:]#*LZ**2
					# Soundspeed_kernel   [qx_out_ind,qy_out_ind,sigma_out_ind,:] = Soundspeed_kernel   [qx_out_ind,qy_out_ind,sigma_out_ind,:]
					# Density_kernel      [qx_out_ind,qy_out_ind,sigma_out_ind,:] = Density_kernel      [qx_out_ind,qy_out_ind,sigma_out_ind,:]
					# Alfvenspeedzz_kernel[qx_out_ind,qy_out_ind,sigma_out_ind,:] = Alfvenspeedzz_kernel[qx_out_ind,qy_out_ind,sigma_out_ind,:]

					sigma_out_ind += 1
				qy_out_ind += 1
			qx_out_ind += 1

		t4 = time.time()

		# print(t2-t1,'sec')
		# print(t3-t2,'sec')
		# print(t4-t3,'sec')


		# return NP.array([(abs_q**2*fj-ddfj)*(kdotkp_hat*dzXi_h_k*NP.conj(Xi_h_kp) + dzXi_z_k*NP.conj(Xi_z_kp)),\
							# NP.dot(k_vec,q_vec)*dfj*(kdotkp_hat*Xi_h_k*NP.conj(Xi_h_kp) + Xi_z_k*NP.conj(Xi_z_kp))])
		return kernels


	def compute_kernels_parallel(self,radial_order,omega_nk,k_mask,qx_ind_arr,qy_ind_arr,BasisClass,kernelType,\
								kIndRange = None,sumOmega=False,multiplyH=False,scaleFields=2,\
								absq_range=NP.array([0,1e9])/RSUN,nbProc=8,reorg_k=True,VERBOSE=True,\
								saveDir = None):


		if not hasattr(kernelType,'__len__'):
			kernelType = [kernelType]
		if not isinstance(kernelType[0],str):
			raise Exception('kernelType should be string or list of strings')


		if hasattr(radial_order,'__len__'):
			if len(radial_order) > 2:
				raise Exception("Too many n given: Can only compute [n,n'] coupling")
			radial_orderp = radial_order[1]
			radial_order  = radial_order[0]
		else:
			radial_orderp = radial_order

		# Compute the kx and ky indices from the input mask
		if k_mask.ndim == 2:
			kx_inds,ky_inds = NP.where(k_mask==1)
		elif k_mask.ndim == 3:
			kx_inds,ky_inds,omega_inds = NP.where(k_mask==1)#[100:300,100:300,:]
			if len(omega_nk) != k_mask.shape[-1]:
				raise Exception('len(omega_nk) != k_mask.shape[-1]')
			omega_nk = omega_nk[omega_inds]

		if omega_nk is None:
			kx,ky,omega = self.computeFreq()
			abskg = NP.sqrt(kx[kx_inds]**2+ky[ky_inds]**2)
			omega_nk = self.ref_fit_params(abskg,radial_order)[0]

		# Clear and compute the globals given the desired input parameters
		if 'ddfj' in globals():
			global ddfj
			del ddfj
		self.compute_kernels([radial_order,radial_orderp],omega_nk[1],0,0,NP.arange(1),NP.arange(1),BasisClass,kernelType,absq_range)

		# Compute the kernels for each kx and ky in the masked area
		if nbProc == 1:
			res_tmp = NP.zeros((2,len(qx_ind_arr),len(qy_ind_arr),1,BasisClass.nbBasisFunctions_,len(kx_inds)),complex)
			if VERBOSE:
				PG = progressBar(len(kx_inds),'serial')
			for ii in range(len(kx_inds)):
				res_tmp[...,ii] = self.compute_kernels(radial_order,omega_nk,kx_inds[ii],ky_inds[ii],qx_ind_arr,qy_ind_arr,BasisClass,kernelType,absq_range,multiplyH,scaleFields)
				if VERBOSE:
					PG .update()
			if VERBOSE:
				del PG
		else:
			res_tmp = reduce(parallelize_classObject,(self.compute_kernels,radial_order,omega_nk,kx_inds,ky_inds,qx_ind_arr,qy_ind_arr,BasisClass,kernelType,absq_range,multiplyH,scaleFields),len(kx_inds),nbProc,progressBar=VERBOSE)

		res_tmp = res_tmp

		if not reorg_k:
			kx,ky,omega = self.computeFreq()

			if saveDir is not None:
				mkdir_p(saveDir)
				for ii in range(len(qx_ind_arr)):
					for jj in range(len(qy_ind_arr)):
						outDir = saveDir + '/Kernels/qx_%i/qy_%i/' % (qx_ind_arr[ii],qy_ind_arr[jj])
						mkdir_p(outDir)
						NP.savez_compressed(outDir + '/Kernels_n%i_np%i.npz' % (radial_order,radial_orderp),\
											Kernels = res_tmp[:,ii,jj],\
											QX = qx_ind_arr,QY = qy_ind_arr,\
											kx = kx[kx_inds],ky=ky[ky_inds],\
											mask = k_mask,\
											dims = ['Poloidal/Torodial','SIGMA','BasisNo.','k'])
				return kx[kx_inds],ky[ky_inds],'SAVED: %s' % saveDir
			return kx[kx_inds],ky[ky_inds],res_tmp

		# if not returnConventional:
		# 	kx_inds_u = NP.unique(kx_inds);ky_inds_u = NP.unique(ky_inds);w_inds_u = NP.unique(omega_inds);
		# 	if k_mask.ndim == 3 and not sumOmega:
		# 		res = NP.zeros((len(kx_inds_u),len(ky_inds_u),len(w_inds_u)) + res_tmp.shape[:-1],complex)*NP.nan
		# 	else:
		# 		res = NP.zeros((len(kx_inds_u),len(ky_inds_u)) + res_tmp.shape[:-1],complex)*NP.nan

		# 	for ii in range(res_tmp.shape[-1]):
		# 		if k_mask.ndim == 3 and not sumOmega:
		# 			res[kx_inds[ii]-min(kx_inds_u),ky_inds[ii]-min(ky_inds_u),omega_inds[ii]-min(w_inds_u)] = res_tmp[...,ii]
		# 		else:
		# 			res[kx_inds[ii]-min(kx_inds_u),ky_inds[ii]-min(ky_inds_u)] += NP.nan_to_num(res_tmp[...,ii])
			# return NP.moveaxis(res,[2,3][int(k_mask.ndim == 3 and not sumOmega)],0) 
		if kIndRange is not None:
			if k_mask.ndim ==3 and not sumOmega:
				res = NP.zeros((kIndRange[1]-kIndRange[0]+1,kIndRange[1]-kIndRange[0]+1,k_mask.shape[-1]) + res_tmp.shape[:-1],complex)#*NP.nan
			else:
				res = NP.zeros((kIndRange[1]-kIndRange[0]+1,kIndRange[1]-kIndRange[0]+1) + res_tmp.shape[:-1],complex)#*NP.nan
			kshift = kIndRange[0]
		else:
			res = NP.zeros((k_mask.shape,k_mask.shape[:-1])[int(k_mask.ndim ==3 and sumOmega)] + res_tmp.shape[:-1],complex)
			kshift=0
		for ii in range(len(kx_inds)):
			if k_mask.ndim == 2 or (k_mask.ndim ==3 and sumOmega):
				res[kx_inds[ii]-kshift,ky_inds[ii]-kshift] += res_tmp[...,ii]
			elif k_mask.ndim ==3:
				res[kx_inds[ii]-kshift,ky_inds[ii]-kshift,omega_inds[ii]] = res_tmp[...,ii]


		# Move axis to create output (nKernels,kx,ky,qx,qy,nbBasisFunctions_)
		res = NP.moveaxis(res,k_mask.ndim-[0,1][int(k_mask.ndim ==3 and sumOmega)],0)

		# if sumOmega and k_mask.ndim == 3:
			# res = NP.nansum(res,axis=3)

		return res

	def compute_N_nk(self,mask,nn,delta_kR,num_linewidths=2,PLOT = False,returnNonAvg=False):

		if PLOT:
			fig,ax = plt.subplots(1,3,figsize=[12,4])

		# Obtain the indices of the kx,ky used
		maskT = NP.ones(mask.shape)
		inds_kx,inds_ky = NP.where(maskT)

		# Calculate the absk of these indices
		kx,ky,omega = self.computeFreq()
		kxg,kyg = NP.meshgrid(kx,ky,indexing='ij')
		abskg = NP.sqrt(kxg**2+kyg**2)
		abskg_mask = abskg[inds_kx,inds_ky]

		# Calculate the resonant Frequency
		omegaG,gammaG,ampG = self.ref_fit_params(abskg_mask,nn,usePrecompute=True)


		# Compute the normalization for each indice
		amps = copy.copy(mask)*1.
		for kk in range(len(inds_kx)):
			if NP.isnan(omegaG[kk]):
				# if we don't have a fit for it, then set amp to zero
				amps[inds_kx[kk],inds_ky[kk]] = 0
				continue
			indl = NP.argmin(abs(omega - (omegaG[kk]-num_linewidths*gammaG[kk])))
			indr = NP.argmin(abs(omega - (omegaG[kk]+num_linewidths*gammaG[kk])))
			POW = abs(phi_kw[indl:indr,inds_ky[kk],inds_kx[kk]])**2
			Loren = self.R_omega_k(self.omega_[indl:indr],omegaG[kk],gammaG[kk])
			amps[inds_kx[kk],inds_ky[kk]] = NP.sum(POW)/NP.sum(abs(Loren)**2)

		# Create bins of width delta_kR
		nBins = NP.ceil((abskg_mask.max()-abskg_mask.min())*RSUN / delta_kR)
		absk_bins = NP.histogram(abskg_mask,bins=int(nBins))[1]

		# Average the N_nk into N_k
		N_k = copy.copy(mask)*1.
		for kk in range(len(absk_bins)-1):
			inds = (abskg_mask >= absk_bins[kk]) * (abskg_mask <= absk_bins[kk+1])
			tmp = NP.nanmean(amps[inds_kx,inds_ky][inds])
			N_k[inds_kx[inds],inds_ky[inds]] = tmp

		if PLOT:
			ax[0].pcolormesh(mask.T)
			ax[1].pcolormesh(amps.T,vmin=0,vmax=1.2*NP.amax(N_k))
			ax[2].pcolormesh(N_k.T,vmin=0,vmax=1.2*NP.amax(N_k))

			ax[0].set_title('Mask')
			ax[1].set_title(r'N$_{\rm n\mathbf{k}}$')
			ax[2].set_title(r'N$_{\rm nk}$')
			plt.tight_layout()


		# Return
		if returnNonAvg:
			return amps
		else:
			return N_k


	def test_plots(self,radial_order = 0,nu = 0.003157,mask = None,vmaxScale=1,lineWidthFactor=1,positiveFreq=True):
		kx,ky,omega = self.computeFreq()

		ind = NP.argmin(abs(omega - nu*2*NP.pi))

		plt.figure()

		if 'phi_kw' not in globals():
			self.computeFFT(storeInInstance=True)
		plt.pcolormesh(kx*RSUN,ky*RSUN,abs(phi_kw[ind]),cmap='Greys',vmax=15*vmaxScale)
		plt.colorbar()
		plt.xlim([-1500,1500])
		plt.ylim([-1500,1500])

		kxg,kyg = NP.meshgrid(kx,ky,indexing='ij')

		absk = NP.sqrt(kxg**2+kyg**2)

		if mask is None:
			mask = self.mask_cube(radial_order,nu*2*NP.pi,2)[0]
		# mask += NP.where((absk*RSUN<750)*(absk*RSUN>550),1,0)

		plt.pcolormesh(kx*RSUN,ky*RSUN,mask,alpha=0.2,cmap='Blues_r')
		plt.colorbar()

		plt.figure()

		plt.pcolormesh(kx*RSUN,omega/(2*NP.pi)*1e6,abs(phi_kw[:,:,NP.argmin(abs(ky))]),vmax=40*vmaxScale,cmap='Greys')
		plt.plot(kx[::5]*RSUN,self.fmode_dispersion_fit_params(abs(kx))[0][::5]/(2*NP.pi)*1e6,'.r',label='Theoretical f mode Dispersion')
		for nn in [0,1,2,3,4,5]:
			plt.plot(kx[::5]*RSUN,self.ref_fit_params(abs(kx),nn)[0][::5]/(2*NP.pi)*1e6,'-.',label='Observed %s Dispersion' % ['f mode','p%i mode' % nn][int(nn !=0)],color='C%i' % nn)
			plt.plot(kx[::5]*RSUN,-self.ref_fit_params(abs(kx),nn)[0][::5]/(2*NP.pi)*1e6,'-.',color='C%i' % nn)

			plt.fill_between(kx[::5]*RSUN,(self.ref_fit_params(abs(kx),nn)[0][::5] + self.ref_fit_params(abs(kx),nn)[1][::5]/2*lineWidthFactor)/(2*NP.pi)*1e6,\
											(self.ref_fit_params(abs(kx),nn)[0][::5] - self.ref_fit_params(abs(kx),nn)[1][::5]/2*lineWidthFactor)/(2*NP.pi)*1e6,color='C%i' % nn,alpha=0.2)
		plt.xlim([0,2000])
		plt.ylim([[-5000,0][int(positiveFreq)],5000])
		plt.legend()

		return abs(phi_kw[ind]),abs(phi_kw[:,:,NP.argmin(abs(ky))])


	def build_synFlow(self,z0,Dz,q_vec,sigma,BasisClass=3,PLOT=False):
		with fits.open('eigs/model.fits') as hdu:
			z,rho = hdu[0].data[[0,3]]
		xgrid,ygrid,tgrid = self.computeRealSpace()
		x_vec = NP.array([xgrid,ygrid])

		self.eig_model_z_   = z
		self.eig_model_rho_ = rho

		if sigma == 0:
			tgrid = NP.zeros(1)

		divh_u = NP.exp(-(z-z0)**2/Dz**2)
		divh_u_p = BasisClass.projectOnBasis(divh_u)

		if PLOT:
			plt.plot(z,divh_u,label='original',linewidth=2)
			plt.plot(z[::10],BasisClass.reconstructFromBasis(divh_u_p)[::10],'.r',label='Reconstructed')
			plt.legend()
			plt.xlabel('Height [Mm]')
			plt.xlabel(r'$\nabla_h\cdot u$')

		return divh_u,divh_u_p

#-----------------------------------------------------------




def RLSinversion_MCA(A,b,alpha,Lmatrix=None,Lcurve=False,knotConstraint=None,GaussianScale = 1,returnNoiseProp=False,Noise=0.):
	# Solve for x in Ax=b using tikhonov regularization
	if Noise == 0.:
		Nrealizations = 1
	else:
		Nrealizations = Noise.shape[1]



	if not hasattr(alpha,'__len__'):
		alpha = [alpha]

	if Lmatrix is None:
		Lmatrix = NP.eye(A.shape[-1])



	A2 = NP.dot(A.T,A)
	Ab = NP.dot(A.T,b)
	L2 = NP.dot(Lmatrix.T,Lmatrix)
	# A2 = A.T@A
	# Ab = A.T@b
	# L2 = Lmatrix.T@Lmatrix


	ans = NP.zeros((len(A2),len(alpha)))
	std = NP.zeros((len(A2),len(alpha)))

	if NP.sum(NP.iscomplex(b)):
		ans = ans.astype(complex)
		std = std.astype(complex)
	if returnNoiseProp:
		noiseProp = NP.zeros((len(A2),len(alpha),Nrealizations))

	
	for ii in range(len(alpha)):
		mat = A2 + alpha[ii]**2*L2

		if knotConstraint is not None:
			knotCoeffs = knotConstraint[0]
			constraint = knotConstraint[1]

			if not hasattr(constraint,'__len__'):
				knotCoeffs = [knotCoeffs]
				constraint = [constraint]


			mat = NP.pad(mat,((0,len(constraint)),(0,len(constraint))),constant_values=((0,0),(0,0)))
			# if len(constraint) == 1:
				# mat[-1,:-1] = knotCoeffs[0];mat[:-1,-1] = knotCoeffs[0]
			# elif len(constraint) == 2:
				# mat[-2,:-2] = knotCoeffs[0];mat[:-2,-2] = knotCoeffs[0]
				# mat[-1,:-2] = knotCoeffs[1];mat[:-2,-1] = knotCoeffs[1]

			for cc in range(1,len(constraint)+1):
				mat[-cc,:-len(constraint)] = knotCoeffs[len(constraint)-cc];mat[:-len(constraint),-cc] = knotCoeffs[len(constraint)-cc]

			if ii == 0:
				Ab = NP.append(Ab,NP.array(constraint).astype(float))


		if NP.linalg.det(mat) == 0:
			ans[:,ii] = NP.nan
			continue

		if knotConstraint is not None:
			ans[:,ii] = NP.linalg.solve(mat,Ab)[:-len(constraint)]
		else:
			ans[:,ii] = NP.linalg.solve(mat,Ab)

		
		if knotConstraint is not None:	
			# if ii == 0:
			noise = Noise/NP.sqrt(GaussianScale)
			forwardCall = NP.dot(A,ans[:,ii])[:,None] + noise
			Abnoise = NP.dot(A.T,forwardCall)			
			# forwardCall = A@(ans[:,ii][:,None]) + noise
			# Abnoise = A.T@forwardCall
			Abnoise = NP.concatenate([Abnoise,NP.array(constraint).astype(float)[:,None]*NP.ones((len(constraint),Nrealizations))],axis=0)
			
			if knotConstraint is not None:
				sol_noise = NP.linalg.solve(mat,Abnoise)[:-len(constraint)]
			else:
				sol_noise = NP.linalg.solve(mat,Abnoise)


			std[:,ii] = NP.std(sol_noise,axis=1)

			if returnNoiseProp:
				noiseProp[:,ii,:] = sol_noise

	# Li = NP.linalg.inv(Lmatrix)
	# AL = NP.dot(A,Li)
	# A22 = NP.dot(AL.T,AL)
	# Ab2 = NP.dot(AL.T,b)

	# mat2 = A22 + alpha**2*NP.eye(len(Li))
	# ans2 = NP.linalg.solve(mat2,Ab2)
	# ans2  = NP.dot(Li,ans2)

	# abort

	if Lcurve:
		residual = [];normLx = []
		for ii in range(len(alpha)):
			residual.append(norm2(b-NP.dot(A,ans[:,ii])))
			normLx.append(norm2(NP.dot(Lmatrix,ans[:,ii])))
		return NP.array([residual,normLx])

	if knotConstraint is not None:
		if returnNoiseProp:
			return noiseProp
		else:
			return NP.array([ans,std])
	else:
		return NP.array([ans,std])




def SOLA_coeffCalc_MCA(KKi,mu,Basis1D,targetDepth,targetWidth,returnKernel=False,returnLcurve=False,\
						NoiseCovInvSqrt = None,SVDmethod=False,rcond = 1e-6,\
						constraint = None):
	# Coefficients are for the scaled Bcoefficients
	# KKi = NP.dot(LambdaSqrtInv,Kernels)


	if not hasattr(targetDepth,'__len__'):
		targetDepth = [targetDepth]
	if not hasattr(targetWidth,'__len__'):
		targetWidth = [targetWidth]

	if len(targetWidth) != len(targetDepth):
		raise Exception(' Vector of targetDepths must be same length as vector of targetWidth')

	# scale by the Noise Covariance
	Eij = NP.eye(len(KKi))

	# Determine K(z)
	KKiSOLA = NP.dot(KKi[:,:Basis1D.nbBasisFunctions_],NP.linalg.inv(Basis1D.mass_))
	KKz = Basis1D.reconstructFromBasis(KKiSOLA,axis=-1)

	# Build the Amatrix
	KKiSOLAM = NP.dot(KKiSOLA,Basis1D.mass_)
	if not SVDmethod:
		Amatrix = NP.pad(NP.dot(KKiSOLA,KKiSOLAM.T)+mu*Eij,((0,1),(0,1)))
		Amatrix[:-1,-1] = NP.sum(KKiSOLAM,axis=-1)
		Amatrix[-1,:-1] = NP.sum(KKiSOLAM,axis=-1)
		Amatrix[-1,-1] = 0
	else:
		Amatrix = NP.dot(KKiSOLA,KKiSOLAM.T)
	

	# And determine the target functions, looping over the target depths
	Targets = [];vMatrix = []
	for ii in range(len(targetDepth)):
		# compute the Target Function and project on basis
		Target = NP.exp(-(Basis1D.x_ - targetDepth[ii])**2/(2*(targetWidth[ii]/(2*NP.sqrt(2*NP.log(2))))**2)); Target = Target/simps(Target,x=Basis1D.x_)
		TargetP = Basis1D.projectOnBasis(Target)

		# Compute the v Matrix
		if not SVDmethod:
			vMatrix.append(NP.pad(NP.dot(KKiSOLAM,TargetP),(0,1),constant_values=1))
		else:
			vMatrix.append(NP.dot(KKiSOLAM,TargetP))




		Targets.append(Basis1D.reconstructFromBasis(TargetP))
	Targets = NP.array(Targets);vMatrix = NP.array(vMatrix).T

	# return Amatrix,vMatrix,Targets

	BsplineCoeffs = [];dBsplineCoeffs = []
	for ii in range(Basis1D.nbBasisFunctions_):
		BsplineCoeffs.append(Basis1D(ii,x=NP.array([0])))
		dBsplineCoeffs.append(Basis1D(ii,x=NP.array([0]),derivative=1))
	BsplineCoeffs = NP.squeeze(BsplineCoeffs)
	dBsplineCoeffs = -NP.squeeze(dBsplineCoeffs) #Negative because -q^2 dz(P) = divh(u) 


	if not SVDmethod:
		# Solve for the coefficients
		coeffs_grid = NP.linalg.solve(Amatrix,vMatrix).T
	else:
		Ainv = NP.linalg.pinv(Amatrix,rcond=rcond,hermitian=True)
		if constraint is not None:
			Ainv = NP.pad(Ainv,((0,len(constraint)),(0,len(constraint))),constant_values=((0,0),(0,0)))
			if len(constraint) == 1:
				Ainv[-1,:-1] = NP.dot(KKiSOLAM,BsplineCoeffs);Ainv[:-1,-1] = NP.dot(KKiSOLAM,BsplineCoeffs)
				vMatrix = NP.pad(vMatrix.astype(complex),((0,1),(0,0)),constant_values=constraint[0][0])
			# elif len(constraint) == 2:
				# mat[-2,:-2] = knotCoeffs[0];mat[:-2,-2] = knotCoeffs[0]
				# mat[-1,:-2] = knotCoeffs[1];mat[:-2,-1] = knotCoeffs[1]
		coeffs_grid = NP.dot(Ainv,vMatrix).T
		if constraint is not None:
			coeffs_grid = coeffs_grid[:,:-1]



	if returnKernel and not returnLcurve:
		return coeffs_grid,KKz,Targets
	elif returnLcurve:
		if NoiseCovInvSqrt is None:
			raise Exception('Must define NoiseCovInvSqrt')
		return [NP.trapz((Targets - NP.dot(coeffs_grid[:,:-1],KKz))**2,x=Basis1D.x_,axis=1),NP.sqrt(NP.sum(coeffs_grid[:,:-1]**2*NoiseCovInvSqrt[None,:],axis=1))]
	else:
		return coeffs_grid


	

# t0 = time.time()
# ref_fit_params(NP.arange(500,1000,50)/RSUN,0,True)
# t1 = time.time()

# print(t1-t0,'sec')
# abort



def units_check():
	plt.figure()
	# load the background and eigenfunctions
	z,rho = fits.open('eigs/model.fits')[0].data[[0,3]]
	eig_k = fits.open('eigs/k.fits')[0].data.squeeze()
	eig_w = fits.open('eigs/omega%02d.fits' % radial_order)[0].data.squeeze()
	Xi_h,Xi_z,dzXi_h,dzXi_z = fits.open('eigs/eig%02d.fits' % radial_order)[0].data[[0,1,3,5]]

	# convert to SI
	eig_k *= 1e-6
	z     *= 1e6
	rho   *= 1e3


	mask = test_plots()

	plt.plot(eig_k*RSUN,eig_w*1e6/(2*NP.pi),'g',label='Eigenfunction model')
	plt.legend()

	plt.figure()
	plt.plot(z,rho,label = 'Eigenfunction model')
	plt.xlim([z.min(),z.max()])
	plt.ylim([rho.min(),rho.max()])
	r_modelS,c_modelS,rho_modelS = NP.genfromtxt('/home/ch3246/mps_montjoie/data/background/modelS_SI_original.txt')[:,:3].T
	plt.plot((r_modelS*RSUN - RSUN)[::10],rho_modelS[::10],'.',label='model S')
	plt.ylabel(r'Density [kg/m$^3$]')
	plt.xlabel('Height [m]')
	plt.legend()

