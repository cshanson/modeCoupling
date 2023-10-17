import numpy as NP
from matplotlib.pyplot import *
from astropy.io import fits
# from scipy.sparse import coo_matrix,csr_matrix,hstack,vstack

from ..Common       import *
from ..Parameters   import *

plt.ion()

plt.close('all')



#-----------------------------------------------------------

class global_modeCoupling(object):

	def __init__(self,fitsPath,spectraPath,HMIDays,subdays = 'All',trackingRate = 0):

		self.fitsPath_    = fitsPath    # Folder that contains time series, with subfolders 001,002,....
		self.spectraPath_ = spectraPath # Folder to store the spectrra
		self.HMIDays_     = HMIDays     # which 'HMIdays' to use, integer of 72
		self.subdays_     = subdays		# which days of the time series to use
		self.trackingRate_= trackingRate # Tracking rate in nHz (e.g. 421.41 + 31.7 is Equatorial)
		
		# In case we want to look at 1 72 period and specify a integer instead of list
		if not hasattr(self.HMIDays_,'__len__'):
			self.HMIDays_ = [self.HMIDays_]


		# Load in the Leakage Matrix
		# if 'leakageMatrix' not in globals():
		# 	global leakageMatrix
		with fits.open('/home/ch3246/mps_montjoie/pyCompHelio/ModeCoupling/leakage/HMI/rleaks1fd.fits') as hdu:
			self.leakageMatrix_ = hdu[0].data

	def HMIday_to_date(self,HMIday):
		dt = datetime.datetime(2010,4,30)+ datetime.timedelta(days=HMIday - 6328)
		print(dt)
		# return dt


	def read_Leakage(self,ell,mm,dell,dmm,HMI=True,vw=False):
		# load in the data
		# leakage = dmm,dell,ell,mm
		# dmm = -15,15  # NOTHING to do with t
		# dell = -6,6  #NOTHING to do with s
		# ell = 0,249
		# mm = -249 249
		ellMax = 249;dellMax = 6;dmmMax = 15

		return self.leakageMatrix_[dmm+dmmMax,dell+dellMax,ell,mm+ellMax]


	def readFits(self,ell):
		# reads in the V_mt cube for chosen ell, returns corrected array (revert jsoc complex number array)
		# HMI day that the global mode series starts
		# Days is which days in the time series to use [DayIndStart,DayIndEnd] e.g. first 3 days [0,3]

		# global phi_mt

		subP = subprocess.Popen('ls %s/%03d/' % (self.fitsPath_,ell),\
	                              shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
		out,err  = subP.communicate()
		files = out.split()
		for tt in range(len(self.HMIDays_)):
			for ii in range(len(files)):
				filesTmp = files[ii].decode('utf-8')
				if (int(filesTmp.split('_')[-1][:4]) == self.HMIDays_[tt]):
					FILE = filesTmp

			with fits.open(self.fitsPath_ + '/%03d/' % ell + FILE) as hdu:
				HEADER = hdu[1].header
				phi_mtT = hdu[1].data
			phi_mtT = phi_mtT[:,::2] - 1.j*phi_mtT[:,1::2] # negative im because of jesper convention on ms
			if tt == 0:
				phi_mt = copy.copy(phi_mtT)
			else:
				phi_mt = NP.concatenate([phi_mt,phi_mtT],axis=-1)

		if self.subdays_ != 'All':
			phi_mt = phi_mt[...,self.subdays_[0]*1920:self.subdays_[-1]*1920]

		return phi_mt,HEADER


	def computeFFT(self,ell,fitsPathLoad = None,fitsPathSave=None):
		# reads in dopplercube with readFits, applies apodization, computes fourier transform,save spectrum


		if fitsPathLoad is not None and fitsPathSave is None:
			with fits.open(fitsPathLoad) as hdu:
				dat = hdu[0].data
			phi_mw = dat[0] + 1.j*dat[1]
			return phi_mw

		# read doppler cube
		phi_mtT,header_mt = self.readFits(ell)
		if self.subdays_ != 'All':
			tInds = NP.arange(self.subdays_[0]*1920,self.subdays_[1]*1920)
		else:
			tInds = NP.arange(phi_mtT.shape[-1])
		# track the doppler data exp(imwt)
		phi_mtT = phi_mtT * NP.exp(1.j*NP.arange(phi_mtT.shape[0])[:,None]*(self.trackingRate_*1e-9*2*NP.pi)* (tInds-tInds[0])[None,:]*45.)
		# get negative frequencies
		nM = NP.arange(1,phi_mtT.shape[0])
		phi_mtT = NP.concatenate([(NP.conj(phi_mtT)[1:,:]*(-1)**nM[:,None])[::-1,:],phi_mtT],axis=0)

		# compute temporal fft and store in globals
		# global phi_mw
		phi_mw = NP.fft.fftshift(NP.fft.ifft(phi_mtT,axis=-1,norm='ortho')/(2*NP.pi),axes=-1)

		# Save power spectrum
		if fitsPathSave is not None:
			header_mw = copy.copy(header_mt)
			header_mw['NAXIS'] = 4
			header_mw['NAXIS3'] = len(phi_mw)
			header_mw['NAXIS4'] = 2
			hdul = fits.PrimaryHDU(NP.array([phi_mw.real,phi_mw.imag]).astype('float32'),header=header_mw)
			hdul.writeto(fitsPathSave,overwrite=True)

		return phi_mw

	def computeFreq(self,ell,dt = 45):
		Ndays = len(self.HMIDays_)*72
		if self.subdays_ != 'All':
			Ndays = self.subdays_[-1] - self.subdays_[0]
		ell = int(ell)
		# computes the spatial and temporal frequencies (m,omega)
		omega = NP.fft.fftshift(NP.fft.fftfreq(int(Ndays*24*3600/dt),dt)*2*NP.pi )
		ms    = NP.arange(-ell,ell+1)

		# omega = omega[:len(omega)//2]
		return ms,omega
	

	def R_omega_lm(self,omega,omega_nl,Gamma_nl,A_nl = 1):
		# Compute the resonant response function (eq 18 Woodard 2006) or (eq 2 Hanasoge 2018)
		# Also called the "Lorentzian"

		if hasattr(omega_nl,'__len__') and hasattr(Gamma_nl,'__len__'):
			return A_nl[:,None]/( omega_nl[:,None]**2 - 1.j*omega[None,:]*Gamma_nl[:,None]/2 - omega[None,:]**2 )

		# return A_nl/( (omega_nl - 1.j*Gamma_nl/2)**2 - omega**2 )
		return A_nl/( omega_nl**2 - 1.j*omega*Gamma_nl/2 - omega**2 )






	def ref_fit_params(self,ell,radial_order,dateString='20140620',central_freq = False):
		ell = int(ell);radial_order = int(radial_order)
		# dateString = 'yyyymmdd'
		import math
		DATA = NP.genfromtxt('/scratch/ch3246/project/OBS_DATA/hmi_v_sht_modes/hmi.v_sht_modes.%s_000000_TAI.0.300.138240.m10qr.7840.36' % dateString)

		inds = (DATA[:,0] == ell)*(DATA[:,1] == radial_order)

		def recur_factorial(n):
			if n == 1:
				return n
			elif n < 1:
				return ("NA")
			else:
				return float(n*recur_factorial(n-1))

		if NP.sum(inds) != 1:
			return [NP.nan,NP.nan,NP.nan]

		omega_nl = DATA[inds,2].squeeze()*1e-6*2*NP.pi
		if central_freq:
			return omega_nl
		acoeff   = (DATA[inds,12:12+36].squeeze())[:ell]
		acoeff[0] -= 31.7
		acoeff *= 1e-9*2*NP.pi
		# print(acoeff.shape)
		gamma = DATA[inds,4].squeeze()*1e-6*2*NP.pi
		amp   = DATA[inds,3].squeeze()*1

		if not os.path.isfile('/home/ch3246/mps_montjoie/pyCompHelio/ModeCoupling/PolyTables/PolynomialTable_ell%i.txt' % ell):
			print(text_special("Polynomial Table not found. Computing Now",'g'))
			os.system('module load mathematica')
			os.system('/home/ch3246/mps_montjoie/pyCompHelio/ModeCoupling/PolyTable.wls %i' % ell)

		Pmatrix = NP.genfromtxt('/home/ch3246/mps_montjoie/pyCompHelio/ModeCoupling/PolyTables/PolynomialTable_ell%i.txt' % ell)

		domega     = acoeff@Pmatrix.T

		# for mm in range(-ell,ell+1):
			# print(mm)
			# for j in range(5):#range(len(acoeff)):
				# if (mm-mm != 0) or ((j+ell+ell)%2 == 1 and (mm==0)):
					# continue
				# clebschGordan = wigner3j(j,ell,ell,0,mm,-mm) * (-1)**(-j+ell-mm)*NP.sqrt(2*ell+1)
				# print(clebschGordan)
				# ell*NP.sqrt(math.factorial(2*ell-j)*math.factorial(2*ell+j+1))/(math.factorial(2*ell)*NP.sqrt(2*ell+1)) * 
				# Pmatrix[mm+ell,j] = clebschGordan * ell*NP.sqrt(recur_factorial(2*ell-j)*recur_factorial(2*ell+j+1))/(recur_factorial(2*ell)*NP.sqrt(2*ell+1))


		return [omega_nl+domega - NP.arange(-ell,ell+1)*self.trackingRate_*1e-9*2*NP.pi,gamma,amp]


	def gammaFunc(self,ell,ellp,sMax,m,t):
		# NOTE even permutations of wigner3j are identical
		return (-1.)**(m+t) * NP.sqrt(2.*NP.arange(sMax+1)+1) * wigner3j_full(ell,ellp,t,m,-(m+t),sMax)
		# return (-1.)**(m+t) * NP.sqrt(2.*s+1) * wigner3j(ellp,s,ell,-(m+t),t,m)

	def precompute_gammaFunc(self,ell,ellp,sMax,saveFilePath = '/home/ch3246/mps_montjoie/pyCompHelio/ModeCoupling/gammaFunc/',VERBOSE = False):
		# precompute gammaFunc
		ellMax = min(ell,ellp)
		gammaFunc_array = NP.zeros((sMax+1,2*sMax+1,2*ellMax+1))

		calcs = 0;computed_pairs = [];
		for mm in range(-ellMax,ellMax+1):
			for tt in range(-sMax,sMax+1):
				if (mm+ell+tt) >= (2*ell+1) or (mm+ell+tt) < 0 or (mm+ellp+tt) >= (2*ellp+1) or (mm+ellp+tt) < 0:
					continue

				if [-mm,-tt] in computed_pairs:
					gammaFunc_array[:,tt+sMax,mm+ellMax] = gammaFunc_array[:,-tt+sMax,-mm+ellMax]*(-1)**(ell+ellp+NP.arange(sMax+1))
				else:
					gammaFunc_array[:,tt+sMax,mm+ellMax] = self.gammaFunc(ell,ellp,sMax,mm,tt)
					calcs += 1
				computed_pairs.append([mm,tt])
		if VERBOSE:
			print('Number of calls to wigner3j: %i' % calcs)
		if saveFilePath is not None:
			NP.save(saveFilePath + '/gammaFunc_ell%i_ellp%i_sMax%i.npy' % (ell,ellp,sMax),gammaFunc_array)
		return 1.



	def compute_bcoeff_serial(self,ell,ellp,sMax,sigma_ind_arr,nu_min=0.0015,nu_max=0.0045,llp_same=True,\
								windFactor=2,radial_order = 'All',TEST=0,\
								powLoadPath = None,SAVEOUTPUT=None,tGrid = None,fitParamsDate='20140620',\
								sensitivity='FLOW',returnNoise=False):
		# Compute the obervable B coefficient
		# B^sigma_k_q = sum_w[H^w_l_lp_sigma* gmma^lst_tm phi^w_lm* phi^w+sigma_lm+t] / sum_w[|H^w_l_lp_sigma|^2]

		# Adjust the input
		ell = int(ell);ellp = int(ellp)
		if radial_order == 'All':
			DATA = NP.genfromtxt('/scratch/ch3246/project/OBS_DATA/hmi_v_sht_modes/hmi.v_sht_modes.%s_000000_TAI.0.300.138240.m10qr.7840.36' % fitParamsDate)
			inds = NP.arange(len(DATA))[DATA[:,0] == ell]
			indsp = NP.arange(len(DATA))[DATA[:,0] == ellp]
			ngrid = NP.union1d(inds,indsp)
			ngrid = NP.unique(DATA[ngrid,1])

		if not hasattr(self.HMIDays_,'__len__'):
			HMIdays = [self.HMIDays_]
		if self.subdays_ == 'All':
			Ndays = len(self.HMIDays_)*72
		else:
			Ndays = self.subdays_[1] - self.subdays_[0]

		if powLoadPath is not None:
			powLoadPath + 'POW_ell%i_hmiday%i_%idays.fits' %(ell,self.HMIDays_[0],len(self.HMIDays_)*72)


		if windFactor < 2:
			windFactor =2
			print(text_special('windFactor should be >=2 for numerical accuracy','y'))


		# compute the m and omega grid for this chosen ell
		ms,omega = self.computeFreq(ell)
		dw_shift = self.trackingRate_*1e-9*2*NP.pi



		# Compute the frequency fft of mode ell 
		# if powLoadDICT:
		# 	phi_mw = fitsDICT['ell%i' % ell]
		# else:
		phi_mw = self.computeFFT(ell,fitsPathLoad = powLoadPath)

		if TEST == 1:
			omega_nl,gamma_nl,amp_nl = self.ref_fit_params(ell,ngrid[0])
			return phi_mw,omega_nl,ngrid[0]


		# intialize the result matricies
		# 31 is max n modes
		# grid is [s,t,sigma,n]
		if tGrid is None:
			bcoeff_num = NP.zeros((sMax+1,sMax+1,len(sigma_ind_arr),31),complex)
			bcoeff_den = NP.zeros((sMax+1,sMax+1,len(sigma_ind_arr),31),complex)
			tGrid = NP.arange(sMax+1)
		else:
			bcoeff_num = NP.zeros((sMax+1,len(tGrid),len(sigma_ind_arr),31),complex)
			bcoeff_den = NP.zeros((sMax+1,len(tGrid),len(sigma_ind_arr),31),complex)			



		preCompGamma = NP.load('/home/ch3246/mps_montjoie/pyCompHelio/ModeCoupling/gammaFunc/gammaFunc_ell%i_ellp%i_sMax%i.npy' % (ell,ellp,sMax))
		# begin loop in n
		m_window = []
		for radial_order in ngrid:

			# load in the mode parameters for this ell and determine the frequency window indices
			omega_nl,gamma_nl,amp_nl = self.ref_fit_params(ell,radial_order,fitParamsDate)
			omega_nlp,gamma_nlp,amp_nlp = self.ref_fit_params(ellp,radial_order,fitParamsDate)

			# if mode params not found return zero array or outside nu range 
			if NP.sum(NP.isnan(omega_nl)) != 0 or NP.nanmean(omega_nl)/(2*NP.pi) > nu_max or NP.nanmean(omega_nl)/(2*NP.pi) < nu_min :
				continue
			if NP.sum(NP.isnan(omega_nlp)) != 0 or NP.nanmean(omega_nlp)/(2*NP.pi) > nu_max or NP.nanmean(omega_nlp)/(2*NP.pi) < nu_min :
				continue

			# load in nlp if need be
			if not llp_same:
				phi_mw_p = self.computeFFT(ellp,fitsPathLoad = powLoadPath)
				print('Will need to rework the loadpath command, e.g. specify the ellp')
			else:
				phi_mw_p = phi_mw




			# # compute the amplitude N_nl and N_nlp at m=0
			# indl = NP.argmin(abs(omega[:,None] - (omega_nl-3*gamma_nl)),axis=0)
			# indr = NP.argmin(abs(omega[:,None] - (omega_nl+3*gamma_nl)),axis=0)
			# N_nl = NP.zeros(len(indl))*NP.nan
			# for mmi in [0]:#range(-min(ell,ellp),min(ell,ellp)+1):
				# Loren = R_omega_lm(omega[indl[mmi+ell]:indr[mmi+ell]],omega_nl[mmi+ell],gamma_nl)
				# N_nl[mmi+ell] = NP.sum(abs(phi_mw[ell][indl[mmi+ell]:indr[mmi+ell]])**2)/NP.sum(abs(Loren)**2)
			# N_nl = NP.nanmean(N_nl)
			# indl = indl[ell];indr = indr[ell]

			inds = ( (omega - (omega_nl[ell]-5*gamma_nl)) > 0) * ( (omega - (omega_nl[ell]+5*gamma_nl)) < 0)
			Loren = self.R_omega_lm(omega[inds],omega_nl[ell],gamma_nl)
			N_nl = NP.sum(abs(phi_mw[ell][inds])**2)/NP.sum(abs(Loren)**2)

			# compute the amplitude N_nlp at m=0
			if llp_same:
				N_nlp = N_nl
			else:
				inds = ( (omega - (omega_nlp[ellp]-5*gamma_nlp)) > 0) * ( (omega - (omega_nlp[ellp]+5*gamma_nlp)) < 0)
				Loren = self.R_omega_lm(omega[inds],omega_nlp[ellp],gamma_nlp)
				N_nlp = NP.sum(abs(phi_mw_p[ellp][inds])**2)/NP.sum(abs(Loren)**2)

			if TEST == 4 and radial_order == 10:
				plt.figure()		
				plt.plot(omega[inds]/(2*NP.pi)*1e6,abs(phi_mw[ell])[inds]**2)
				plt.plot(omega[inds]/(2*NP.pi)*1e6,N_nl*abs(Loren)**2)
				plt.axvline(x=omega_nl[ell]/(2*NP.pi)*1e6,color='k')
				plt.title('Plot 1')
			


			for mm in range(-min(ell,ellp,100),min(ell,ellp,100)+1):
				# compute the frequencies around the mode
				omega_nl_ind  = NP.arange(len(omega))[(abs(omega) < omega_nl[mm+ell]   + windFactor*gamma_nl/2)*(abs(omega) > omega_nl[mm+ell]   - windFactor*gamma_nl/2)]
				omega_nlp_ind = NP.arange(len(omega))[(abs(omega) < omega_nlp[mm+ellp] + windFactor*gamma_nl/2)*(abs(omega) > omega_nlp[mm+ellp] - windFactor*gamma_nl/2)]

				omega_ind_all = NP.union1d(omega_nl_ind,omega_nlp_ind)
				omega_all     = omega[omega_ind_all]

				if TEST == 4 and mm == 0 and radial_order == 10:
					plt.plot(omega[omega_ind_all]/(2*NP.pi)*1e6,abs(phi_mw[mm+ell])[omega_ind_all]**2)
					plt.plot(omega_all/(2*NP.pi)*1e6,N_nl*abs(R_omega_lm(omega_all,omega_nl[mm+ell],gamma_nl))**2)
					plt.xlim(NP.array([omega[inds][0],omega[inds][-1]])/(2*NP.pi)*1e6)

					plt.figure()
					plt.plot(omega/(2*NP.pi)*1e6,abs(phi_mw[mm+ell])**2)
					plt.plot(omega_all/(2*NP.pi)*1e6,abs(phi_mw[mm+ell])[omega_ind_all]**2)
					plt.plot(omega_all/(2*NP.pi)*1e6,N_nl*abs(R_omega_lm(omega_all,omega_nl[mm+ell],gamma_nl))**2)
					plt.xlim(NP.array([omega[inds][0],omega[inds][-1]])/(2*NP.pi)*1e6)
					plt.title('Plot 2')
					print('Returning test plots')
					# return

				

				phi  = phi_mw[mm+ell][omega_ind_all]



				if TEST == 2:
					TMP = NP.zeros(omega.shape)
					TMP[omega_ind_all] = 1
					m_window.append(TMP)
					continue

				if len(NP.intersect1d(omega_nl_ind,omega_nlp_ind)) == 0:
					continue

				
				
				t_ind = 0
				for tt in tGrid:
					sigma_out_ind = 0
					
					for sigma_ind in sigma_ind_arr:
						if (mm+ell+tt) >= (2*ell+1) or (mm+ell+tt) < 0 or (mm+ellp+tt) >= (2*ellp+1) or (mm+ellp+tt) < 0:
							sigma_ind += 1
							continue
						else:
							omega_all_sigma = omega[omega_ind_all+sigma_ind] #+tt*4 WTF
							phip = phi_mw_p[mm+ellp+tt][omega_ind_all + sigma_ind] #+tt*4 WTF

							# Compute the R and  H func
							R_lm  = self.R_omega_lm(omega_all,omega_nl[mm+ell],gamma_nl)
							R_lmp = self.R_omega_lm(omega_all_sigma,omega_nlp[mm+ellp+tt],gamma_nlp)
							H_func = (N_nlp * NP.conj(R_lm)   * abs(R_lmp)**2 + \
													N_nl * R_lmp * abs(R_lm)**2)
							if sensitivity.upper() == 'FLOW':
								H_func = -2.*omega_all*H_func

							# Multiply by the leakage
							H_func = H_func*self.read_Leakage(ell,mm,0,0)*self.read_Leakage(ell,mm+tt,0,0)





						# if preCompGammaPath is not None:
						gamma_func = preCompGamma[:,tt+sMax,mm+min(ell,ellp)]
						# else:
						# 	if (abs(tt) <= sMax) and (abs(ellp-ell) <= sMax) and (abs(ell-sMax) <= ellp) and (abs(ellp-sMax)<=ell) and (abs(mm+tt) <= ellp):
						# 		gamma_func = self.gammaFunc(ell,ellp,sMax,mm,tt)
						# 	else:
						# 		gamma_func = NP.zeros(sMax+1)

						if TEST == 4 and mm == 0 and tt == 0 and radial_order ==10:
							plt.figure()
							plt.plot(omega_all/(2*NP.pi)*1e6,abs(phi)**2)
							plt.plot(omega_all/(2*NP.pi)*1e6,abs(phip)**2)
							plt.xlim([2762,2764])
							plt.title('Plot 3')

							plt.figure()
							plt.plot(H_func)
							plt.title('Plot 4')

						if returnNoise:
							bcoeff_num[:,t_ind,sigma_out_ind,int(radial_order)] += NP.sum(NP.abs(H_func[None,:]* gamma_func[:,None])**2  * N_nl * abs(R_lm)**2 * N_nlp * abs(R_lmp)**2,axis=1)
							bcoeff_den[:,t_ind,sigma_out_ind,int(radial_order)] += NP.sum(abs(H_func[None,:] * gamma_func[:,None])**2,axis=1)**2
						else:
							bcoeff_num[:,t_ind,sigma_out_ind,int(radial_order)] += NP.sum(NP.conj(H_func[None,:]) * gamma_func[:,None] * NP.conj(phi)[None,:] * phip[None,:],axis=1)
							bcoeff_den[:,t_ind,sigma_out_ind,int(radial_order)] += NP.sum(abs(H_func[None,:] * gamma_func[:,None])**2,axis=1)

						
						sigma_out_ind += 1

					t_ind += 1


		# ellp_ind += 1


		if SAVEOUTPUT is not None:
			mkdir_p(SAVEOUTPUT)
			NP.savez_compressed(SAVEOUTPUT + '/MCA_BCOEFFS%s[ell%i][ellp%i][sMax%i].npz' % (['','_Noise'][int(returnNoise)],ell,ellp,sMax),\
								Bcoeffs = NP.where(abs(bcoeff_den) != 0, bcoeff_num/bcoeff_den,NP.nan + NP.nan*1.j))
		elif TEST ==2 :
			return NP.array(m_window)
		else:
			return NP.where(abs(bcoeff_den) != 0, bcoeff_num/bcoeff_den,NP.nan + NP.nan*1.j)

	def compute_Kernels_Serial(self,ell,ellp,sMax,radial_order = 'All',fitParamsDate='20140620'):

		# Adjust the input
		ell = int(ell);ellp = int(ellp)
		sgrid = NP.arange(sMax+1)

		gamma_ell  = NP.sqrt((2*ell+1)/(4*NP.pi))
		gamma_ellp = NP.sqrt((2*ellp+1)/(4*NP.pi))
		gamma_s    = NP.sqrt((2*sgrid+1)/(4*NP.pi))
		Bm_lp_s_l  = 0.5*(1-(-1)**(sgrid+ell+ellp))*(-1) * NP.sqrt(ellp*(ellp+1)*ell*(ell+1)) * wigner3j_full(ell,ellp,0,-1,1,sMax) * (-1)**(sgrid+ell+ellp)

		Bm_lp_s_l  = NP.where(Bm_lp_s_l == 0,NP.nan,Bm_lp_s_l)

		with NP.load('/scratch/ch3246/Private/Mode_Coupling/Eigenfunctions/eigenfunctions_combined/eigs%02d.npz' % 0) as npzFile:
			rr    = npzFile['z']
			rho   = npzFile['rho']

		kernel_Toroidal = NP.zeros((sMax+1,31,len(rho)))*NP.nan

		if radial_order == 'All':
			DATA = NP.genfromtxt('/scratch/ch3246/project/OBS_DATA/hmi_v_sht_modes/hmi.v_sht_modes.%s_000000_TAI.0.300.138240.m10qr.7840.36' % fitParamsDate)
			inds = NP.arange(len(DATA))[DATA[:,0] == ell]
			indsp = NP.arange(len(DATA))[DATA[:,0] == ellp]
			ngrid = NP.union1d(inds,indsp)
			ngrid = NP.unique(DATA[ngrid,1])
		if len(ngrid) == 0:
			return kernel_Toroidal


		for nn in ngrid:
			# Load in the eigenfunctions
			with NP.load('/scratch/ch3246/Private/Mode_Coupling/Eigenfunctions/eigenfunctions_combined/eigs%02d.npz' % nn) as npzFile:
				Ugrid = npzFile['Xi_z'] # U is vertical eigenfunction
				Vgrid = npzFile['Xi_h'] # V is horizontal eigenfunction
				ells  = npzFile['ells']

			ind  = NP.argmin(abs(ells - ell ))
			indp = NP.argmin(abs(ells - ellp))

			U  = Ugrid[ind]
			V  = Vgrid[ind]
			Up = Ugrid[indp]
			Vp = Vgrid[indp]

			rT1 = Up*V + Vp*U - Up*U
			rT2 = rT1[None,:] - 0.5*(Vp*V)[None,:]*(ell*(ell+1) + ellp*(ellp+1) - sgrid[:,None]*(sgrid[:,None]+1))
			rT3 = 4*NP.pi*(rr*rho)[None,:]*gamma_ell*gamma_ell * rT2 * Bm_lp_s_l[:,None]
			kernel_Toroidal[:,int(nn),:] = rT3
		return NP.array(kernel_Toroidal)


	def skipDist_calc(self,freq,ell,Nskips_sphere=False):
		th = raypath(freq, ell)[1]
		if Nskips_sphere:
			return 360./(th[-1] - th[0])
		return th[-1] - th[0]

