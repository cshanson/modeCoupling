from   ..Common          import *
import ctypes
from numpy.ctypeslib import ndpointer
from .SaveSparse import *

#---------------------------------------------


def BuildGauntTables_Axisymmetric(lMax,l3_min,l3_max,m,outDir,clearFirst = False):
	if clearFirst:
		os.system('rm -rf %s' % outDir)
	mkdir_p(outDir)
	mkdir_p(outDir +'/RAW/Ylm/');mkdir_p(outDir +'/RAW/Plm_Sum1/');mkdir_p(outDir +'/RAW/Plm_Sum-1/')

	if getClusterName().upper() == 'CONDOR':
		Gaunt_exe = pathToMPS() + '/bin/HelioCluster/computeKernel1D.x'
	elif getClusterName().upper() == 'SLURM':
		Gaunt_exe = pathToMPS() + '/bin/SlurmCluster/computeKernel1D.x'

	tini = time.time()

	os.system(Gaunt_exe + ' Ylm %i %i %i %i %i %i %i %s' % (lMax,lMax,l3_min,l3_max,m,-m,0,outDir + '/RAW/Ylm/'))
	os.system(Gaunt_exe + ' Plm %i %i %i %i %i %i %i %s' % (lMax,lMax,l3_min,l3_max,m,-m+1,0,outDir + '/RAW/Plm_Sum1/'))
	os.system(Gaunt_exe + ' Plm %i %i %i %i %i %i %i %s' % (lMax,lMax,l3_min,l3_max,m,-m-1,0,outDir + '/RAW/Plm_Sum-1/'))

	return (time.time()-tini)

def wigner3j(l1,l2,l3,m1,m2,m3):
	if getClusterName().upper() == 'CONDOR':
		Gaunt_exe = pathToMPS() + '/bin/HelioCluster/gaunt.x'
	elif getClusterName().upper() == 'SLURM':
		Gaunt_exe = pathToMPS() + '/bin/DalmaCluster/gaunt.x'	

	cmd = Gaunt_exe + ' W3 %i %i %i %i %i %i' % (l1,l2,l3,m1,m2,m3)


	subP = subprocess.Popen(cmd,\
                              shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
	out,err  = subP.communicate()
	return float(out.decode('utf-8'))

def wigner3j_full(l2,l3,m1,m2,m3,lMax):
	if getClusterName().upper() == 'CONDOR':
		Gaunt_exe = pathToMPS() + '/bin/HelioCluster/gaunt.x'
	elif getClusterName().upper() == 'SLURM':
		Gaunt_exe = pathToMPS() + '/bin/DalmaCluster/gaunt.x'	

	cmd = Gaunt_exe + ' W3_full %i %i %i %i %i %i' % (l2,l3,m1,m2,m3,lMax)


	subP = subprocess.Popen(cmd,\
                              shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
	out,err  = subP.communicate()
	return NP.array(out.decode('utf-8').split(),dtype=float)


def ProcessGauntTables_Axisymmetric(lMax,l3,inDir):
	tini = time.time()
	Ylm_Big       = scipy.sparse.lil_matrix((2*lMax+1,(lMax+1)**2))
	Plm_Sum1_Big  = scipy.sparse.lil_matrix((2*lMax+1,(lMax+1)**2))
	Plm_SumM1_Big = scipy.sparse.lil_matrix((2*lMax+1,(lMax+1)**2))

	for m in range(-lMax,lMax+1):

		Ylm_TablePath = inDir + '/RAW/Ylm/l3_%i/table_m%i_sumM0.txt' % (l3,abs(m))
		Plm_Sum1_TablePath = inDir + '/RAW/Plm_Sum1/l3_%i/table_m%i_sumM1.txt' % (l3,abs(m))
		Plm_SumM1_TablePath = inDir + '/RAW/Plm_Sum-1/l3_%i/table_m%i_sumM-1.txt' % (l3,abs(m))
		if not os.path.isfile(Ylm_TablePath):
			raise Exception('Ylm Tables need computing')
		elif not os.path.isfile(Plm_Sum1_TablePath):
			raise Exception('Plm Tables (sum m = 1) need computing')
		elif not os.path.isfile(Plm_SumM1_TablePath):
			raise Exception('Plm Tables (sum m = -1) need computing')

		Ylm_Table         = loadMat(Ylm_TablePath)
		Plm_Sum1_Table_t  = loadMat(Plm_Sum1_TablePath)
		Plm_SumM1_Table_t = loadMat(Plm_SumM1_TablePath)

		if Ylm_Table is None:
			Ylm_Table         = scipy.sparse.coo_matrix((lMax+1,lMax+1))
		if Plm_Sum1_Table_t is None:
			Plm_Sum1_Table_t  = scipy.sparse.coo_matrix((lMax+1,lMax+1))
		if Plm_SumM1_Table_t is None:
			Plm_SumM1_Table_t = scipy.sparse.coo_matrix((lMax+1,lMax+1))


		Ylm_Table         = scipy.sparse.csr_matrix(Ylm_Table)
		Plm_Sum1_Table_t  = scipy.sparse.csr_matrix(Plm_Sum1_Table_t)
		Plm_SumM1_Table_t = scipy.sparse.csr_matrix(Plm_SumM1_Table_t)

		# utilize symmetries
		if NP.sign(m) == -1:
			Plm_Sum1_Table  = -Plm_SumM1_Table_t; 
			Plm_SumM1_Table = -Plm_Sum1_Table_t;
		else:
			Plm_Sum1_Table  = Plm_Sum1_Table_t
			Plm_SumM1_Table = Plm_SumM1_Table_t


		if abs(m) < lMax:
			if NP.sign(-(m+1)) == -1:
				gm1M = -scipy.sparse.csr_matrix(loadMat(inDir + '/RAW/Plm_Sum1/l3_%i/table_m%i_sumM1.txt' % (l3,abs(m+1))))
			else:
				gm1M = scipy.sparse.csr_matrix(loadMat(inDir + '/RAW/Plm_Sum-1/l3_%i/table_m%i_sumM-1.txt' % (l3,abs(m+1))))
			if NP.sign(-(m-1)) == -1:
				gp1P = -scipy.sparse.csr_matrix(loadMat(inDir + '/RAW/Plm_Sum-1/l3_%i/table_m%i_sumM-1.txt' % (l3,abs(m-1))))
			else:
				gp1P = scipy.sparse.csr_matrix(loadMat(inDir + '/RAW/Plm_Sum1/l3_%i/table_m%i_sumM1.txt' % (l3,abs(m-1))))
		else:
			gm1M = scipy.sparse.csr_matrix((lMax+1,lMax+1))
			gp1P = scipy.sparse.csr_matrix((lMax+1,lMax+1))

		Plm_SumM1_Table = scipy.sparse.tril(Plm_SumM1_Table) + scipy.sparse.tril(gm1M,-1).T
		Plm_Sum1_Table  = scipy.sparse.tril(Plm_Sum1_Table) + scipy.sparse.tril(gp1P,-1).T

		Ylm_Big[m+lMax] = Ylm_Table.toarray().reshape(-1)
		Plm_SumM1_Big[m+lMax] = Plm_SumM1_Table.toarray().reshape(-1)
		Plm_Sum1_Big[m+lMax] = Plm_Sum1_Table.toarray().reshape(-1)

	l3Folder = '/l3_%i_%i/' % (int(l3)/NfilesPerDir*NfilesPerDir,int(l3)/NfilesPerDir*NfilesPerDir + NfilesPerDir-1)

	mkdir_p(inDir +'/Processed/Ylm/' + l3Folder);mkdir_p(inDir +'/Processed/Plm_Sum1/' +l3Folder);mkdir_p(inDir +'/Processed/Plm_Sum-1/' +l3Folder)
	save_sparse_npz(inDir +'/Processed/Ylm/%s/ProcessedTable_l3_%i_sumM0.npz' % (l3Folder,l3),scipy.sparse.csr_matrix(Ylm_Big).T)
	save_sparse_npz(inDir +'/Processed/Plm_Sum-1/%s/ProcessedTable_l3_%i_sumM-1.npz' % (l3Folder,l3),scipy.sparse.csr_matrix(Plm_SumM1_Big).T)
	save_sparse_npz(inDir +'/Processed/Plm_Sum1/%s/ProcessedTable_l3_%i_sumM1.npz' % (l3Folder,l3),scipy.sparse.csr_matrix(Plm_Sum1_Big).T)

	return (time.time()-tini)

def LoadGauntTables_Axisymmetric(l3,TableDirectory):
	l3Folder = '/l3_%i_%i/' % (int(l3)/NfilesPerDir*NfilesPerDir,int(l3)/NfilesPerDir*NfilesPerDir + NfilesPerDir-1)
	return load_sparse_npz(TableDirectory +'/Processed/Ylm/%s/ProcessedTable_l3_%i_sumM0.npz' % (l3Folder,l3)),\
	       load_sparse_npz(TableDirectory +'/Processed/Plm_Sum-1/%s/ProcessedTable_l3_%i_sumM-1.npz' % (l3Folder,l3)),\
	       load_sparse_npz(TableDirectory +'/Processed/Plm_Sum1/%s/ProcessedTable_l3_%i_sumM1.npz' % (l3Folder,l3))



# _gauntCode = ctypes.CDLL(pathToMPS() + '/bin/HelioCluster/gaunt.so')

# gauntPlm = _gauntCode.generalGaunt
# gauntPlm.argtypes = (ctypes.c_int,ctypes.c_int,\
# 	              ctypes.c_double,ctypes.c_double,\
# 	              ctypes.c_double,ctypes.c_double,
# 	              ndpointer(dtype='f8', ndim=2, flags='aligned, c_contiguous'))
# # gauntPlm.restype = ctypes.c_double

# gauntYlm = _gauntCode.gaunt
# gauntYlm.argtypes = (ctypes.c_double,ctypes.c_double,\
# 	              ctypes.c_double,ctypes.c_double)
# gauntYlm.restype = ctypes.c_double


# # gauntPlmTable = _gauntCode.generalGauntTable
# # gauntPlmTable.argtypes = (ctypes.c_int,ctypes.c_int,\
# # 	              ctypes.c_int,ctypes.c_double,\
# # 	              ctypes.c_double,ctypes.c_double,\
# # 	              ndpointer(dtype='f8', ndim=1, flags='aligned, c_contiguous'))

# gauntYlmTable = _gauntCode.gauntTable
# gauntYlmTable.argtypes = (ctypes.c_int,ctypes.c_double,\
	              # ctypes.c_double,ctypes.c_double,ctypes.c_double,\
	              # ndpointer(dtype='f8', ndim=1, flags='aligned, c_contiguous'))


# def BuildGauntTables(lMax,lK,outDir,Mtest = None):
#   # routine to build the Gaunt Tables

#   # make outDir
#   mkdir_p(outDir)
#   # Check if File exists
#   fileName = outDir + '/gauntTables_lK_%i_lMax_%i.npz' % (lK,lMax)

#   MMax = lMax+1
#   # modify the maximum M for testing
#   if Mtest is not None:
#     MMax = Mtest

#   # if file exists, load it and check how many m have been computed thus far
#   if os.path.isfile(fileName):
#     G = load_sparse_npz(fileName)
#     G = G.toarray().reshape(3,lMax+1,lMax+1,-1)
#     if G.shape[-1] >= lMax+1:
#       # if all m computed, then finish
#       return
#     else:
#       # if not all the m are computed then we will find out how many and rearrange for further computation
#       print 'found in complete table, continuing from last m = %i' % G.shape[-1]
#       Gtmp = NP.moveaxis(G,-1,0)
#       G = []
#       for i in range(Gtmp.shape[0]):
#         G.append(Gtmp[i])
#       Mrange = NP.arange(Gtmp.shape[0],MMax)
#       del Gtmp
#   else:
#     # if file doesn't exist then we start from the beginning
#     G = []
#     Mrange = NP.arange(0,MMax)

#   # Initialise the arrays to be filled with the c++ output
#   gg = NP.empty((lMax+1,lMax+1))
#   g  = NP.empty(lMax+1)

#   for m in Mrange:
#     # initialise the gaunt tables
#     # have to allocate into array otherwise it is symbolic link
#     # print 'Computing table for m = %i' % m
#     sys.stdout.write("Computing table for m = %i \n" % m)
#     gauntCoeff = NP.zeros((lMax+1,lMax+1),complex)
#     gauntCoeffPm1t1 = NP.zeros((lMax+1,lMax+1),complex)
#     gauntCoeffPp1t1 = NP.zeros((lMax+1,lMax+1),complex)

#     for l2 in range(0,lMax+1):
#       gauntYlmTable(lMax,l2,lK,m,g)
#       gauntCoeff[:,l2] = g
#     gauntCoeff = NP.array(gauntCoeff)
#     gauntCoeff = NP.tril(gauntCoeff)

#     gauntPlm(lMax,lMax,lK,m  ,-m-1,0,gg)
#     gauntCoeffPm1t1[:lMax+1,:lMax+1] = gg 
#     gauntPlm(lMax,lMax,lK,m  ,-m+1,0,gg)
#     gauntCoeffPp1t1[:lMax+1,:lMax+1] = gg

#     # append to total table output and ensure they are real
#     G.append(solarFFT.testRealFFT([gauntCoeff,gauntCoeffPm1t1,gauntCoeffPp1t1]))

#     Gsparse = scipy.sparse.csr_matrix(NP.moveaxis(G,0,-1).reshape(3,-1))

#     # save file, if m < lMax will be a temporary file
#     save_sparse_npz(fileName,Gsparse)

#   if Mtest is None:
# 	  print 'All m computed, utilizing symmetries'
# 	  # Once completely built we utilize the symmetries and build the -m
# 	  G = gauntSymmetries(lMax,NP.moveaxis(G,0,-1))
# 	  Gsparse = scipy.sparse.csr_matrix(G.reshape(3,-1))
# 	  save_sparse_npz(fileName,Gsparse)



# def gauntSymmetries(lMax,gauntTable):
# 	g,gm1,gp1 = gauntTable
# 	g = NP.concatenate((g[...,1:][...,::-1],g),axis=-1)
# 	gm1t = NP.concatenate((-gp1[...,1:][...,::-1],gm1),axis=-1)
# 	gp1t = NP.concatenate((-gm1[...,1:][...,::-1],gp1),axis=-1)
# 	gp1 = gp1t;gm1 = gm1t
# 	# del gm1t; del gp1t

# 	for m in range(-lMax,lMax+1):
# 		g[:,:,m+lMax] = NP.tril(g[...,m+lMax]) + NP.tril(g[...,m+lMax],-1).T
# 		if m != -lMax and m != lMax:
# 			gm1[:,:,m+lMax] = NP.tril(gm1t[...,m+lMax]) + NP.tril(gm1t[...,-(m+1)+lMax],-1).T 
# 			gp1[:,:,m+lMax] = NP.tril(gp1t[...,m+lMax]) + NP.tril(gp1t[...,-(m-1)+lMax],-1).T 
# 	return solarFFT.testRealFFT([g,gm1,gp1])



def reshape_sparse(a, shape,ReturnType = 'coo'):
    """Reshape the sparse matrix `a`.
    https://stackoverflow.com/questions/16511879/reshape-sparse-matrix-efficiently-python-scipy-0-12
    Returns a coo_matrix with shape `shape`.
    can return ReturnType sparse matrix
    """
    if not hasattr(shape, '__len__') or len(shape) != 2:
        raise ValueError('`shape` must be a sequence of two integers')

    c = a.tocoo()
    nrows, ncols = c.shape
    size = nrows * ncols

    new_size =  shape[0] * shape[1]
    if new_size != size:
        raise ValueError('total size of new array must be unchanged')

    flat_indices = ncols * c.row + c.col
    new_row, new_col = divmod(flat_indices, shape[1])

    b = scipy.sparse.coo_matrix((c.data, (new_row, new_col)), shape=shape)

    if ReturnType.upper() == 'CSR':
        b = scipy.sparse.csr_matrix(b)
    elif ReturnType.upper() == 'CSC':
        b = scipy.sparse.csc_matrix(b)
    return b