import numpy        as NP
import itertools    as IT
import scipy.sparse as SP
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class interpGrid(object):

  def __init__ (self,coords,method='linear',\
                fillOutside=True,fillWithZeros=True):
    ''' Interpolation on a n-dimensionnal grid
      
      - If fillOutside is set to False, interpolation will stop if 
        a point is outside the grid

      - Coords is a tuple of 1D arrays, each one positionning the points 
        in one grid dimension
    '''

    self.coords         = coords
    self.ndims          = len(coords)
    self.gridShape      = [len(i) for i in coords]
    self.fillOutside    = fillOutside
    if not method in ['nearest','linear']:
      raise NotImplementedError('Method %s not defined' % method)
    self.method         = method
    self.fillValue      = NP.nan
    self.fillWithZeros  = fillWithZeros
    self.computedMatrix = False

  def __str__(self):

    print( "Interpolation grid :")
    print( "  - method : ", self.method)
    print( "  - grid :", self.gridShape)

  def __call__(self,data,ncoords=None):
    ''' Returns the interpolated values at points self.ncoords
        where data are the values of the function at grid points
    '''

    # check data
    data = NP.asarray(data)
    array_ok = all([ a >= b for a,b in zip(data.shape,self.gridShape)])
    if not array_ok: 
      raise ValueError("Array of given data %s not large enough to match interpolation grid %s." % (str(data.shape),str(self.gridShape)))

    # compute matrix ?
    if not ncoords is None:
      self.setNewCoords(ncoords)
    if (not self.computedMatrix) and (ncoords is None):
      raise Exception("No new set of coordinates was given previously or at function call.") 

    # keep subarray of data and mutitply by A
    sl = [slice(None,m) for m in self.gridShape]
    if not self.fillWithZeros:
      return self.A.dot(data[sl].ravel())
    else:
      return self.A.dot(data[sl].ravel())*self.isOut

  def setNewCoords(self,ncoords):
    ''' Compute the interpolation matrix for points at ncoords.
        New coordinates should be given as a tuple of 1D arrays : ([x0,...,xNpts],[y0,...,yNpts],...)
        meshgrid results have to be expanded
    '''

    # check given points and cut if needed
    if len(ncoords) < self.ndims:
      raise ValueError("Given coordinates do not match grid dimension: %dD coords, %dD grid" % (len(ncoords),self.ndims))
    ncoords = ncoords[0:self.ndims]
    Npoints = NP.min ([len(i) for i in ncoords])
    Ngrid   = NP.prod([len(i) for i in self.coords])

    # check boundary points
    if not self.fillOutside:
      for i in range(self.ndims):
        if not (NP.all(ncoords[i] <= self.coords[i][0]) and NP.all(ncoords[i] >= self.coords[i][-1]) ):
          raise ValueError("A point (at least) is outside grid range in dimension %d." %i)

    # fill with zeros
    if self.fillWithZeros:
      m = [NP.min(i) for i in self.coords]
      M = [NP.max(i) for i in self.coords]
      self.isOut  = NP.zeros((self.ndims,len(ncoords[0])),dtype=bool)
      for j in range(0,self.ndims):
        self.isOut[j] = [m[j]<=i<=M[j] for i in ncoords[j]]
      self.isOut = NP.asarray(self.isOut)
      self.isOut = NP.prod(self.isOut,axis=0)

    # get position of points in grid
    indices    = []
    distances  = []     
    isOutside = NP.zeros(Npoints,dtype=bool)

    for points,gridPoints in zip(ncoords,self.coords):
      L       = len(gridPoints)
      index   = NP.searchsorted(gridPoints,points)-1
      index[index<0  ] = 0
      index[index>L-2] = L-2
      distance         = (points-gridPoints[index])/(gridPoints[index+1]-gridPoints[index])
      isOutside += points < gridPoints[0]
      isOutside += points > gridPoints[-1]

      indices.append(index)
      distances.append(distance)

    self.A = SP.lil_matrix((Npoints,Ngrid))

    # set interpolation coefficients
    if   self.method=='linear':
      self.setCoeffLinear(indices,distances)
    elif self.method=='nearest':
      self.setCoeffNearest(indices,distances)

    self.A.tocsr()
    self.computedMatrix = True    

  def setCoeffNearest(self,indices,distances):
    ''' sets the matrix coefficient corresponding to the closest point to 1
    '''
    ires = []
    for index,dist in zip(indices,distances):
      ires.append( where(dist<0.5,index,index+1) )
    
    self.A[NP.arange(len(indices[0])),self.gridToMonoD(ires)] = 1.0

  def setCoeffLinear(self,indices,distances):
    ''' linear interpolation:
        sets the coefficients dx,1-dx in each dimension
    ''' 

    if not self.ndims in [2,3]:
    # 2D and 3D are not so long to develop in the code
    # and much more efficient than doing the cartesian product (?)

      sommets = IT.product(*[[i,i+1] for i in indices])

      # loop through all the corners of the n-dim cell
      count = 0
      for sommet in sommets:
        count += 1
        weight = 1
        for sdim, idim, hdim in zip(sommet,indices,distances):
          weight *= NP.where(sdim==idim,1.0-hdim,hdim)
      
        self.A[NP.arange(len(indices[0])),self.gridToMonoD(sommet)] = weight

    elif self.ndims == 2:
      w0 = (1.0-distances[0])*(1.0-distances[1]) # i    j
      w1 =      distances[0] *(1.0-distances[1]) #(i+1) j
      w2 = (1.0-distances[0])*     distances[1]  # i   (j+1)
      w3 =      distances[0] *     distances[1]  #(i+1)(j+1)

      i0 = self.gridToMonoD((indices[0]  ,indices[1]  )) 
      i1 = self.gridToMonoD((indices[0]+1,indices[1]  )) 
      i2 = self.gridToMonoD((indices[0]  ,indices[1]+1)) 
      i3 = self.gridToMonoD((indices[0]+1,indices[1]+1)) 

      self.A[NP.arange(len(indices[0])),i0] = w0
      self.A[NP.arange(len(indices[0])),i1] = w1
      self.A[NP.arange(len(indices[0])),i2] = w2
      self.A[NP.arange(len(indices[0])),i3] = w3

    else: #3D

      i0 = self.gridToMonoD((indices[0]  ,indices[1]  ,indices[2]  )) 
      i1 = self.gridToMonoD((indices[0]+1,indices[1]  ,indices[2]  )) 
      i2 = self.gridToMonoD((indices[0]  ,indices[1]+1,indices[2]  )) 
      i3 = self.gridToMonoD((indices[0]  ,indices[1]  ,indices[2]+1)) 
      i4 = self.gridToMonoD((indices[0]+1,indices[1]+1,indices[2]  )) 
      i5 = self.gridToMonoD((indices[0]  ,indices[1]+1,indices[2]+1)) 
      i6 = self.gridToMonoD((indices[0]+1,indices[1]  ,indices[2]+1)) 
      i7 = self.gridToMonoD((indices[0]+1,indices[1]+1,indices[2]+1)) 

      w0 = (1.0-distances[0])*(1.0-distances[1])*(1.0-distances[2])
      w1 =      distances[0] *(1.0-distances[1])*(1.0-distances[2])
      w2 = (1.0-distances[0])*     distances[1] *(1.0-distances[2])
      w3 = (1.0-distances[0])*(1.0-distances[1])*     distances[2] 
      w4 =      distances[0] *     distances[1] *(1.0-distances[2])
      w5 = (1.0-distances[0])*     distances[1] *     distances[2] 
      w6 =      distances[0] *(1.0-distances[1])*     distances[2] 
      w7 =      distances[0] *     distances[1] *     distances[2] 

      self.A[NP.arange(len(indices[0])),i0] = w0
      self.A[NP.arange(len(indices[0])),i1] = w1
      self.A[NP.arange(len(indices[0])),i2] = w2
      self.A[NP.arange(len(indices[0])),i3] = w3
      self.A[NP.arange(len(indices[0])),i4] = w4
      self.A[NP.arange(len(indices[0])),i5] = w5
      self.A[NP.arange(len(indices[0])),i6] = w6
      self.A[NP.arange(len(indices[0])),i7] = w7

  def gridToMonoD(self,multi_index,n=0):
    ''' conversion between multiD and 1D indices
    '''
    if n==self.ndims-1:
      return multi_index[0]
    else:
      return multi_index[0]*NP.prod(self.gridShape[n+1:]) + self.gridToMonoD(multi_index[1:],n+1)

  def monoDToGrid(self,index):
    ''' conversion between multiD and 1D indices
    '''
    idxtmp = index
    mindex = []
    for i in range(self.ndims):
      mindex.append(idxtmp % self.gridShape[self.ndims-1-i])
      idxtmp = idxtmp // self.gridShape[self.ndims-1-i]
    return mindex[::-1]


