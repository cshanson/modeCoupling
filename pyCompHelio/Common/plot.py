import numpy                 as NP
import matplotlib.pyplot     as PLOT
import matplotlib.cm         as CM
import matplotlib.colors     as COLS
from   mpl_toolkits.mplot3d  import Axes3D
from   math                  import cos,sin
from   .misc                 import *

def plotNodal(data,nodalPoints,fileName=None,**plotArgs):

    x,y = nodalPoints.getCartesianCoords()

    if fileName is None or getExtension(fileName) != 'vtk':

      PLOT.figure()
      PLOT.scatter(x,y,c=data,**plotArgs)
      PLOT.ion()

      try:
        PLOT.show()
      except:
        pass

      if fileName is not None:
        PLOT.savefig(fileName)

    else:

      Npts = len(x)
      of   = open(fileName,'w')
      # Header 
      of.write('# vtk DataFile Version 3.0\n');
      of.write('%s\n'%cutExtension(fileName));
      of.write('ASCII\n');
      # Write points coordinates
      of.write('DATASET UNSTRUCTURED_GRID\n');
      of.write('POINTS %d double\n' % Npts);
      for xi,yi in zip(x,y):
        of.write('%1.16f 0. %1.16f \n' % (xi,yi))
      # Write data
      of.write('POINT_DATA %d\n'%Npts);
      if data.ndim == 1:
        of.write('SCALARS %s double\n'%cutExtension(fileName))
        of.write('LOOKUP_TABLE default\n')
        for d in data:
          of.write('%1.16f\n'%d)
        of.close()
      else:
        of.write('VECTORS %s double\n'%cutExtension(fileName))
        for i in range(data.shape[1]):
          of.write('%1.16f %1.16f %1.16f\n'%(data[0,i],data[1,i],data[2,i]))
        of.close()

def plotOnMeshGrid(data,points,NmeshGrid,isScalar,fileName=None,**plotArgs):
  ''' points is a list of coordinates either (2,Npoints) or (3,Npoints)
      Npoints is the shape of the meshgrid used to generate points
      (points is not given directly as a meshgrid to be able to plot 2D slices in 3D output '''

  dimPoints = points.shape[0]
  fig       = PLOT.figure()
  PLOT.ion()

  # Plots in 3D space
  if dimPoints == 3:

    # Scalar data on a surface (sphere or plane)
    if isScalar:
      X = points[0].reshape(NmeshGrid)
      Y = points[1].reshape(NmeshGrid)
      Z = points[2].reshape(NmeshGrid)

      ax    = fig.gca(projection='3d')
      # Normalize data and get colors
      ndata = data.reshape(NmeshGrid)
      if 'vmin' in plotArgs:
        vmin = plotArgs['vmin']
        ndata = vmin*(ndata<=vmin) + ndata*(ndata>vmin)
      if 'vmax' in plotArgs:
        vmax = plotArgs['vmax']
        ndata = vmax*(ndata>=vmax) + ndata*(ndata<vmax)
      ndata  = ndata/NP.amax(abs(ndata))     
      colors = PLOT.cm.jet(ndata)
 
      m = PLOT.cm.ScalarMappable(cmap=PLOT.cm.jet)
      m.set_array(ndata)

      cplt = ax.plot_surface(X,Y,Z,rstride=1,cstride=1,facecolors=colors, linewidth=0, antialiased=False, shade=False,**plotArgs)
      PLOT.colorbar(m)
      
    # quiver plots
    else:
      X = points[0]
      Y = points[1]
      Z = points[2]
      if data.shape[0] == 2: 
        w = NP.zeros(tuple(NmeshGrid))
      else:
        w = data[2]
      ax   = fig.gca(projection='3d')
      ax.quiver(X,Y,Z,data[0],data[1],w,**plotArgs)
 
    dimMax = max(max(NP.amax(abs(X)),NP.amax(abs(Y))),NP.amax(abs(Z)))
    ax.set_xlim([-dimMax,dimMax])
    ax.set_ylim([-dimMax,dimMax])
    ax.set_zlim([-dimMax,dimMax])



  # Plots in 2D space
  else:
 
    # Scalar data: pcolormesh
    if isScalar:
      X = points[0].reshape(NmeshGrid)
      Y = points[1].reshape(NmeshGrid)
      D = data.reshape(NmeshGrid)
      ax = fig.gca()
      ax.pcolormesh(X,Y,D,**plotArgs)
    # Vector data: quiver
    else:
      X = points[0]
      Y = points[1]
      ax = fig.gca()
      ax.quiver(X,Y,ndata[0],data[1],**plotArgs)
 
    dimMax = max(NP.amax(abs(X)),NP.amax(abs(Y)))
    ax.set_xlim([-dimMax,dimMax])
    ax.set_ylim([-dimMax,dimMax])

  try:
    PLOT.show()
  except:
    pass
  if fileName is not None:
    PLOT.savefig(fileName)
    
def plotOnMeshGridVTK(data,points,nMeshGrid,fileName,isScalar,fieldName='u'):
  ''' writes data defined on a meshgrid '''

  data = NP.nan_to_num(data)

  with open(fileName,'w') as OF:

    # Header
    OF.write('# vtk DataFile Version 3.0\n')
    OF.write('blah blah\n')
    OF.write('ASCII\n')

    # Points
    nMeshGrid = NP.array(nMeshGrid)
    nPts      = NP.prod(nMeshGrid) 
    OF.write('DATASET UNSTRUCTURED_GRID\n')
    OF.write('POINTS %d float\n'%nPts)
  
    for p in points.T:

      OF.write('%1.12g %1.12g'%(p[0],p[1]))
      if len(p) == 3:
        OF.write(' %1.12g\n'%p[2])
      else:
        OF.write(' 0.0\n')

    # Cells
    nCells = NP.prod(nMeshGrid-1)
    if len(nMeshGrid) == 2:
      nWords = 5*nCells
    elif len(nMeshGrid) == 3:
      nWords = 9*nCells

    OF.write('CELLS %d %d\n' %(nCells,nWords))

    if len(nMeshGrid) == 2:
      for i in range(nMeshGrid[0]-1):
        for j in range(nMeshGrid[1]-1):
            ID1 = i   *nMeshGrid[1] + j
            ID2 =(i+1)*nMeshGrid[1] + j
            ID3 =(i+1)*nMeshGrid[1] + j+1
            ID4 = i   *nMeshGrid[1] + j+1
            OF.write('4 %d %d %d %d\n' % (ID1,ID2,ID3,ID4))
      OF.write('CELL_TYPES %d\n' % nCells)
      for i in range(nCells):
        OF.write('9\n') # VTK_QUAD
    elif len(nMeshGrid) == 3:
      for i in range(nMeshGrid[0]-1):
        for j in range(nMeshGrid[1]-1):
          for k in range(nMeshGrid[2]-1):
            p   = nMeshGrid[2]*nMeshGrid[1]
            ID1 = i   *p + j   *nMeshGrid[2] + k
            ID2 = i   *p + j   *nMeshGrid[2] + k+1
            ID3 = i   *p +(j+1)*nMeshGrid[2] + k+1
            ID4 = i   *p +(j+1)*nMeshGrid[2] + k
            ID5 =(i+1)*p + j   *nMeshGrid[2] + k
            ID6 =(i+1)*p + j   *nMeshGrid[2] + k+1
            ID7 =(i+1)*p +(j+1)*nMeshGrid[2] + k+1
            ID8 =(i+1)*p +(j+1)*nMeshGrid[2] + k
            OF.write('8 %d %d %d %d %d %d %d %d\n' % (ID1,ID2,ID3,ID4,ID5,ID6,ID7,ID8))
      OF.write('CELL_TYPES %d\n' % nCells)
      for i in range(nCells):
        OF.write('12\n') # VTK_HEXAEDRON

    # Data
    OF.write('POINT_DATA %d\n' % nPts)
    if isScalar:
      OF.write('SCALARS %s float\n' % fieldName)
      OF.write('LOOKUP_TABLE default\n')
      for d in data.ravel():
        OF.write('%1.12g\n' % d)
    else:
      OF.write('VECTORS %s float\n' % fieldName)
      if data.shape[0] == 2:
        for d in NP.vstack([data[0].ravel(),data[1].ravel()]).T:
          OF.write('%1.12g %1.12g 0.0\n' % tuple(d))
      elif data.shape[0] == 3:
        for d in NP.vstack([data[0].ravel(),data[1].ravel(),data[2].ravel()]).T:
          OF.write('%1.12g %1.12g %1.12g\n' % tuple(d))


# =============================================================================
# Colormaps

def cMapB2R  (field, vRange=None, nColors=1000, whitePerCent=5.):

  maxi = NP.amax(field)
  mini = NP.amin(field)
  if vRange:
    mini = vRange[0]
    maxi = vRange[1]

  eps      = whitePerCent/100  
  ratioNeg = -mini/(maxi-mini) -eps
  ratioPos =  maxi/(maxi-mini) -eps
  colors1  = PLOT.cm.get_cmap('Blues_r')
  colors2  = PLOT.cm.get_cmap('Reds')

  symmetrize =(NP.abs(maxi+mini) / max(NP.abs(maxi), NP.abs(mini))) < 0.2 
  if symmetrize: # symmetrize if max \approx - min
    colors1 = colors1(NP.linspace(0.,1.,nColors))
    colors2 = colors2(NP.linspace(0.,1.,nColors))
    maxi    = NP.amax(NP.abs(field))
    mini    = -maxi
  else:
    Nm = int(NP.round(nColors*ratioNeg))
    Np = int(NP.round(nColors*ratioPos))
    if maxi > -mini:
      colors1 = colors1(NP.linspace(1.-ratioNeg,1.,Nm))
      colors2 = colors2(NP.linspace(0.         ,1.,Np))
    else:
      colors1 = colors1(NP.linspace(0.         ,1.,Nm))
      colors2 = colors2(NP.linspace(1.-ratioPos,1.,Np))

  white     = NP.ones((int(NP.round(eps*nColors)),4))
  colorList = NP.vstack((colors1, white, colors2))

  return  COLS.LinearSegmentedColormap.from_list('red_white_blue',colorList)
 
def cMapRYWCB(field, vRange=None, nColors=1000,whitePerCent=5.,\
              colorFrac=[0.25,0.25,0.25,0.25],reverse=False):

  cminInput = NP.amin(field)
  cmaxInput = NP.amax(field)
  if vRange:
    cminInput = vRange[0]
    cmaxInput = vRange[1]

  if NP.isnan(cminInput) or NP.isnan(cmaxInput):
    raise Exception('Error cmax or cmin = Nan')

  nColorsPerSide = nColors*NP.array(colorFrac)
  colorNumtotal  = nColors
  eps     = whitePerCent/100 
  redYel = None
  yelWht = None
  whtCyn = None
  cynBlu = None

  for i in range(nColors):
    if redYel is None:
      redYel = [[1.,0.,0.]]
      yelWht = [[1.,1.,0.]]
      whtCyn = [[1.,1.,1.]]
      cynBlu = [[0.,1.,1.]]
    else:
      if i < nColorsPerSide[0]:
        redYel = NP.append(redYel,[[1.,(nColorsPerSide[0]+i)/float(nColorsPerSide[0])-1,0.]],axis=0)
      if i < nColorsPerSide[1]:
        yelWht = NP.append(yelWht,[[1.,1.,(nColorsPerSide[1]+i)/float(nColorsPerSide[1])-1]],axis=0)
      if i < nColorsPerSide[2]:
        whtCyn = NP.append(whtCyn,[[(nColorsPerSide[2]-i)/float(nColorsPerSide[2]),1.,1.  ]],axis=0)
      if i < nColorsPerSide[3]:
        cynBlu = NP.append(cynBlu,[[0.,(nColorsPerSide[3]-i)/float(nColorsPerSide[3]),1.  ]],axis=0)

  whiteMiddle = NP.ones((int(NP.round(eps*nColors)),3))
  ucmap       = NP.concatenate((redYel,yelWht,whiteMiddle,whtCyn,cynBlu,[[0.,0.,1.]]),axis=0)

  if not reverse:
    ucmap = NP.flipud(ucmap)

  return COLS.ListedColormap(ucmap)

def blueToRed(field, Ncolors = 1000, whitePerCent=5.):

  maxi = NP.amax(field)
  mini = NP.amin(field)
  eps = whitePerCent / 100  
  ratioNeg = -mini / (maxi-mini) - eps
  ratioPos = maxi / (maxi-mini) - eps
  colors1 = CM.get_cmap('Blues_r')
  colors2 = CM.get_cmap('Reds')

  symmetrize =(NP.abs(maxi+mini) / max(NP.abs(maxi), NP.abs(mini))) < 0.2 
  if symmetrize: # symmetrize if max \approx - min
    colors1 = colors1(NP.linspace(0.,1., Ncolors))
    colors2 = colors2(NP.linspace(0.,1., Ncolors))
    maxi = NP.amax(NP.abs(field))
    mini = -maxi
  else:
    if maxi > -mini:
      colors1 = colors1(NP.linspace(1. - ratioNeg,1., int(NP.round(Ncolors * ratioNeg))))
      colors2 = colors2(NP.linspace(0.,1., int(NP.round(Ncolors * ratioPos))))
    else:
      colors1 = colors1(NP.linspace(0.,1., int(NP.round(Ncolors * ratioNeg))))
      colors2 = colors2(NP.linspace(1.-ratioPos, 1., int(NP.round(Ncolors * ratioPos))))

  white     = NP.ones((int(NP.round(eps * Ncolors)),4))
  colorList = NP.vstack((colors1, white, colors2))
  mymap     = COLS.LinearSegmentedColormap.from_list('red_white_blue', colorList)
 
  return mymap, mini, maxi

def heatCM():
  p = os.getcwd().split('mps_montjoie')[0]+'mps_montjoie/'
  colors = NP.loadtxt(p+'/pyCompHelio/Common/gist_heat.txt')
  return COLS.ListedColormap(colors,'heatCM')


