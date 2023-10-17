import numpy  as NP
import scipy.interpolate as ITP
import matplotlib.pyplot as PLOT
from ..Common     import *
from ..Parameters import *

def get_arc_lonlat(longitude,latitude,distance,subtend_ang,relative_pos,npt):
# /* longitude,latitude: center position (in degrees) of the arc
#  * distance: the distance (in degrees) from the center to the arc
#  * subtend_ang: the angle (in degrees) subtended by the arc
#  * relative_pos: the position of the arc w.r.t. the center.
#  *   if the arc is to the north of the center, then set to 0; if left then
#  *   set 90; if south then set 180; if right then set 270.
#  */


    deg2rad=NP.pi/180.;
    # double x[3],y[3],z[3];
    x = NP.zeros(3);y = NP.zeros(3);z = NP.zeros(3)
    dist=distance*deg2rad;
    sub_ang=subtend_ang*deg2rad;
    ang0=relative_pos*deg2rad;
    y_rot_ang=NP.pi/2-latitude*deg2rad;
    z_rot_ang=-(NP.pi+longitude*deg2rad);
    arc_clat = dist
    lon      = dist
    clat     = dist;

    cos_y = cos(y_rot_ang);
    sin_y = sin(y_rot_ang);
    cos_z = cos(z_rot_ang);
    sin_z = sin(z_rot_ang);

    if npt < 1:
        raise Exception("number of output points must be > 0")

    arc_lon = []; arc_lat = []
    for i in range(npt):
        if npt == 1:
            lon = ang0
        else:
            lon = sub_ang*(float(i)/(npt-1)-0.5)+ang0;

        #     /* (lon,clat) -> (x,y,z) */
        x[0]=NP.sin(clat)*NP.cos(lon);
        y[0]=NP.sin(clat)*NP.sin(lon);
        z[0]=NP.cos(clat);
        
        #     /* rotate about y-axis */
        x[1]= x[0]*cos_y-z[0]*sin_y;
        y[1]= y[0];
        z[1]= x[0]*sin_y+z[0]*cos_y;

        #     /* rotate about z-axis */
        x[2]= x[1]*cos_z+y[1]*sin_z;
        y[2]=-x[1]*sin_z+y[1]*cos_z;
        z[2]= z[1];
    
    #     /* (x,y,z) -> (lon,clat) */
        if(z[2] >  1 and z[2] <  1+1e6):
	        z[2]=1;
        if(z[2] < -1 and z[2] > -1-1e6):
        	z[2]=-1;
        arc_clat=NP.arccos(z[2])/deg2rad;
        arc_lon.append(NP.arctan2(y[2],x[2])/deg2rad)
        arc_lat.append(90.-arc_clat)
    return NP.array(arc_lon),NP.array(arc_lat)


def get_npt_arc2arc(trvldist, subang=30, scale=0.6):
    """\
subang: subtended angle in degree
trvldist: separation distance in degree
scale: image scale in deg/pix
"""
    return NP.round(0.5*trvldist/scale * NP.deg2rad(subang)).astype(int)
