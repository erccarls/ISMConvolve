import multiprocessing as mp
from functools import partial
import numpy as np
from numpy import *

# Load 3-D Gas Distributions
from scipy.interpolate import RegularGridInterpolator as rgi
import pyfits
hdulist_H2_PEB = pyfits.open('./mod-total-rev2int.fit.gz')
hdulist_HI_NS = pyfits.open('./HI_NS.fits.gz')
H2_PEB = hdulist_H2_PEB[0].data
HI_NS = hdulist_HI_NS[0].data

def __LOS_Gas_Grid(tNum, n_thread, l_max,b_max,res,H2,HI,H2_mult,r_min,r_max,z_step=0.02,H2_map='PEB'):
    """
    LOS Integration Kernel for Pohl, Englmaier, Bissantz 2008  (arXiv: 0712.4264)
    Grid_points are linearly interpolated in 3-dimensions.
    """    
    #==============================================
    # Integration Parameters
    #==============================================
    import tmp
    reload(tmp)
    func = tmp.func
    R_solar = 8.5 # Sun at 8.5 kpc
    kpc2cm  = 3.08568e21
    z_start,z_stop,z_step = max(R_solar-r_max,0),R_solar+r_max,z_step # los start,stop,step-size in kpc
    
    # distances along the LOS
    zz = np.linspace(start=z_start,stop=z_stop,num=int(np.ceil((z_stop-z_start)/z_step)))
    deg2rad = np.pi/180.
    pi = np.pi
    # List of lat/long to loop over.
    bb = np.linspace(-b_max+res/2.,b_max+res/2.,int(np.ceil(2*b_max/res)))*deg2rad
    # divide according to thread
    stride = len(bb)/n_thread
    if tNum == n_thread-1: bb = bb[tNum*(stride):]
    else: bb = bb[tNum*stride:(tNum+1)*stride]
        
    ll = np.linspace(-l_max+res/2.,l_max+res/2.,np.ceil(int(2*l_max/res)))*deg2rad
    
    # Master projected skymap 
    proj_skymap = np.zeros(shape=(len(bb),len(ll)))
    
    #Prepare the interpolator based on the image info
    grid_x = np.linspace(-14.95, 14.95,300)
    grid_z = np.linspace(-.4875, .4875,40)
    H2_PEB_Interp = rgi((grid_z,grid_x,grid_x),H2_PEB,fill_value=np.float32(0), method='linear',bounds_error=False)
    grid_x = np.linspace(-50, 50,501)
    grid_z = np.linspace(-2, 2,51)
    HI_NS_Interp = rgi((grid_z,grid_x,grid_x),HI_NS,fill_value=np.float32(0), method='linear',bounds_error=False)
    X_CO_MAX = 100*np.ones(len(zz))
    
    # Loop latitudes 
    for bi in range(len(bb)):
        # loop longitude
        for lj in range(len(ll)):
            los_sum=0.
            l,b = ll[lj], bb[bi]
            # z in cylindrical coords
            z = zz*sin(b) 
            x,y = -zz*cos(l)*cos(b)+R_solar, +zz*sin(l)*cos(b)
            r_2d = sqrt(x**2+y**2)
            
            if H2==True:
                # Call the interpolator on the current set of points
                if H2_map=='PEB':
                    X_CO = 3e19*np.minimum( exp(r_2d/5.3),X_CO_MAX)
                    los_sum += H2_mult*sum(func(x,y,z)*H2_PEB_Interp((z,-y,x)))*z_step*kpc2cm
                elif H2_map=='NS': 
                    pass 
                else: raise Exception("Invalid H2 map chosen")
                     
            if HI==True:
                los_sum += sum(func(x,y,z)*HI_NS_Interp((z,y,x)))*z_step*kpc2cm
              
            proj_skymap[bi,lj] = los_sum 
    return proj_skymap


def LOS_Gas_Grid(l_max, b_max, res, H2, HI, H2_mult, r_min, r_max, z_step=0.02,func='func = lambda x,y,z: 1.',H2_map='PEB'):
    
    # See LOS_Gas_Ferierre for better documentation 
    # Open a file tmp.py and write the passed function to it.  This is required because of 
    # some issues surrounding multithreading in python.
    
    f = open('tmp.py','wb')
    f.write(func)
    f.close()
    import tmp
    reload(tmp)

    n_threads = mp.cpu_count()
    kernel = partial(__LOS_Gas_Grid,
                     n_thread=n_threads, l_max=l_max, b_max=b_max,
                     res=res, H2=H2, HI=HI, H2_mult=H2_mult, 
                     r_min=r_min, r_max=r_max, z_step=z_step,H2_map=H2_map)
    
    p = mp.Pool(n_threads)
    slices = p.map(kernel,range(n_threads))
    p.close()

    proj_skymap = slices[0]
    for slice_ in slices[1:]:
        proj_skymap = np.append(proj_skymap,slice_,axis=0)
        
    return proj_skymap