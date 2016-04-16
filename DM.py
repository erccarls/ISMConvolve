########################################
# gas_integrator.py
# Written by Eric Carlson (erccarls@ucsc.edu) 
# August 2014
# 
# This module contains interfaces to integration routines for various gas models and free
# electrons.  The basic structure is the following.  For each model, there is file containing
# the kernel, suffixed by "_thread", followed by the main routine which slices the
# region of interest and distributes the integration to a pool of worker threads.
# Finally, there are two routines in this file: GenGasSkymapAnnulus and ConvolveGasSkymap.  The
# former integrates galactocentric rings and calls the various methods, while the
# latter is the primary interface called by the end user which also takes care of
# remormalization of gas maps to the various survey column densities.

from numpy import *
import numpy as np
import multiprocessing as mp
from functools import partial


def __LOS_DM(tNum, n_thread, l_max,b_max,res,r_min,r_max,z_step=0.02):
    """
    LOS Integration Kernel for two passed distributions
    """    
    #==============================================
    # Integration Parameters
    #==============================================
    import tmp,tmp2
    
    reload(tmp)
    reload(tmp2)
    func1 = tmp.func
    func2 = tmp2.func
    
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
    
    
    # Loop latitudes 
    for bi in range(len(bb)):
        # loop longitude
        for lj in range(len(ll)):
            los_sum=0.
            l,b = ll[lj], bb[bi]
            # z in cylindrical coords
            z = zz*sin(b) 
            x,y = -zz*cos(l)*cos(b)+R_solar, +zz*sin(l)*cos(b)
            #r_2d = sqrt(x**2+y**2)
            
            los_sum += sum(func1(x,y,z)*func2(x,y,z))
              
            proj_skymap[bi,lj] = los_sum*z_step*kpc2cm
    return proj_skymap


def LOS_DM(l_max, b_max, res, z_step=0.02,func1='func = lambda x,y,z: 1.',func2='func = lambda x,y,z: 1.'):
    
    # See LOS_Gas_Ferierre for better documentation 
    # Open a file tmp.py and write the passed function to it.  This is required because of 
    # some issues surrounding multithreading in python.
    
    # Write func to a file so it is importable by child threads
    f = open('tmp.py','wb')
    f.write(func1)
    f.close()

    f = open('tmp2.py','wb')
    f.write(func2)
    f.close()
    
    n_threads = mp.cpu_count()

    kernel = partial(__LOS_DM,
                     n_thread=n_threads, l_max=l_max, b_max=b_max,
                     res=res, r_min=0, r_max=30, z_step=z_step)
    
    p = mp.Pool(n_threads)
    slices = p.map(kernel,range(n_threads))
    p.close()

    proj_skymap = slices[0]
    for slice_ in slices[1:]:
        proj_skymap = np.append(proj_skymap,slice_,axis=0)
        
    return proj_skymap



