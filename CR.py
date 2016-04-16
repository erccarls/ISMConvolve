import multiprocessing as mp
from functools import partial
import numpy as np
from numpy import *

import pyfits
hdu_Lorimer = pyfits.open('nuclei_full_54_Lorimer_z4kpc_R20kpc_Ts150K_EBV5mag.gz')
hdu_SNR = pyfits.open('nuclei_full_54_SNR_z4kpc_R20kpc_Ts150K_EBV5mag.gz')
hdu_Yusifov = pyfits.open('nuclei_full_54_Yusifov_z4kpc_R20kpc_Ts150K_EBV5mag.gz')
hdu_OBstars = pyfits.open('nuclei_full_54_OBstars_z4kpc_R20kpc_Ts150K_EBV5mag.gz')



from scipy.interpolate import RegularGridInterpolator as rgi
def GetInterpolater(hdulist, E,species):
    "Returns a regulargridinterpolator object which can be called as interp((r,z),method='linear')."
    
    # Returns Ebin nor nuclei output file assuming E=10*1.2^i in Kin. Energy in MeV
    EBin = int(np.round(np.log(E/10.)/np.log(1.2)))
    
    if species == 'proton':
        "Primary + Secondary Protons."
        vals = np.sum(hdulist[0].data[5:7,EBin],axis=0)
    if species == 'antiproton':
        "Secondary + Tertiary Antiproton"
        vals = np.sum(hdulist[0].data[3:5,EBin],axis=0)
    if species == 'electron':
        "Primary + Secondary Electrons."
        vals = hdulist[0].data[0,EBin]
    if species == 'positron':
        "Secondary + Tertiary Positron"
        vals = np.sum(hdulist[0].data[1:3,EBin],axis=0)
        
    # Build Grid for interpolator based on FITS header
    height = -hdulist[0].header['CRVAL2']
    radius = hdulist[0].header['CDELT1']*hdulist[0].header['NAXIS1']
    grid_r = np.linspace(0,radius,hdulist[0].header['NAXIS1'])
    grid_z = np.linspace(-height, height,hdulist[0].header['NAXIS2'])
    interp = rgi((grid_r,grid_z),np.transpose(vals),fill_value=np.float32(0), method='linear',bounds_error=False)
    return interp







#=================================================
# This section integrates regions R>3kpc from GC
#=================================================
def __LOS_CR_Thread(tNum, n_thread, l_max, b_max,res,r_min, r_max, z_step, interpolator):
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
    
    # Loop latitudes 
    for bi in range(len(bb)):
        # loop longitude
        for lj in range(len(ll)):
            los_sum=0.
            l,b = ll[lj], bb[bi]
            # z in cylindrical coords
            z = zz*sin(b) 
            x,y = -zz*cos(l)*cos(b)+R_solar, zz*sin(l)*cos(b)
            
            r_2d = sqrt(x**2+y**2)            
            
            # Multiply by passed function and call CR interpolator
            los_sum += sum(func(x,y,z)*interpolator((r_2d,z))*z_step)
            
            proj_skymap[bi,lj] = los_sum 
    return proj_skymap


def ConvolveCR(l_max, b_max, res,  species, E, r_min=0., r_max=20.,source_dist='Lorimer',z_step=0.02,func='func = lambda x,y,z: 1.'):
    '''
    Multithreaded convolution over cosmic ray densities.  Densities are taken from Fermi Diffuse 
    Models for several CR source distributions and propagated using galprop.  The source distribution
    is allowed to change, but the models are based on Z=4kpc halo, R=20 kpc, T_S=150K, EBV5Mag. Other than perhaps 
    the halo height, these settings should have very little effect. 

    @param l_max Output map runs from longitudes -size_l to +size_l in degrees
    @param b_max Output map runs from latitudes -size_b to +size_b in degrees
    @param res degrees/pixel
    @param r_min Float specifying the min 2-d radius for integration 
    @param r_max Float specifying the max 2-d radius for integration
    @param species CR species to convolve.  Can be 'electron', 'positron', 'proton', 'antiproton'.
    @param E *Kinetic* energy of CRs in MeV
    @param source_dist Distribution of CR sources.  Can be 'Lorimer', 'SNR', 'Yusifov', 'OBStars'. See DOI:10.1088/0004-637X/750/1/3 for more info.
    
    @returns proj_skymap 2-d array in lat/long at specified resolution
    '''

    # See LOS_Gas_Ferierre for better documentation 
    # Open a file tmp.py and write the passed function to it.  This is required because of 
    # some issues surrounding multithreading in python.
    f = open('tmp.py','wb')
    f.write(func)
    f.close()
    import tmp
    reload(tmp)

    if source_dist == 'Lorimer'    : hdulist = hdu_Lorimer
    elif source_dist == 'SNR'      : hdulist = hdu_SNR  
    elif source_dist == 'Yusifov'  : hdulist = hdu_Yusifov
    elif source_dist == 'OBStars'  : hdulist = hdu_OBstars
    else: raise("Invalid source distribution.  Must be 'Lorimer', 'SNR', 'Yusifov', or 'OBStars'")

    # Get the cosmic ray interpolator
    interpolator = GetInterpolater(hdulist, E,species)

    n_threads = mp.cpu_count() # number of cpus to use
    kernel = partial(__LOS_CR_Thread,
                     n_thread=n_threads, l_max=l_max, b_max=b_max,
                     res=res, r_min=r_min, r_max=r_max, z_step=z_step,interpolator=interpolator)
    
    p = mp.Pool(n_threads)
    slices = p.map(kernel,range(n_threads))
    p.close()
            
    proj_skymap = slices[0]
    for slice_ in slices[1:]:
        proj_skymap = np.append(proj_skymap,slice_,axis=0)
    return proj_skymap