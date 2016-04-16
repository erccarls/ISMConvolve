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
import gas_grid, ferriere, dickey_lockman



def GenGasSkymapAnnulus(size_l,size_b,res, components,H2_mult=2.,r_min=0, r_max=20,z_step=0.02,func='func = lambda x,y,z:1.',square_free_e=False):
    ''' 
    Master method to generate a skymap with disk + bulge components. 
    Also wraps ugly numba functionality. 
    
    @param size_l Output map runs from longitudes -size_l to +size_l in degrees
    @param size_b Output map runs from latitudes -size_b to +size_b in degrees
    @param res degrees/pixel
    @param components list of components to include from 'H2','HI','HII','Free_e', 'H2_PEB','HI_NS
    @param H2_mult float specifying what to multiply the H2 component by.  Default is 1
    @param r_min Float specifying the min 2-d radius for integration 
    @param r_max Float specifying the max 2-d radius for integration
    
    @returns proj_skymap 2-d array in lat/long at specified resolution
    '''
    # Determine which components to compute
    def CheckInList(key):
        """
        'in' operator doesn't work as needed ('HI' in ('HII',) returns true)
        We perform an exact string comparison here.
        """
        '''Options ['H2','HI','HII', 'H2_PEB','HI_NS','Free_e'] '''
        for comp in components:
            if comp==key:
                return 1
        return 0
    
    # Determine which components to compute
    H2     = CheckInList('H2')
    HI     = CheckInList('HI')
    HII    = CheckInList('HII')
    Free_e = CheckInList('Free_e')
    H2_PEB = CheckInList('H2_PEB')
    HI_NS = CheckInList('HI_NS')
    
    # Turn off numpy error reporting (underflows happen from tails of exponentials.)
    np.seterr(all='ignore')
    
    # Write func to a file so it is importable by child threads
    f = open('tmp.py','wb')
    f.write(func)
    f.close()
    reload(tmp)


    # Initialize projected skymap
    len_b, len_l = int(round(2*size_b/res)),int(round(2*size_l/res))
    proj_skymap = np.zeros(shape=(len_b,len_l))
    
    # Are we asking for one of the gridded distributions?
    if (H2_PEB or HI_NS) ==1: 
        proj_skymap += gas_grid.LOS_Gas_Grid(size_l,size_b,res,H2_PEB,HI_NS,H2_mult,r_min,r_max,H2_map='PEB',z_step=z_step,func=func)
        
    # Or one of the semi-analytic models...
    if (H2 or HI or HII or Free_e) == 1:
        # If this section doesn't require inner galaxy, don't integrate it.
        if r_min<3.0:
            # Integrate region within 3.0 kpc (in 3-d coords) of GC
            proj_skymap += ferriere.LOS_Gas_FerriereNP(size_l, size_b, res, H2, HI, HII ,Free_e,
                                                H2_mult,r_min,r_max,z_step=z_step,func=func,
                                                square_free_e=square_free_e)
            # Integrate outer regions of galaxy
            proj_skymap += dickey_lockman.LOS_Gas_Dickey_LockmanNP(size_l,size_b,res,H2,HI,HII,Free_e,
                                                    H2_mult,r_min,r_max, z_step=z_step,func=func,
                                                    square_free_e=square_free_e)
        else:
            # Integrate outer regions of galaxy
            proj_skymap += dickey_lockman.LOS_Gas_Dickey_LockmanNP(size_l,size_b,res,H2,HI,HII,Free_e,
                                                    H2_mult,r_min,r_max, z_step=z_step,func=func,
                                                    square_free_e=square_free_e)
    # Zeros are bad later on. 
    proj_skymap.clip(1e-30)
    
    return proj_skymap



#==============================================
# MASTER INTEGRATOR
#==============================================
import pyfits
from scipy.ndimage.interpolation import zoom
import numpy as np
from numpy import *
import tmp
    
# Load survey normalizations
hdulist_CO = pyfits.open('CO_residuals_qdeg_ferriere.fits.gz')
hdulist_HI = pyfits.open('HI_residuals_qdeg_ferriere.fits.gz')
hdulist_CO_PEB = pyfits.open('CO_residuals_qdeg_PEB.fits.gz')
hdulist_HI_NS = pyfits.open('HI_residuals_qdeg_NS.fits.gz')
hdulist_Free_e = pyfits.open('H_alpha_res_NE2001.fits.gz')
#print np.min(hdulist_CO[0].data), np.max(hdulist_CO[0].data)

def ConvolveGasSkymap(size_l,size_b,res, components,H2_mult=2.,z_step=0.02,func='func = lambda x,y,z:1.',
                        square_free_e=False, renorm_free_e=False):
    '''
    Master method to generate a skymap with disk + bulge components. 
    Also wraps ugly numba functionality. 
    @param size_l Output map runs from longitudes -size_l to +size_l in degrees
    @param size_b Output map runs from latitudes -size_b to +size_b in degrees
    @param res degrees/pixel
    @param components list of components to include from 'H2','HI','HII','Free_e', 'H2_PEB','HI_NS'
    @param H2_mult float specifying what to multiply the H2 component by.  Default is 2
    @param square_free_e Square the free electron density? 
    @param renorm_free_e renormalize free_e and HII col dens to the H_alpha map?  (via emission measure)
    '''
    
    kpc2cm = 3.08567e21

    def CheckInList(key):
        """
        'in' operator doesn't work as needed ('HI' in ('HII',) returns true)
        We perform an exact string comparison here.
        """
        '''Options ['H2','HI','HII', 'H2_PEB','HI_NS','Free_e'] '''
        
        for comp in components:
            if comp==key:
                return 1
        return 0
    # If single component and not in tuple, recast as list
    if (type(components) != type(())) and (type(components) != type([])):components = (components,)
    
    
    # Determine which components to compute
    H2     = CheckInList('H2')
    HI     = CheckInList('HI')
    HII    = CheckInList('HII')
    Free_e = CheckInList('Free_e')
    H2_PEB = CheckInList('H2_PEB')
    HI_NS = CheckInList('HI_NS')
    if (H2 or HI or HII or Free_e or H2_PEB or HI_NS) == False: raise Exception('No valid components specified.')
    
    # Write func to a file so it is importable by child threads
    f = open('tmp.py','wb')
    f.write(func)
    f.close()
    
    # reload the tmp module
    reload(tmp)
    
    # Function to interpolate the map and select out the relevant portion
    def interpMap(galprop_norm, mapRes):
        center_b, center_l = np.int_(np.round(np.array(galprop_norm.shape)/2.)) # find center indices
        start_l, stop_l = center_l - int(round(size_l/mapres)), center_l + int(round(size_l/mapres))
        start_b, stop_b = center_b - int(round(size_b/mapres)), center_b + int(round(size_b/mapres))
        # Resize the galprop residual templates to match our resolution
        galprop_norm = galprop_norm[start_b:stop_b,start_l:stop_l]
        resized = zoom(galprop_norm,mapRes/res,order=1)
        #print np.min(resized-galprop_norm), np.max(resized-galprop_norm)
        return resized
        
    TotalSkymap = np.zeros( (int(round(2*size_b/res)),int(round(2*size_l/res))) )
    
    if H2==1:
        print 'Integrating H2 rings'
        # Load residual normalizations suvey annuli radii 
        rings    = hdulist_CO[1].data
        mapres   = hdulist_CO[0].header['CDELT1'] # angular distance per pixel
       
        #=================================== 
        # For each annulus
        for i in range(len(rings)):
            # Get inner outer radii
            inner,outer = rings[i] 
        
            # X_CO from table 2 of 1202.4039.
            # X_CO = 1.67e20 # units cm^-2 (K (km/s))^-1
            
            # Interpolate residual template, select out portion we need, and multiply by X_CO factor
            # Take X_CO exp form taken from Ferrier 2007, roughly fit to give 3e19 in GC and 1.5e20 at R_solar
            # Max out X_CO according to Strong 2004 
            X_CO = 3e19*np.min((exp(0.5*(inner+outer)/5.3),100.))
            #X_CO = 1.67e20 # units cm^-2 (K (km/s))^-1

            galprop_norm = X_CO*interpMap(hdulist_CO[0].data[i], mapres)
            
            # Now convolve the H2 gas dist against 'func'
            annulus = GenGasSkymapAnnulus(size_l,size_b,res, components=('H2',),H2_mult=H2_mult,
                                          r_min=inner, r_max=outer,z_step=z_step,func=func)
            
            # Multiply by the galprop residual template
            annulus *= galprop_norm
            TotalSkymap += annulus
            
            
    if HI==1:
        print 'Integrating HI rings'
        # Load residual normalizations suvey annuli radii 
        rings    = hdulist_HI[1].data
        mapres   = hdulist_HI[0].header['CDELT1'] # angular distance per pixel
        #=================================== 
        # For each annulus
        for i in range(len(rings)):
        #for i in range(1):
            # Get inner outer radii
            inner,outer = rings[i] 

            # Interpolate residual template, select out portion we need, and multiply by 1e21 factor (units of map)
            galprop_norm = 1e20*interpMap(hdulist_HI[0].data[i], mapres)
            
            # Now convolve the H2 gas dist against 'func'
            annulus = GenGasSkymapAnnulus(size_l,size_b,res, components=('HI',),H2_mult=1.,
                                          r_min=inner, r_max=outer,z_step=z_step,func=func)
            
            # Multiply by the galprop residual template
            annulus *= galprop_norm
            #annulus[np.where(annulus>1e24)]=0
            TotalSkymap += annulus
    
    
    if HII==1:
        print 'Integrating HI (WIM+HIM+VHIM)'

        #Correctly? Normalize the map 
        mapres   = hdulist_Free_e[0].header['CD1_2'] # angular distance per pixel
        # 0.9 converts the free_e density to HII 
        galprop_norm = .9*interpMap(hdulist_Free_e[0].data, mapres)
        if renorm_free_e==True:
            TotalSkymap += kpc2cm*galprop_norm * GenGasSkymapAnnulus(size_l,size_b,res, components=('HII',),H2_mult=1.,
                                          r_min=0., r_max=30.,z_step=z_step,func=func)
        else:
            TotalSkymap += kpc2cm*GenGasSkymapAnnulus(size_l,size_b,res, components=('HII',),H2_mult=1.,
                                          r_min=0., r_max=30.,z_step=z_step,func=func)



    if Free_e==1:
        print 'Integrating Free_e'
        mapres   = hdulist_Free_e[0].header['CD1_2'] # angular distance per pixel
        galprop_norm = interpMap(hdulist_Free_e[0].data, mapres)
        if square_free_e == True: 
            resExponent = 2.
        else:
            resExponent = 1.
        if renorm_free_e==True:
            TotalSkymap += kpc2cm*galprop_norm**resExponent*GenGasSkymapAnnulus(size_l,size_b,res, components=('Free_e',),H2_mult=1.,
                                          r_min=0., r_max=30.,z_step=z_step,func=func, square_free_e=square_free_e)
        else:
            TotalSkymap += kpc2cm*GenGasSkymapAnnulus(size_l,size_b,res, components=('Free_e',),H2_mult=1.,
                                      r_min=0., r_max=30.,z_step=z_step,func=func, square_free_e=square_free_e)
            
    if H2_PEB==1:
        print 'Integrating H2_PEB'
        
        mapres   = hdulist_CO_PEB[0].header['CDELT1'] # angular distance per pixel
        # this factor in front just renormalizes the max value to surveys 
        galprop_norm = interpMap(hdulist_CO_PEB[0].data, mapres)
        #print np.max(hdulist_CO_PEB[0].data)
        
        #TotalSkymap += GenGasSkymapAnnulus(size_l,size_b,res, components=('H2_PEB',),H2_mult=H2_mult,
        TotalSkymap += galprop_norm*GenGasSkymapAnnulus(size_l,size_b,res, components=('H2_PEB',),H2_mult=H2_mult,
                                          r_min=0., r_max=30.0,z_step=z_step,func=func)
        
        
        
    if HI_NS==1:
        print 'Integrating HI_NS'
        
        mapres   = hdulist_HI_NS[0].header['CDELT1'] # angular distance per pixel
        galprop_norm = 1e20*interpMap(hdulist_HI_NS[0].data, mapres)
        
        TotalSkymap += galprop_norm*GenGasSkymapAnnulus(size_l,size_b,res, components=('HI_NS',),H2_mult=H2_mult,
        #TotalSkymap +=GenGasSkymapAnnulus(size_l,size_b,res, components=('HI_NS',),H2_mult=H2_mult,
                                          r_min=0., r_max=30.0,z_step=z_step,func=func)
    
    #TotalSkymap = TotalSkymap.clip(1e-30)
    
    return TotalSkymap
    
    
    
    
    
    
