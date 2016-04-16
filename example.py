
#------------------------------------------------------------------------------------------------
# example.py
# In this script we demonstrate the usage of convolution code from arXiv xxxx.xxxx.  
# Basically, one can define a function of x,y,z in the Galactic coordinates.
# (sun lies at (8.5,0,0) with +y corresponding to the direction of positive longitudes)
# Which is can then be integrated against gas, CR electrons or protons, dark matter, 
# or free electrons.
# 
# Author: Eric Carlson erccarls@ucsc.edu
# Affiliation: University of California Santa Cruz
# Date: April 25 2015
#------------------------------------------------------------------------------------------------

import numpy as np

#------------------------------------------------------------------------------------------------------------
# Here we define dark matter distributions as the function we are going to convolve against gas along the line of sight
# In order to support multithreaded convolutions, the functions must be defined as strings which are then written to file.
# and imported by each thread.  Thus required imports should be included at the beginning of the string as is done below.  
# The function must be called "func", and the operations must accept x,y,z as numpy vectors since broadcasting is used.

# Standard NFW
NFWFunc='''
# NFW
gamma=1.
r_s=20.
import numpy as np
def func(x,y,z):    
    r=np.sqrt(x*x+y*y+z*z)
    return r**-gamma*(1/(1+r/r_s)**(3-gamma))
'''

# Contracted NFW
NFWConFunc ='''
import numpy as np
gamma=1.2
r_s=20.
def func(x,y,z):    
    r=np.sqrt(x*x+y*y+z*z)
    return r**-gamma*(1/(1+r/r_s)**(3-gamma))
'''

# Einasto
EinFunc ='''
# Einasto
import numpy as np
def func(x,y,z,alpha=1.):
    r=np.sqrt(x*x+y*y+z*z)
    return np.exp(-2/0.17*((r/20.)**0.17-1))
'''


# imports the gas integration module which also contains the free electron distributions.
import gas_integrator as gi

# Some examples of integrating different dark matter profiles against the gas. 
# Here are som parameter descriptions
# size_l: returned skymap will range from longitude -size_l/2. to size_l/2.
# size_b: returned skymap will range from latitude -size_l/2. to size_l/2.
# res: output resolution in degrees/pixel
# components: can be single string or list of strings (components are then added) from the following list 
#    H2 : Ferierre 2007 model for molec. hydrogen
#    HI : Ferierre 2007 model for atomic
#    HII: Ferierre 2007 model for ionized hydrogen (based somewhat on NE2001)
#    H2_PEB: Pohl, Englmeir, Bissantz model 
#    HI_NS: Nakanishi, Sofue model
#    Free_e: NE2001 model for free electrons
# H2_mult: Multiply the molecular component by this number, typically 2 if considering interactions with protons.
# z_step: line-of-sight element for integration in kpc. 
# func: function to integrate agsinst, specified above.
# square_free_e: Can be useful if interested in emission measure to compare to Halpha maps.

# F07 model
NFW_F07 = gi.ConvolveGasSkymap(size_l=40, size_b=40, res=.25, components=['H2','HI','HII'], H2_mult=2.0, z_step=0.02, 
                    func=NFWFunc, square_free_e=False)
NFWCon_F07 = gi.ConvolveGasSkymap(size_l=40, size_b=40, res=.25, components=['H2','HI','HII'], H2_mult=2.0, z_step=0.02, 
                    func=NFWConFunc, square_free_e=False)
Ein_F07 = gi.ConvolveGasSkymap(size_l=40, size_b=40, res=.25, components=['H2','HI','HII'], H2_mult=2.0, z_step=0.02, 
                    func=EinFunc, square_free_e=False)

# PEB+NS model
NFW_PEBNS = gi.ConvolveGasSkymap(size_l=40, size_b=40, res=.25, components=['H2_PEB','HI_NS','HII'], H2_mult=2.0, z_step=0.02, 
                    func=NFWFunc, square_free_e=False)
NFWCon_PEBNS = gi.ConvolveGasSkymap(size_l=40, size_b=40, res=.25, components=['H2_PEB','HI_NS','HII'], H2_mult=2.0, z_step=0.02, 
                    func=NFWConFunc, square_free_e=False)
Ein_PEBNS = gi.ConvolveGasSkymap(size_l=40, size_b=40, res=.25, components=['H2_PEB','HI_NS','HII'], H2_mult=2.0, z_step=0.02, 
                    func=EinFunc, square_free_e=False)

# Free electrons
NFW_Free = gi.ConvolveGasSkymap(size_l=40, size_b=40, res=.25, components='Free_e', H2_mult=2.0, z_step=0.02, 
                    func=NFWFunc, square_free_e=False)
NFWCon_Free = gi.ConvolveGasSkymap(size_l=40, size_b=40, res=.25, components='Free_e', H2_mult=2.0, z_step=0.02, 
                    func=NFWConFunc, square_free_e=False)
Ein_Free = gi.ConvolveGasSkymap(size_l=40, size_b=40, res=.25, components='Free_e', H2_mult=2.0, z_step=0.02, 
                    func=EinFunc, square_free_e=False)

#------------------------------------------------------------------------------------------------------------
# Convolve dark matter against itself (or somthing else by specifiying func1 and func2)
import DM
NFW2_Map = DM.LOS_DM(40,40,.25,func1=NFWFunc,func2=NFWFunc,z_step=.02)

#------------------------------------------------------------------------------------------------------------
# Convolve against cosmic-rays. 
import CR

# Here we want to run several energies, so we create a log-spaced list.
energies = np.logspace(1, 6, 21)
# Define the all of the primary source distributions available.
dists = 'Yusifov', 'Lorimer','SNR', 'OBStars'

# For each source_dist (primary distribution), convolve the species='electron' cosmic-rays dist at energy E specified in MeV
# species: CR species. Can be 'electron' or 'proton'
# source_dist: Primary source distribution 'Yusifov', 'Lorimer','SNR', 'OBStars'

components = []
for dist in dists:    
    cube = np.array([CR.ConvolveCR(l_max=40, b_max=40, res=.25,  species='electron', E=E, r_min=0., r_max=20.,source_dist=dist,z_step=0.02,func=NFWFunc) for E in energies])
    components.append(cube)

components_p = []
for dist in dists:    
    cube = np.array([CR.ConvolveCR(l_max=40, b_max=40, res=.25,  species='proton', E=E, r_min=0., r_max=20.,source_dist=dist,z_step=0.02,func=NFWFunc) for E in energies])
    components_p.append(cube)





#------------------------------------------------------------------------------------------------------------
# Example of plotting the results.
fig = plt.figure(figsize=(10,7))
for i in range(0,21,2):
    plt.subplot(3,4,i/2+1)
    im = plt.imshow(np.log10(hdulist['CR_ELECTRON_YUS'].data[i]/hdulist['CR_ELECTRON_YUS'].data[i].max()), extent=[-20,20,-20,20], origin='lower',cmap='CMRmap', vmin=-2.5, vmax=0)
    plt.text(-18,-18,r'\noindent NFW $\times$ CR $e^- \\E=$%1.3f'%(energies[i]/1e3) + ' GeV', color='w', fontsize=7)
    plt.xlim(20,-20)




#------------------------------------------------------------------------------------------------------------
# Here is an example of writing the results out to a FITS file. Skip if not needed. 

# import pyfits
# pyfits.header.Header()
# # Specify header keywords
# hdr = {'CDELT1':.25, 'CDELT2':.25, 'CRVAL1':-20, 'CRVAL2':-20, 'CTYPE1':'GLAT', 'CTYPE2':'GLON'}
# hdulist = pyfits.HDUList()

# names=['Free_e_NFW', 'Free_e_Contracted_NFW', 'Free_e_Ein']
# for i, component in enumerate([NFW_Free, NFWCon_Free, Ein_Free]):
#     hdu = pyfits.ImageHDU(data=component.astype(np.float32), name=names[i])
#     for key, val in hdr.items():
#         hdu.header[key] = val
#     hdulist.append(hdu)

# names=['F07_NFW', 'F07_Contracted_NFW', 'F07_Ein']
# for i, component in enumerate([NFW_F07, NFWCon_F07, Ein_F07]):
#     hdu = pyfits.ImageHDU(data=component.astype(np.float32), name=names[i])
#     for key, val in hdr.items():
#         hdu.header[key] = val
#     hdulist.append(hdu)
    
# names=['PEBNS_NFW', 'PEBNS_Contracted_NFW', 'PEBNS_Ein']
# for i, component in enumerate([NFW_PEBNS, NFWCon_PEBNS, Ein_PEBNS]):
#     hdu = pyfits.ImageHDU(data=component.astype(np.float32), name=names[i])
#     for key, val in hdr.items():
#         hdu.header[key] = val
#     hdulist.append(hdu)    
# 
# hdulist.info()
# hdulist.writeto('component_skymaps.fits', clobber=True)





