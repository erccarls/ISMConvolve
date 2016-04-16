# ISMConvolve
Line-of-sight integrals of arbitrary distributions against the Milky Way Interstellar Medium. 


Supplementary skyaps and convolution code for arXiv:1504.04782
Introduction
We present supplemental material for arXiv:1504.04782. These skymaps contain spatial distribtion of electromagnetic or neutrino radiation which is proportional to the product of dark matter particles with cosmic-ray electrons, cosmic-ray protons, free electrons, and interstellar gas. This product of dark matter times the matter component of interest is then integrated along the line of sight in order to obtain the projected intensity skymap. More generically, we also provide a python code which can be used to convolve an arbitrary three-dimensional distribution gas, cosmic ray, or free electron densities. Below, we breifly describe each component.

Cosmic-ray distribtions are derived using the numerical code Galprop (INSERT GALPROP LINK), under standard assumptions regarding cosmic-ray transport, and assuming several different distributions for the injection of primary cosmic-rays. Free electrons are based on the NE2001 model with corrections to the thick disk scale-height taken from Gaensler et al (2007). For the gas density (2*H2+HI+HII), we employ two models of Galactic gas. The first is based on Dickey & Lockman (1990) with corrections in the inner galaxy due to Bronfman et al (2000), and with the inner 3 kiloparsecs following Ferriere (2007). The second model utilizes velocity data from a the Leiden-Argentine-Bonn 21 cm and Dame et al (2001) surveys commbined with spatial deconvolutions by Nakanishi & Sofue and Pohl, Englemier, & Martin, respectively.

As described in our paper, Our implementation fo the NE2001 free-electron distribution neglects components which are small outside of the Galactic center region. This should be noted before using this code to integrate against non-centrally peaked distributions.

Code
Code should be self contained and is written in python. You can download and extract the following tarball to its own working directory. 
You will also need to have the following python packages installed: numpy, scipy, pyfits. A file "example.py" is commented thoroughly and demonstrates most of the functionalty of interest.


Code + support files tarball (~250 MB): dmcr_convolve.tar.gz
We also display a portion of the example.py code here to demonstrate ease of use (note that most of the below code is just comments).
 
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
    cube = np.array([CR.ConvolveCR(l_max=40, b_max=40, res=.25,  species='electron', E=E, r_min=0., r_max=20.,source_dist=dist,z_step=0.02,func=NFWFunc)
                     for E in energies])
    components.append(cube)

components_p = []
for dist in dists:    
    cube = np.array([CR.ConvolveCR(l_max=40, b_max=40, res=.25,  species='proton', E=E, r_min=0., r_max=20.,source_dist=dist,z_step=0.02,func=NFWFunc) 
                     for E in energies])
    components_p.append(cube)

           


Skymaps
A single FITS file provides an intensity skymap of the inner 40x40 degree window surrounding the Galactic center for each combination of matter component and dark matter profile. For cosmic-rays, the data is given as a cube, with the additional dimension corresponding to log-spaced energies between 10 MeV and 1 TeV. Full information can be found in the FITS HDUList and component headers.

The FITS file can be downloaded here: component_skymaps.fits.gz

  

Contact
Questions or bug reports should be directed to Eric Carlson.

email: erccarls@ucsc.edu

