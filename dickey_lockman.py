import multiprocessing as mp
from functools import partial
import numpy as np
from numpy import *

#=================================================
# This section integrates regions R>3kpc from GC
#=================================================
def __LOS_Gas_Dickey_Lockman_Thread(tNum, n_thread, l_max,b_max,res,H2,HI,HII,Free_e,H2_mult,r_min,r_max,z_step=0.02,square_free_e=False):
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
    bb = np.linspace(-b_max,b_max,int(np.ceil(2*b_max/res)))*deg2rad
    # divide according to thread
    stride = len(bb)/n_thread
    if tNum == n_thread-1: bb = bb[tNum*(stride):]
    else: bb = bb[tNum*stride:(tNum+1)*stride]
        
    ll = np.linspace(-l_max+res/2.,l_max+res/2.,np.ceil(int(2*l_max/res)))*deg2rad
    
    # Params for coordinate later transformations from ferriere 2007
    theta_c = np.deg2rad(70.)
    alpha, beta, theta_d = np.deg2rad((13.5, 20., 48.5))
    alpha_vhim = np.deg2rad(21.)
    phi_00 = -3.84229569333e+14 # gravitational potential at zero
    
    ###################################################
    # H2 Gas model from astro-ph/0610769
    # Neglecting variations in center value of z0 since 
    # this is much smaller than FWHM of molecular disk
    # Provides improved molecular gas model in GC region
    # Values are linearly interpolated as a function of 
    # galactocentric radius.
    r_H2 = [i+0.5 for i in range(11)]
    n_0_H2  = lambda r: np.interp(r,r_H2, [8.73,1.28,0.76,0.75,0.83,0.66, 0.48, 0.26, 0.09, 0.05, 0.05],right=0.)
    FWHM_H2 = lambda r: np.interp(r,r_H2,np.array([48,56,68,88,78,102,84,90,186,182,160])/1.e3,right=0.)

    # Now n_H2(r,z) gives the density at any point. 
    n_H2 = lambda r, z: n_0_H2(r)/cosh(np.log(1+sqrt(2))*z/FWHM_H2(r))**2 

    ##################################################
    # HI gas model from 1976 ApJ 208, 346G
    # Overall renormalized to Dickey & Lockman 1990 (1990ARA&A..28..215D),
    # along with z-dependence
    r_HI = [(i+0.5)/2 for i in range(4,32)]
    # Normalized to 1 at 8.5 kpc. Z dep normalizes to correct value according to 
    n_0_HI  = lambda r: np.interp(r,r_HI, np.array([.13,.14,.16,.19,.25,.30,.33,.32,.31,.30,.37,.38,
              .36,.32,.29,.28,.40,.25,.23,.32,.36,.32,.25,.16,.10,.09,.08,.06])/0.34,right=0.)

    z_HI,z_dens_HI = transpose(np.genfromtxt('dickey_lockman.dat',delimiter=','))
    n_HI = lambda r,z : n_0_HI(r)*interp(z,z_HI/1e3,z_dens_HI,right=0.)

    # Total gas is then atomic hydrogen + 2 * molecular hydrogen
    GasTotal = lambda r,z: n_HI(r,np.abs(z))+2*n_H2(r,np.abs(z))
    
    # Master projected skymap 
    proj_skymap = np.zeros(shape=(len(bb),len(ll)))
    
    # Loop latitudes 
    for bi in range(len(bb)):
        # loop longitude
        for lj in range(len(ll)):
            los_sum=0.
            l,b = ll[lj], bb[bi]
            # z in cylindrical coords
            z_2d = zz*sin(b) 
            x,y = -zz*cos(l)*cos(b)+R_solar, zz*sin(l)*cos(b)
            
            r_2d = sqrt(x**2+y**2)
            r_3d = sqrt(r_2d**2.+z_2d**2.)
            
            # We now have r_2d and z_2d to plug into gas model.
            # Find where we are inside the central bulge and set to zero
            idx = np.where( (r_3d>3.0) & (r_2d>=r_min) & (r_2d<r_max) )[0]
            
            x,y,z_2d = x[idx],y[idx],z_2d[idx]
            z = z_2d
            r_2d, r_3d = r_2d[idx],r_3d[idx]
            
            fxyz = func(x,y,z)

            if H2==True:
                los_sum += H2_mult*np.sum(fxyz*n_H2(r_2d,np.abs(z_2d)))*z_step
            if HI==True:
                los_sum += np.sum(fxyz*n_HI(r_2d,np.abs(z_2d)))*z_step
            
            
            ###############################################################
            # 3-d Model of WIM/HIM/VHIM from Ferrier et al (2007) astro-ph/0702532
            # based on NE2001 model Cordes & Lazio (2002)
            # Defined in galactic coordinate system with x-axis Sun-GC line
            # Units in cm^-3 and kpc. 
            #scale_H2 = 0.95 # orig NE2001 
            scale_H2 = 2. # NE2001 with Gaensler 2008
            if HII==1:
                # WIM
                los_sum += sum(fxyz*8.0 * (exp(-(x**2+(y+0.01)**2)/.145**2)  *  exp(-(z+0.02)**2/0.026**2)
                        + 0.009*exp(-((r_2d-3.7)/1.85)**2 ) / cosh(z/.140)**2
                        + 0.005*cos(pi*r_2d/34.) / cosh(z/scale_H2)**2 * np.int_((17.-r_2d)>=0))) *z_step


                # eqn 13 of astro-ph/0702532... Gravitational potential of MW
                r2,z2 = r_2d*r_2d, z*z
                q1    = sqrt(1+ (144+r2+z2)/210.) # quantity which is evaluated twice
                phi = -(225e5)**2* ( 8.887/(sqrt(r2+((6.5)+sqrt(z**2+0.26**2))**2) 
                                    + 3.0/(0.7+sqrt(r2+z2))
                                    - 0.325*log((q1-1)/(q1+1))))

                # n_HII_HIM
                los_sum += sum(fxyz*(0.009**.6667-1.5e-17*(phi-phi_00) )**1.5*z_step)

                # n_HII_VHIM
                eta  = y*cos(alpha_vhim) + z*sin(alpha_vhim)
                zeta = -y*sin(alpha_vhim) + z*cos(alpha_vhim)
                los_sum += sum(fxyz*0.29*exp(-( (x*x+eta*eta)/0.162**2 + zeta*zeta/0.09**2))*z_step)


            ###############################################################
            # 3-d Model of free electrons from Ferrier et al (2007) astro-ph/0702532
            # based on NE2001 model Cordes & Lazio (2002)
            # Defined in galactic coordinate system with x-axis Sun-GC line
            # Units in cm^-3 and kpc.  
            if (Free_e == 1):
                r2,z2 = r_2d*r_2d, z*z
                los_sum_e = np.zeros(len(x))
                q1    = sqrt(1+ (144+r2+z2)/210.) # quantity which is evaluated twice
                phi = -(225e5)**2* ( 8.887/(sqrt(r2+((6.5)+sqrt(z**2+0.26**2))**2) 
                                    + 3.0/(0.7+sqrt(r2+z2))
                                    - 0.325*log((q1-1)/(q1+1))))

               
                los_sum_e += (  np.int_((17.-r_2d)>=0)*0.05*cos(pi*r_2d/34.) / cosh(z/scale_H2)**2
                 +0.09*exp(-((r_2d-3.7)/1.85)**2 )/ cosh(z/.140)**2 
                 +10. * exp(-(x*x+(y+.010)**2)/.145**2) * exp(-(z+.020)**2/.026**2)   )*z_step
                
                
                los_sum_e += (0.011**.6667-1.73e-17*(phi-phi_00) )**1.5*z_step

                if square_free_e == True: 
                    los_sum += np.sum(fxyz*los_sum_e**2)
                else:
                    los_sum += np.sum(fxyz*los_sum_e)
            
            proj_skymap[bi,lj] = los_sum 
    return proj_skymap


def LOS_Gas_Dickey_LockmanNP(l_max, b_max, res, H2, HI, HII,Free_e,H2_mult, r_min, r_max, z_step=0.02,func='func = lambda x,y,z: 1.',square_free_e=False):
    
    # See LOS_Gas_Ferierre for better documentation 
    # Open a file tmp.py and write the passed function to it.  This is required because of 
    # some issues surrounding multithreading in python.
    
    f = open('tmp.py','wb')
    f.write(func)
    f.close()
    import tmp
    reload(tmp)

    n_threads = mp.cpu_count() # number of cpus to use
    kernel = partial(__LOS_Gas_Dickey_Lockman_Thread,
                     n_thread=n_threads, l_max=l_max, b_max=b_max,
                     res=res, H2=H2, HI=HI,HII=HII,Free_e=Free_e, H2_mult=H2_mult, 
                     r_min=r_min, r_max=r_max, z_step=z_step,square_free_e=square_free_e)
    
    p = mp.Pool(n_threads)
    slices = p.map(kernel,range(n_threads))
    p.close()
            
    proj_skymap = slices[0]
    for slice_ in slices[1:]:
        proj_skymap = np.append(proj_skymap,slice_,axis=0)
    return proj_skymap