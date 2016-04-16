from numpy import *
import numpy as np
import multiprocessing as mp
from functools import partial



def __LOS_Gas_FerriereNP_thread(tNum, n_thread, l_max, b_max, res, H2, HI, HII, Free_e, H2_mult, 
                                r_min, r_max, z_step=0.02,func='func = lambda x,y,z:1.',square_free_e=False):
    #==============================================
    # Integration Parameters
    #==============================================
    R_solar = 8.5 # Sun at 8.5 kpc
    kpc2cm  = 3.08568e21
    # Only integrate over necessary regions.  Add a bit extra to z_stop since we are integrating at non-zero z as well
    z_start,z_stop,z_step = max(R_solar-r_max,0),R_solar+r_max+2.,z_step # los start,stop,step-size in kpc
    import tmp
    # load the tmp function (the one passed to use in the convolution)
    func = tmp.func
    # distances along the LOS
    zz = np.linspace(start=z_start,stop=z_stop,num=int(np.ceil((z_stop-z_start)/z_step)))
    # some conveinient constants
    deg2rad = np.pi/180.
    pi = np.pi
    # List of lat/long to loop over.
    bb = np.linspace(-b_max,b_max,int(np.ceil(2*b_max/res)))*deg2rad
    ll = np.linspace(-l_max,l_max,np.ceil(int(2*l_max/res)))*deg2rad
    # divide latitudes according to thread
    stride = len(bb)/n_thread
    if tNum == n_thread-1: bb = bb[tNum*(stride):]
    else: bb = bb[tNum*stride:(tNum+1)*stride]
    
    # Params for coordinate transformations from ferriere 2007.  Don't want to compute every time.
    theta_c = np.deg2rad(70.)
    alpha, beta, theta_d = np.deg2rad((13.5, 20., 48.5))
    alpha_vhim = np.deg2rad(21.)
    phi_00 = -3.84229569333e+14 # gravitational potential at zero
    # precompute some of the transformations
    ctheta_c, stheta_c = cos(theta_c), sin(theta_c)
    cbeta, sbeta       = cos(beta), sin(beta)
    calpha, salpha     = cos(alpha), sin(alpha)
    ctheta_d, stheta_d = cos(theta_d), sin(theta_d)

    # Master projected skymap to be returned (stores column density).  
    proj_skymap = np.zeros(shape=(len(bb),len(ll)))
    # Iterate over latitudes and longitudes of this slice of the map
    for bi in range(len(bb)):
        for lj in range(len(ll)): 
            # should add this optimization at some point.  
            # Don't integrate at latitudes outside the current annulus
            #if np.abs(ll[lj])>arctan(4/8.5): continue
                    
            # current lat/long/z_los
            b, l, z_los = bb[bi], ll[lj], zz
            # x,y,z in galactocentric frame with sun at (x,y,z) = (8.5,0,0)
            x = -z_los*cos(l)*cos(b)+R_solar
            y = z_los*sin(l)*cos(b)
            z = z_los*sin(b)
            # Galactocentric radius in 2d/3d
            r_2d = sqrt(x*x+y*y)
            r_3d = sqrt(x*x+y*y+z*z)
            # This is the bulge component only.  Valid in inner 3 kpc
            idx = np.where((r_3d<3.0)  & (r_2d>r_min) & (r_2d<r_max) )[0]
            # Select only the relevant portions. 
            x,y,z = x[idx],y[idx],z[idx]
            r_2d, r_3d = r_2d[idx], r_3d[idx]

            los_sum = np.zeros(len(zz[idx]))
            # Evaluate our function once for this line of sight
            fxyz = func(x,y,z)
            
            ###############################################################
            # 3-d Model of H2 and HI CMZ from Ferriere et al (2007) astro-ph/0702532
            # Defined in galactic coordinate system with x-axis Sun-GC line
            # Units in cm^-3 and kpc

            # Compute 
            X = (x+.050)*ctheta_c + (y-.050)*stheta_c
            Y = -(x+.050)*stheta_c + (y-.050)*ctheta_c
            # H2 CMZ
            if H2==1:
                los_sum += fxyz*H2_mult * 150.*exp(-(( sqrt(X**2+(2.5*Y)**2)-.125)/.137)**4)*exp(-(z/0.018)**2)*z_step
            # HI CMZ
            if HI==1:
                los_sum += fxyz*8.8*exp(-(( sqrt(X**2+(2.5*Y)**2)-.125)/.137)**4)*exp(-(z/0.054)**2)*z_step


            ###############################################################
            # 3-d Model of H2 and HI disk from Ferrier et al (2007) astro-ph/0702532
            # Defined in galactic coordinate system with x-axis Sun-GC line
            # Units in cm^-3 and kpc
            X = (x*cbeta*ctheta_d 
                 - y*(salpha*sbeta*ctheta_d-calpha*stheta_d)
                 - z*(calpha*sbeta*ctheta_d+salpha*stheta_d))
            Y = (-x*cbeta*stheta_d
                 +y*( salpha*sbeta*stheta_d+calpha*ctheta_d )
                 +z*( calpha*sbeta*stheta_d-salpha*ctheta_d))
            Z = (x*sbeta
                 +y*salpha*cbeta
                 +z*calpha*cbeta)
            
            if (H2==1):
                los_sum += fxyz*H2_mult * 4.8*exp(-((sqrt(X**2+(3.1*Y)**2)-1.2)/.438)**4)*exp(-(Z/0.042)**2)*z_step
            if (HI==1):
                los_sum += fxyz*0.34*exp(- ((sqrt(X**2+(3.1*Y)**2)-1.2)/.438)**4)*exp(-(Z/0.120)**2)*z_step


            ###############################################################
            # 3-d Model of WIM/HIM/VHIM from Ferrier et al (2007) astro-ph/0702532
            # based on NE2001 model Cordes & Lazio (2002)
            # Defined in galactic coordinate system with x-axis Sun-GC line
            # Units in cm^-3 and kpc. 
            #scale_H2 = 0.95 # orig NE2001 
            scale_H2 = 2. # NE2001 with Gaensler 2008
            if HII==1:
                # WIM
                los_sum += fxyz*8.0 * (exp(-(x**2+(y+0.01)**2)/.145**2)  *  exp(-(z+0.02)**2/0.026**2)
                        + 0.009*exp(-((r_2d-3.7)/1.85)**2 ) / cosh(z/.140)**2
                        + 0.005*cos(pi*r_2d/34.) / cosh(z/scale_H2)**2 * np.int_((17.-r_2d)>=0)) *z_step


                # eqn 13 of astro-ph/0702532... Gravitational potential of MW
                r2,z2 = r_2d*r_2d, z*z
                q1    = sqrt(1+ (144+r2+z2)/210.) # quantity which is evaluated twice
                phi = -(225e5)**2* ( 8.887/(sqrt(r2+((6.5)+sqrt(z**2+0.26**2))**2) 
                                    + 3.0/(0.7+sqrt(r2+z2))
                                    - 0.325*log((q1-1)/(q1+1))))

                # n_HII_HIM
                los_sum += fxyz*(0.009**.6667-1.5e-17*(phi-phi_00) )**1.5*z_step

                # n_HII_VHIM
                eta  = y*cos(alpha_vhim) + z*sin(alpha_vhim)
                zeta = -y*sin(alpha_vhim) + z*cos(alpha_vhim)
                los_sum += fxyz*0.29*exp(-( (x*x+eta*eta)/0.162**2 + zeta*zeta/0.09**2))*z_step

            ###############################################################
            # 3-d Model of free electrons from Ferrier et al (2007) astro-ph/0702532
            # based on NE2001 model Cordes & Lazio (2002)
            # Defined in galactic coordinate system with x-axis Sun-GC line
            # Units in cm^-3 and kpc.  
            if (Free_e == 1):
                r2,z2 = r_2d*r_2d, z*z
                los_sum_e = np.zeros(len(los_sum))
                q1    = sqrt(1+ (144+r2+z2)/210.) # quantity which is evaluated twice
                phi = -(225e5)**2* ( 8.887/(sqrt(r2+((6.5)+sqrt(z**2+0.26**2))**2) 
                                    + 3.0/(0.7+sqrt(r2+z2))
                                    - 0.325*log((q1-1)/(q1+1))))

               
                los_sum_e += (   np.int_((17.-r_2d)>=0)*0.05*cos(pi*r_2d/34.) / cosh(z/scale_H2)**2
                 +0.09*exp(-((r_2d-3.7)/1.85)**2 )/ cosh(z/.140)**2 
                 +10. * exp(-(x*x+(y+.010)**2)/.145**2) * exp(-(z+.020)**2/.026**2)   )*z_step
                
                
                los_sum_e += (0.011**.6667-1.73e-17*(phi-phi_00) )**1.5*z_step

                if square_free_e == True: 
                    los_sum += fxyz*los_sum_e**2
                else:
                    los_sum += fxyz*los_sum_e
                
                
            # Sum along line of sight where we are within contributing zones
            proj_skymap[bi,lj] = np.sum(los_sum)
    return proj_skymap

def LOS_Gas_FerriereNP(l_max, b_max, res, H2, HI, HII, Free_e, H2_mult, r_min, r_max, z_step=0.02,func='func = lambda x,y,z: 1.',square_free_e=False):
    # Open a file tmp.py and write the passed function to it.  This is required because of 
    # some issues surrounding multithreading in python.
    f = open('tmp.py','wb')
    f.write(func)
    f.close()
    import tmp
    reload(tmp)
    # Count CPUs and setup integration kernel. 
    n_threads = mp.cpu_count()
    kernel = partial(__LOS_Gas_FerriereNP_thread,
                     n_thread=n_threads, l_max=l_max, b_max=b_max,
                     res=res, H2=H2, HI=HI, HII=HII, Free_e=Free_e, H2_mult=H2_mult, 
                     r_min=r_min, r_max=r_max, z_step=z_step,func=func,square_free_e=square_free_e)
    # Initialize the thread pool. 
    p = mp.Pool(n_threads)
    # Parallel Map each slice 
    slices = p.map(kernel,range(n_threads))
    # close the thread pool
    p.close()
    # Merge the slices into the full image.
    proj_skymap = slices[0]
    for slice_ in slices[1:]:
        proj_skymap = np.append(proj_skymap,slice_,axis=0)
        
    return proj_skymap
