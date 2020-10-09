import sys
# Add the ptdraft folder path to the sypath list
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
import numpy as np
from AGN_LF_config import LF_config
from cosmology import *
from Survey import *
from astropy.io import fits as pyfits
import random

import scipy.integrate

def get_luminosity( Fx ,Fx_err, z, z_err=0, power_law=LF_config.pl):
    Fx, Fx_err, z, z_err, power_law = np.asarray(Fx), np.asarray(Fx_err), np.asarray(z), np.asarray(z_err), np.asarray(power_law)
    dL = np.asarray( [luminosity_dist(zi) for zi in np.nditer(z)] )
    Ddl = np.asarray( [derivative_lum_dist(zi) for zi in np.nditer(z)] )
    Lum = Fx*4.0*np.pi*(dL**2.0)*(1.0+z)**(power_law-1.0)
    Lx = np.log10(Lum)
    delta_L = np.sqrt( (4.0 * np.pi * dL * (1.0 + z) * Fx * ( 2.0 * Ddl * (1.0 + z)**(power_law - 2.0) + dL * (power_law - 1.0)) * z_err)**2.0+ (( 4.0 * np.pi * dL**2.0 * (1.0 + z)**(power_law - 1.0) )*Fx_err)**2.0) 
    Lx_err = delta_L / ( Lum * np.log(10.0) )
    if len(Lx)>1:
        return Lx
    else:
        return Lx[0]


def old_get_luminosity( Fx ,Fx_err, z, z_err=0, power_law=None):
    if power_law is None: power_law = LF_config.pl#_min + (LF_config.pl_max - LF_config.pl_min) * random.random()
    #print power_law
    #print Fx, type(Fx), z, type(z), isinstance(Fx, float) , isinstance(z, float)
    #print Fx, type(z), isinstance(z, np.float32), isinstance(z, np.float64)
    #print (isinstance(Fx, np.float32) or isinstance(Fx, np.float64) or isinstance(Fx, float)) and (isinstance(z, np.float32) or isinstance(z, np.float64) or isinstance(z, float))
    if (isinstance(Fx, np.float32) or isinstance(Fx, np.float64) or isinstance(Fx, float)) and (isinstance(z, np.float32) or isinstance(z, np.float64) or isinstance(z, float)):
        dL = luminosity_dist(z)
        Ddl = derivative_lum_dist(z)
        Lum = Fx*4.0*np.pi*(dL**2.0)*(1.0+z)**(power_law-1.0)
        #print Lum, z
        if Lum>0:
            logL = np.log10(Lum)
        else:
            logL = 0
        #if logL>42.0:
        Lx = logL
        delta_L = np.sqrt( (4.0 * np.pi * dL * (1.0 + z) * Fx * ( 2.0 * Ddl * (1.0 + z)**(power_law - 2.0) + dL * (power_law - 1.0)) * z_err)**2.0+ (( 4.0 * np.pi * dL**2.0 * (1.0 + z)**(power_law - 1.0) )*Fx_err)**2.0) 
        if Lum>0:
            Lx_err = delta_L / ( Lum * np.log(10.0) )
        else:
            Lx_err = 0 
    else:    
        Lx = []
        Lx_err = []
        if z_err == 0:
            z_err = np.zeros(len(z))
        for i in np.arange(0,len(z)):
            dL = luminosity_dist(z[i])
            Ddl = derivative_lum_dist(z[i])
            Lum = Fx[i]*4.0*np.pi*(dL**2.0)*(1.0+z[i])**(power_law-1.0)
            logL = np.log10(Lum)
            #if logL>42.0:
            Lx.append(logL)
            delta_L = np.sqrt( (4.0 * np.pi * dL * (1.0 + z[i]) * Fx[i] * ( 2.0 * Ddl * (1.0 + z[i])**(power_law - 2.0) + dL * (power_law - 1.0)) * z_err[i])**2.0
                            + (( 4.0 * np.pi * dL**2.0 * (1.0 + z[i])**(power_law - 1.0) )*Fx_err[i])**2.0 )
            Lx_err.append( delta_L / ( Lum * np.log(10.0) ) )
    return Lx, Lx_err

def get_flux(Lx,z,power_law=LF_config.pl):
    dL = luminosity_dist(z)
    return np.power(10.,Lx) / (4.*np.pi*np.power(dL,2.)*np.power((1.+z),(power_law-1.)))

def get_V( Lx, zmin, zmax, zstep):     
    redshifts = np.linspace(zmin, zmax, (zmax-zmin)/zstep)
    dv = []   
    z = []
    for red in redshifts:
        a = get_area( get_flux(Lx, red) )
        V = dif_comoving_Vol(red, a)*3.4036771e-74# vol in Mpc^3
        if V<1.0e-60: break
        dv.append( V ) # vol in Mpc^3
        z.append(red)
    return sum(dv)*zstep

def get_Vint( Lx, zmin, zmax, zstep): 
   
    redshifts = np.linspace(zmin, zmax, (zmax-zmin)/zstep)
    dv = []   
    z = []
    for red in redshifts:
        a = get_area( get_flux(Lx, red) )
        V = dif_comoving_Vol(red, a)*3.4036771e-74# vol in Mpc^3
        if V<1.0e-60: break
        dv.append( V ) # vol in Mpc^3
        z.append(red)
    integral = scipy.integrate.simps(dv, z)
    return integral

def dV_dz(l,z):
    flux = get_flux(l, z)
    area = get_area(flux)
    return dif_comoving_Vol(z, area)*3.4036771e-74 # vol in Mpc^3

def return_area( LL, zz):
    Flux = get_flux(LL, zz)
    area = return_area(Flux)
    return area

def Dz_DV( LLx, zz):

    Flux = np.vectorize(get_flux)
    temp_Fx = Flux(LLx, zz)

    dv = np.vectorize(dif_Vc)
    dvc = dv(zz)
    
    Data = zip(temp_Fx, dvc*3.4036771e-74) # vol in Mpc^3
    
    return Data
# print "Source ok"
if __name__ == "__main__":
    print get_flux(42,3)
    t1=time.time()
    print get_luminosity(1.48542303838e-17,1e-14,3)
    print time.time()-t1
    t2 = time.time()
    print old_get_luminosity(1.48542303838e-17,1e-14,3.0)
    print time.time()-t2
    
    