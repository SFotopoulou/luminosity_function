import numpy as np
from cosmology import luminosity_dist,dif_comoving_Vol,dif_Vc#,derivative_lum_dist
from Survey import get_area

import scipy.integrate

def get_luminosity( Fx ,Fx_err, z, power_law):
    Fx, Fx_err, z, power_law = np.asarray(Fx), np.asarray(Fx_err), np.asarray(z), np.asarray(power_law)
    dL = np.asarray( [luminosity_dist(zi) for zi in np.nditer(z)] )
    Lum = Fx*4.0*np.pi*(dL**2.0)*(1.0+z)**(power_law-1.0)
    Lx = np.log10(Lum)

    if len(Lx)>1:
        return Lx
    else:
        return Lx[0]

def get_flux(Lx,z,power_law):
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
        print(a, V)
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
    area = get_area(Flux)
    return area

def Dz_DV( LLx, zz):

    Flux = np.vectorize(get_flux)
    temp_Fx = Flux(LLx, zz)

    dv = np.vectorize(dif_Vc)
    dvc = dv(zz)
    
    Data = zip(temp_Fx, dvc*3.4036771e-74) # vol in Mpc^3
    
    return Data

if __name__ == "__main__":
    print(get_flux(42,3))
    print(get_luminosity(1.48542303838e-17,1e-14,3))    
    
