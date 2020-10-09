import sys
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
from AGN_LF_config import LF_config
import numpy as np
import scipy.integrate
import time

#global H0, tH, DH, Om, Ol, Ok

H0 = LF_config.H0*0.32407788499e-19 # 1/s
tH = 1./H0
DH = LF_config.c/H0
Om = LF_config.Om
Ol = LF_config.Ol
Ok = LF_config.Ok
        
        
def Ez(z):
    return 1. / np.sqrt(Om * np.power((1. + z), 3.) + Ok * np.power((1.+z), 2.) + Ol)

def comoving_LOS(z):
    integral, error = scipy.integrate.quad(lambda x: Ez(x), 0.0, z, limit=1000)
    Dc = DH * integral
    return Dc

def comoving_trns(z):
    Dc = comoving_LOS(z)    
    if Ok == 0:
        Dm = Dc 
    if Ok > 0:
        Dm = DH * (np.sqrt(1./Ok) * np.sinh(np.sqrt(Ok) * Dc/DH))
    if Ok < 0:
        Dm = DH * (np.sqrt(1./abs(Ok)) * np.sin(np.sqrt(abs(Ok)) * Dc/DH))
    return Dm

def angular_diameter(z):
    Dm = comoving_trns(z)
    Da = Dm/(1.+z)
    return Da

def two_point_angular_diameter(z1,z2):
    DM1 = comoving_trns(z1)
    DM2 = comoving_trns(z2)
    D12 = (DM2*np.sqrt(1+Ok*(DM1**2/DH**2))-DM1*np.sqrt(1+Ok*(DM2**2/DH**2)))/(1+z2)
    return D12

def luminosity_dist(z):
    Dm = comoving_trns(z)
    DL = Dm * (1.+z)
    return DL

def derivative_lum_dist(z):
    los = comoving_LOS(z)
    E = Ez(z)
    return los + (1.0 + z) * DH * E
    
def dist_modulus(z):
    DL = luminosity_dist(z)*1.0e6/3.08568e24
    DMod = 5.*np.log10(DL/10.)
    return DMod

def dif_comoving_Vol(z,area):
    Da = angular_diameter(z)
    DVc_Dz = DH*np.power((1.+z),2.)*np.power(Da,2.)*Ez(z)*area # in cm^3
    return DVc_Dz

def dif_Vc(z):
    Da = angular_diameter(z)
    DVc = DH*np.power((1.+z),2.)*np.power(Da,2.)*Ez(z)
    return DVc

def comoving_Vol(zmin,zmax,area):
    integral, error = scipy.integrate.quad(lambda x: dif_comoving_Vol(x,area), zmin, zmax)
    return integral

def lookbacktime(zmax):
    integral, error = scipy.integrate.quad(lambda z: Ez(z)/(1.+z) , 0.0, zmax)         
    return integral*tH

def dlookbacktime(z):
    return tH*Ez(z)/(1.+z)
    
#comoving volume 151.057125321 Gpc^3
#0.225499868393
#angular distance 1651.913332 Mpc
#0.0101449489594
#luminosity distance 6607.653329 Mpc
#0.0111730098724
#comoving line of sight 3303.826665 Mpc
#0.0101490020752
#
#comoving volume 151.057453917 Gpc^3
#angular distance 1651.91453006 Mpc
#luminosity distance 6607.65812022 Mpc

if __name__ == "__main__":
    redshift = np.linspace(0,5,10000)
    
    distance = []
    import matplotlib.pyplot as plt
    for z in redshift:
        distance.append( dif_Vc(z)/np.power(DH,3) )
    
    plt.plot(redshift, distance)
    plt.show()
    t1 = time.time()
    print("comoving volume",comoving_Vol(0.0,1,4.0*np.pi)*np.power(3.2407788499e-28,3), "Gpc^3")
    t2 = time.time()
    print(t2-t1)
    print("angular distance", angular_diameter(1)/3.08568e24, "Mpc")
    t3 = time.time()
    print(t3-t2)
    
    print("luminosity distance", luminosity_dist(1)/3.08568e24, "Mpc")
    t4 = time.time()
    print(t4-t3)
    
    print("comoving line of sight", comoving_LOS(1)/3.08568e24, "Mpc")
    t5 = time.time()
    print(t5-t4)
    
    
    print(Ez(1))
    print(comoving_LOS(1))
    print(comoving_trns(1))
    print(angular_diameter(1))
    print(comoving_Vol(0, 1, 0.2))
    print(angular_diameter(1))
    print(luminosity_dist(1))
    print(derivative_lum_dist(1))
    print(dist_modulus(3))
    #print vec_dif_comoving_Vol(1,0.2)
    print(dif_comoving_Vol(1,0.2))
    print(dif_Vc(1))
    print(comoving_Vol(0, 1, 0.2))
    print(lookbacktime(1))
    print(dlookbacktime(1))
    print(time.time()-t1)
