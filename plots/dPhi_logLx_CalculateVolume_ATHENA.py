import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import itertools
import sys
import copy
# Add the ptdraft folder path to the sys.path list
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/models')
from LFunctions import PLE,PDE,ILDE,LADE,Fotopoulou
from AGN_LF_config import LF_config
from cosmology import *
from Survey import *
from Source import *
from SetUp_data import Set_up_data
import astroML as ML
from matplotlib.ticker import NullFormatter
from astropy.io import fits as pyfits
from scipy import interpolate
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

large_flux, large_curve = np.loadtxt('/home/Sotiria/Documents/Luminosity_Function/data/ATHENA/area_curves/athenapp_wfi_0030ks_psf5.0_hard.acurve', unpack=True, usecols=[0,1])
medium_flux, medium_curve = np.loadtxt('/home/Sotiria/Documents/Luminosity_Function/data/ATHENA/area_curves/athenapp_wfi_0100ks_psf5.0_hard.acurve', unpack=True, usecols=[0,1])
small_flux, small_curve = np.loadtxt('/home/Sotiria/Documents/Luminosity_Function/data/ATHENA/area_curves/athenapp_wfi_0300ks_psf5.0_hard.acurve', unpack=True, usecols=[0,1])
tiny_flux, tiny_curve = np.loadtxt('/home/Sotiria/Documents/Luminosity_Function/data/ATHENA/area_curves/athenapp_wfi_1000ks_psf5.0_hard.acurve', unpack=True, usecols=[0,1])

e_flux = np.power(10.0, large_flux) 

total_area = ( 4.0 * tiny_curve + 20.0 * small_curve + 75.0 * medium_curve + 250.0 * large_curve ) * 0.00030461742

#e_flux = np.power(10., e_flux)
#print e_flux
#print np.log10(e_flux)
#e_curve = e_curve * 0.00030461742 * 41235 
#print min(e_flux), max(e_flux)
plt.plot(e_flux, total_area, 'k')
plt.plot(e_flux, 250.0*large_curve)
plt.plot(e_flux, 75.0*medium_curve)
plt.plot(e_flux, 20.0*small_curve)
plt.plot(e_flux, 4.0*tiny_curve)
plt.ylabel('$area/(deg^2)$')
plt.xlabel('$F_{2-10\,keV}/(erg\,s^{-1}\,cm^{-2})$')
plt.xscale('log')
plt.xlim([5e-17, 1e-13])
plt.savefig('/home/Sotiria/Documents/Luminosity_Function/data/ATHENA/area_curves/athena_curve_survey.pdf', dpi=300)
plt.show()


e_area = interpolate.interp1d(e_flux, total_area)
ATHENA_curve = zip(e_flux, total_area)
# save area curve
np.savetxt('/home/Sotiria/Documents/Luminosity_Function/data/ATHENA/area_curves/total_ATHENA_curve.txt', ATHENA_curve)
 
def calc_Vol(Lmin, Lmax, zmin, zmax, zpoints=10, Lpoints=10):
    LL = np.array([np.ones( (zpoints), float )*item for item in 
                   np.linspace(Lmin, Lmax, Lpoints)])
    # make LL 1D
    L = LL.ravel()
    # repeat as many times as Lpoints
    Z = np.tile(np.logspace(np.log10(zmin), np.log10(zmax), zpoints), Lpoints) 

# Set up grid for survey integral
    vecFlux = np.vectorize(get_flux)
    temp_Fx = vecFlux(L, Z)
#    print Lmin, Lmax, zmin, zmax
#    print temp_Fx
    temp_Fx1 = np.where( temp_Fx > max(e_flux) , max(e_flux) , temp_Fx)
    temp_Fx2 = np.where( temp_Fx1 < min(e_flux), min(e_flux), temp_Fx1)
    area = e_area(temp_Fx2)
#    print temp_Fx2
##    print
#    raw_input()
    vecDifVol = np.vectorize(dif_comoving_Vol) 
    DVc = np.where( area>0, vecDifVol(Z, area), 0) 
    DVcA = DVc*3.4036771e-74 # vol in Mpc^3

        
    Redshift_int = Z[0:zpoints]
    Luminosity_int = np.linspace(Lmin, Lmax, Lpoints)
    
    y = []
    
    count_r = xrange(0, Lpoints)
    for count in count_r:
        startz = count * zpoints
        endz = startz + zpoints
        x = DVcA[startz:endz]
        
        int1 = simps(x, Redshift_int, even='last')
        y.append(int1)
    
    DV_int = simps(y, Luminosity_int, even='last')
    return DV_int



path = '/home/Sotiria/workspace/Luminosity_Function/src/LF_plots/forPaper/'
param_path = '/home/Sotiria/workspace/Luminosity_Function/src/MultiNest/safe_keep/'

import time
zbin = 1001
Lbin = 1001
zlim = 4.0
# LF grid
ZZ = np.linspace(LF_config.zmin, zlim, zbin)
LL = np.linspace(LF_config.Lmin, LF_config.Lmax, Lbin)
zbin_width = (ZZ[1] - ZZ[0])/2.0
Lbin_width = (LL[1] - LL[0])/2.0

print
print "Calculating volume"
t1 = time.time()
N = []
for j in range(0, len(LL)-1):
    print LL[j]
    t2 = time.time()
    vol = []
    for i in range(0, len(ZZ)-1):    
        #print round(LL[j],3), round(ZZ[i],3), ph 
        vol.append(calc_Vol(LL[j], LL[j+1], ZZ[i], ZZ[i+1]))
        
    vol = np.array(vol)
    N.append(vol) 
    print "redshift loop:", time.time() - t2
np.savetxt( path + 'data/Volume_for_ATHENA.dat', N)
print "total time needed", time.time() - t1