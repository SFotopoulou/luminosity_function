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
    area = get_area(temp_Fx)   
   
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

get_data()
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
np.savetxt( path + 'data/Volume_for_N_1000.dat', N)
print "total time needed", time.time() - t1