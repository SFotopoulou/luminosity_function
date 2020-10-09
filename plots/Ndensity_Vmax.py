import sys
# Add the module path to the sys.path list
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/no_uncertainties_included/MLE/MLE_modules/')
#
import time, itertools
from numpy import arange,savetxt, linspace,logspace,sqrt,log10,array,ones,tile, median, asarray, genfromtxt, power
from Source import Source
import matplotlib.pyplot as plt
import scipy.integrate
from LFunctions import Models
from scipy.integrate import simps
from parameters import Parameters

params = Parameters()
Lmin, Lmax = Parameters.L(params)
zmin, zmax = Parameters.z(params)
model = Models()

zbin = [0.01, 0.2, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 3.0, 4.0]
Lbin = [41.0, 42.0, 43.0, 44.0, 45.0, 46.0]
linecolor = (0., 0., 0.)
fillcolor = (0.75, 0.75, 0.75)
pointcolor = (0., 0., 0.)
linestyles = itertools.cycle( ['-', '--',':','-.','steps'] )
colors = itertools.cycle( ['k', 'k','k','k','gray'] )

# Observations
s = Source('data')
ID, F, F_err, Z_i, Z_flag, Field = s.get_data(zmin, zmax)
Lx_i, Lx_err = s.get_luminosity(F, F_err, Z_i)
#    Sort according to luminosity
Lx_s, z_s, F_s = zip(*sorted(zip(Lx_i, Z_i, F))) 
save_data = []

for j in arange(0,len(Lbin)-1):    
    for i in arange(0,len(zbin)-1):
        count = 0
        dN_dV = 0.0
        err_dN_dV = 0.0
        Ll = []
        Zz = []
        for Lx, z, f in zip(Lx_s, z_s, F_s):
            if zbin[i] <= z < zbin[i+1] and Lbin[j] <= Lx < Lbin[j+1]:
                count = count + 1
                Vmax =  s.get_V(Lx, zbin[i], zbin[i+1], 0.01) 
                dN_dV = dN_dV + 1.0/Vmax
                err_dN_dV = err_dN_dV + power( (1.0/Vmax), 2.0)
                
                Ll.append(Lx)        
                Zz.append(z)
        err_dN_dV = sqrt( err_dN_dV )
        if count>0:
            print median(Zz), median(Ll), count, log10(dN_dV), 0.434*err_dN_dV/dN_dV, median(Ll)-Lbin[j], Lbin[j+1]-median(Ll)
            datum = [median(Zz), median(Ll), count, log10(dN_dV), 0.434*err_dN_dV/dN_dV, median(Ll)-Lbin[j], Lbin[j+1]-median(Ll)]
            save_data.append(datum)
#savetxt('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/Ndensity_files/Ndensity_Vmax.dat', save_data)