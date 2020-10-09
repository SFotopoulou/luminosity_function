import sys
# Add the module path to the sys.path list
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
#
import time, itertools
from numpy import arange,savetxt, linspace,logspace,sqrt,log10,array,ones,tile, median, asarray
from Source import *
import matplotlib.pyplot as plt
from cosmology import *
import scipy.integrate
from LFunctions import *
from scipy.integrate import simps
from parameters import Parameters

params = Parameters()
Lmin, Lmax = Parameters.L(params)
zmin, zmax = Parameters.z(params)

# Observations
s = Source('data')
ID, F, F_err, Z_i, Z_flag, Field = s.get_data(zmin, zmax)
Lx_i, Lx_err = s.get_luminosity(F, F_err, Z_i)

# Normalization
ndata = len(ID)

Lx_s, z_s, F_s = zip(*sorted(zip(Lx_i, Z_i, F))) 
# cute
#Lbin_cycle = itertools.cycle([[42.0, 42.75, 43.0, 43.25, 43.5, 43.75, 44.],
##                             [42.3,42.8, 43.2, 43.3, 43.4, 43.5, 43.6, 43.7, 43.8, 43.9,44.0, 44.35, 44.8],
##                             [42.75, 43.25, 43.5, 43.6, 43.7, 43.8, 43.9, 44.0, 44.1, 44.2, 44.3, 44.4, 44.75, 45.25],
##                             [43.4, 43.8, 44.0, 44.5,44.7, 45.0],
###                             [43.4, 43.8, 44.0, 44.5, 45.0]])
# For thesis
Lbin_cycle = itertools.cycle([[41.17, 42.0, 42.75, 43.0, 43.25, 43.5, 43.75, 44.],
                              [42.3,42.8, 43.2, 43.3, 43.4, 43.5, 43.6, 43.7, 43.8, 43.9,44.0, 44.35, 44.8],
                              [42.75, 43.25, 43.5, 43.6, 43.7, 43.8, 43.9, 44.0, 44.1, 44.2, 44.3, 44.4, 44.75, 45.25],
                              linspace(43.0, 45.0, 5),
                              linspace(43.0, 45.0, 4)]) 
### test
##Lbin_cycle = itertools.cycle([linspace(40.0, 46.0, 20),
##                              linspace(40.0, 46.0, 20),
##                              linspace(40.0, 46.0, 20),
##                              linspace(40.0, 46.0, 20),
##                              linspace(40.0, 46.0, 20)])

lcolors = itertools.cycle(['blue','cyan','green','orange','red'])
#zlabel = itertools.cycle(['0.25', '0.75', '1.4', '2.5', '3.5'])
zlabel = []
zbin = [0.01, 0.5, 1.0, 2.0, 3.0, 4.0]

#Lbin_cycle = []
#for i in arange(0,len(zbin)-1):
#    luminosity =[]
#    for Lx, z, f in zip(Lx_s, z_s, F_s):
#        if zbin[i] <= z < zbin[i+1]:
#            luminosity.append(Lx)    
#  # Lbin_cycle.append(linspace(max(42.0,min(luminosity)),  max(luminosity), 20 ))
##Lbin_cycle = itertools.cycle(Lbin_cycle)              


fig = plt.figure(figsize=(8,8))
fig.add_subplot(111)
fig.subplots_adjust(left=0.16, right=0.97, top=0.98, bottom=0.15,wspace=0.43, hspace=0.23)

save_data = []
for i in arange(0,len(zbin)-1):
    LF = []

    Lbin = Lbin_cycle.next()            
#    Find median redshift
    Zz = []
    for Lx, z, f in zip(Lx_s, z_s, F_s):
        if zbin[i] <= z < zbin[i+1]:
            Zz.append(z)    

    #dz = (zbin[i+1]-zbin[i])/0.01
    dz = 5
    zspace = linspace(zbin[i], zbin[i+1], dz)
    
    for j in arange(0,len(Lbin)-1):        
#    Calculate denominator- 2D integral
#        dL = (Lbin[j+1] - Lbin[j]) / 0.01
        dL = 5
        Lspace = linspace(Lbin[j], Lbin[j+1], dL)   

        int = []
        for z in zspace:
            dV = []
            for l in Lspace:
                dV.append( s.dV_dz(l, z) )
            int.append(scipy.integrate.simps(dV, Lspace))
        integ = scipy.integrate.simps(int, zspace)

#    Count sources per bin
        Ll = []
        count = 0
        for Lx, z, f in zip(Lx_s, z_s, F_s):
            if zbin[i] <= z < zbin[i+1] and Lbin[j] <= Lx < Lbin[j+1]:
                count = count + 1
                Ll.append(Lx)        
        temp_Phi = count/integ
        temp_err = sqrt(count)/integ

        datum = [median(Zz), median(Ll), count, log10(temp_Phi), 0.434*temp_err/temp_Phi, median(Ll)-Lbin[j], Lbin[j+1]-median(Ll)]
        LF.append(datum)
        save_data.append(datum)
    LFU_model = []
    LFF_model = []
    ll = linspace(42.0,46.0)
        
    redshift = asarray(LF)[:,0]
    luminosity = asarray(LF)[:,1]
    number = asarray(LF)[:,2]
    Phi = asarray(LF)[:,3]
    err_Phi = asarray(LF)[:,4]
    lbin_l = asarray(LF)[:,5]
    lbin_h = asarray(LF)[:,6]

    name = str("%.2f") % median(Zz)
    print number
    colors = lcolors.next()
    plt.errorbar(luminosity, Phi, ls = '--',yerr=err_Phi, xerr=[lbin_l, lbin_h], label=name, markersize = 8.5,color=colors, marker='o',markeredgecolor='black')
savetxt("PC_Vmax_Points.txt", save_data)
plt.xticks([41, 42, 43, 44, 45, 46],fontsize='medium')
plt.yticks([-7.0, -6.0, -5.0, -4.0],fontsize='medium')
plt.xlim([41., 46.2])
plt.ylim([-7.5,-3.5])
plt.legend(loc=0)
plt.xlabel("$Log[Lx/(erg/sec)]$",fontsize='medium')
plt.ylabel("$Log[d\Phi/dLogLx/(Mpc^{-3})]$",fontsize='medium')
#for ext in ['pdf','svg','eps','jpg','png']:
#    plt.savefig("./output_files/new_PC_Vmax."+ext)
plt.show()
