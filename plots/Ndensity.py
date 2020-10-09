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
from AGN_LF_config import LF_config
#import astroML as ML

path = '/home/Sotiria/workspace/Luminosity_Function/src/LF_plots/XXL_full/'
param_path = '/home/Sotiria/workspace/Luminosity_Function/src/MultiNest/'

# Vmax Points
zbin_min_Vmax, zbin_max_Vmax, z_min_Vmax, z_max_Vmax, z_Vmax, Lbin_min_Vmax, Lbin_max_Vmax, L_min_Vmax, L_max_Vmax, L_Vmax, N_Vmax, Ndensity_Vmax, Ndensity_err_Vmax = np.loadtxt( path + 'data/Vmax_Ndensity.dat', unpack=True )

# LF grid
redshift = np.linspace(0., 4., 10)
luminosity = [42., 43., 44., 45.]

# model
models = ['LDDE']#, 'LADE', 'ILDE', 'PDE', 'PLE', 'Ueda14', 'Fotopoulou2', 'Fotopoulou3']
fig = plt.figure(figsize=(10,10))
fig.subplots_adjust(left=0.15, top=0.96, bottom=0.10, right=0.96, wspace=0.03, hspace=0.03)   

for model in models:      
    N_interval = np.loadtxt(path + 'data/Ndensity_interval_'+model+'.dat')
    ls = itertools.cycle(['-', ':', '-.', '-', '--'])
    lc = itertools.cycle(['k', 'k', 'k', 'gray', 'gray'])
    fc = itertools.cycle(['k', 'white', 'k', 'gray', 'gray'])
    points = itertools.cycle(['o', 's', '^', 'o', 'v'])
   

    for i in range(0, len(luminosity)-1):
        L = luminosity[i]
        LL = np.where( N_interval[:,0] == L )
        
        zz =  N_interval[LL, 2][0]
        
        N = N_interval[LL, 7][0]
        N_low = N_interval[LL, 3][0]
        N_high = N_interval[LL, 4][0]
        N_mode1 = N_interval[LL, 10][0]
        N_mode2 = N_interval[LL, 11][0]
        
        plt.fill_between(zz, N_low, N_high, color='gray', alpha=0.15)
        plt.plot(zz, N, ls=ls.next(), color=lc.next(), label='$\mathrm{'+str(luminosity[i])+'< logL_x<'+str(luminosity[i+1])+'}$')   
        
        # best fit paramters from 2 modes of parameters distribution
        #plt.plot(zz, N_mode1, 'r--')   
        #plt.plot(zz, N_mode2, 'r:')   

        #plt.yticks([-10, -8, -6, -4, -2], size='x-small', visible=True)
        #plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], size='x-small', visible=True)

        # Vmax
    #for i in range(0, len(luminosity)-1):
        #L = luminosity[i]
        
        V_mask = np.where( Lbin_min_Vmax==L )
        
        Vmax_zmin = zbin_min_Vmax[V_mask][0]
        Vmax_zmax = zbin_max_Vmax[V_mask][0]
        #Vmax_z = z_Vmax[V_mask][0]
        
        Vmax_Lmin = Lbin_min_Vmax[V_mask][0]
        Vmax_Lmax = Lbin_max_Vmax[V_mask][0]
        
        Vmax_N = N_Vmax[V_mask]
        Vmax_dN = Ndensity_Vmax[V_mask]
        Vmax_dN_err = Ndensity_err_Vmax[V_mask]
        
        
        Vmax_z = z_Vmax[V_mask]
        Vmax_z_minErr = z_Vmax[V_mask] - zbin_min_Vmax[V_mask]
        Vmax_z_maxErr = zbin_max_Vmax[V_mask] - z_Vmax[V_mask]
        
        Vmax_count = N_Vmax[V_mask]
        
        plt.errorbar(Vmax_z, Vmax_dN, Vmax_dN_err, [Vmax_z_minErr, Vmax_z_maxErr], ls='', ecolor='k', color=fc.next(), marker=points.next(), markersize=12)
        for k in range(0, len(Vmax_N) ):
            plt.text(Vmax_z[k]-0.13, Vmax_dN[k]+0.115, int(Vmax_N[k]), size=20)
        
        xmin = -0.1
        xmax = 4.1
        ymin = -8.0
        ymax = -2.5

        plt.ylim([ymin, ymax])
        plt.xlim([xmin, xmax])
        
        plt.ylabel('$\mathrm{Number\,Density/(Mpc^{-3})}$')
        plt.xlabel('$\mathrm{Redshift}$')
                
        #plt.title(model)
        plt.legend(loc=4)       
     #   plt.savefig('plots/Ndensity_'+model+'.pdf', dpi=200)
    #plt.clf()

    plt.show()