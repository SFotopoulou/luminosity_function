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
z_min_Vmax, z_max_Vmax, z_Vmax, L_min_Vmax, L_max_Vmax, L_Vmax, N_Vmax, dPhi_Vmax, dPhi_err_Vmax = np.loadtxt( path + 'data/Vmax_dPhi_literature.dat', unpack=True )

# LF grid
redshift = np.unique(z_Vmax)
#luminosity = np.linspace(LF_config.Lmin, LF_config.Lmax, 30)

# model
models = ['LDDE']#, 'LADE', 'ILDE', 'PDE', 'PLE']

for model in models:      
    LF_interval = np.loadtxt(path + 'data/dPhi_interval_'+model+'.dat')
    
    # L z dPhi_low_90 dPhi_high_90 dPhi_low_99 dPhi_high_99 dPhi_mode dPhi_mean dPhi_median
    z0 = np.where(LF_interval[:,1]==0)
    Phi0 = LF_interval[z0, 6][0]

    L = LF_interval[z0, 0][0]
        
    fig = plt.figure(figsize=(10,10))
    fig.subplots_adjust(left=0.13, top=0.96, bottom=0.10, right=0.98, wspace=0.03, hspace=0.03)   
    
    for z in redshift:
        # model fit
        plt_indx = np.where(redshift==z)[0][0]+1
        fig.add_subplot( 3, 3, plt_indx)
        zz = np.where( LF_interval[:,1] == z )

        Phi = LF_interval[zz, 6][0]
        Phi_low = LF_interval[zz, 2][0]
        Phi_high = LF_interval[zz, 3][0]
        Phi_mode1 = LF_interval[zz, 9][0]
        Phi_mode2 = LF_interval[zz, 10][0]
        
        plt.fill_between(L, Phi_low, Phi_high, color='gray', alpha=0.5)
        plt.plot(L, Phi0, 'k--')
        plt.plot(L, Phi, 'k-')   
        # best fit paramters from 2 modes of parameters distribution
        plt.plot(L, Phi_mode1, 'r--')   
        plt.plot(L, Phi_mode2, 'r:')   
        
        plt.yticks([-10, -8, -6, -4, -2], size='x-small', visible=False)
        plt.xticks([42, 43, 44, 45], size='x-small', visible=False)
        if plt_indx in set([1, 4, 7]): plt.yticks(visible=True)
        if plt_indx in set([7, 8, 9]): plt.xticks(visible=True)
        
        xmin = 41
        xmax = 46
        ymin = -11
        ymax = -1
        plt.ylim([ymin, ymax])
        plt.xlim([xmin, xmax])
        
        if plt_indx == 4 : plt.ylabel('$\mathrm{Log[d\Phi/logL_x/(Mpc^{-3})]}$')
        if plt_indx == 8 : plt.xlabel('$\mathrm{Log[L_x/(erg/sec)]}$')

        # Vmax
        V_mask = np.where( z_Vmax==z )
        Vmax_zmin = z_min_Vmax[V_mask][0]
        Vmax_zmax = z_max_Vmax[V_mask][0]
        
        Vmax_dPhi = dPhi_Vmax[V_mask]
        Vmax_dPhi_err = dPhi_err_Vmax[V_mask]
        
        Vmax_L = L_Vmax[V_mask]
        Vmax_L_minErr = L_Vmax[V_mask] - L_min_Vmax[V_mask]
        Vmax_L_maxErr = L_max_Vmax[V_mask] - L_Vmax[V_mask]
        
        Vmax_count = N_Vmax[V_mask]
        
        plt.errorbar(Vmax_L, Vmax_dPhi, Vmax_dPhi_err, [Vmax_L_minErr, Vmax_L_maxErr], 'ko', markersize=7)
        
        for i in range(0, len(Vmax_count) ):
            if Vmax_count[i]>0: plt.text(Vmax_L[i], Vmax_dPhi[i]+0.6, int(Vmax_count[i]), size=13)        

        plt.text(xmin + (xmax-xmin)*0.07, ymin + (ymax-ymin)*0.07, str(Vmax_zmin)+'<z<'+str(Vmax_zmax), size='x-small')        

    #plt.savefig('../XXL_full/dPhi_'+model+'.pdf', dpi=200)



    plt.show()