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
import astroML as ML

path = '/home/Sotiria/workspace/Luminosity_Function/src/LF_plots/forPaper/'
param_path = '/home/Sotiria/workspace/Luminosity_Function/src/MultiNest/'

lum = 'high'

# logLx<44.15
if lum=='low':
    z_COSMOS = [3.09, 3.29, 3.45]
    z_COSMOS_err = [0.09, 0.11, 0.05]
    N_COSMOS = [np.log10(5.63e-06), np.log10(3.56e-06), np.log10(1.04e-05)]
    N_COSMOS_err = [0.434*1.86e-06/(5.63e-06), 0.434*1.28e-06/(3.56e-06), 0.434*4.86e-6/(1.04e-05)]
    
    z_ikeda = [4]
    N_ikeda = [ np.log10(1.3e-6)]
    N_ikeda_err = [0.434*0.6e-06/(1.3e-06)]
    
    z_glikman = [4]
    N_glikman = [np.log10(4.6e-6)]
    N_glikman_err = [0.434*2.0e-06/(4.6e-06)]
else:
    # logLx>44.15
    z_COSMOS =[3.1, 3.3, 3.6,  4.05, 4.9, 6.2 ]  
    z_COSMOS_err = [0.1, 0.1, 0.2, 0.25, 0.6, 0.7 ]
    N_COSMOS = [np.log10(5.17e-06), np.log10(4.18e-06), np.log10(2.95e-06), np.log10(1.18e-06), np.log10(1.04e-06), -6.5]
    N_COSMOS_err = [0.434*1.383e-06/(5.17e-06), 0.434*1.394e-06/(4.18e-06), 0.434*8.518e-07/(2.95e-06), 0.434*4.84e-07/(1.18e-06), 0.434*3.908e-07/(1.04e-06), 0.08]
    N_COSMOS_limit = [False, False, False, False, False, True]
     
    z_XMM_COSMOS =[3.075, 3.275, 3.85]  
    z_XMM_COSMOS_err = [0.075, 0.125, 0.45 ]
    N_XMM_COSMOS = [-5.357, -5.37, -5.96]
    N_XMM_COSMOS_err = [0.434*1.325e-06/np.power(10, -5.357), 0.434*1.065e-06/np.power(10, -5.37), 0.434*2.9e-07/np.power(10.0,-5.96)]
    
# Vmax Points
zbin_min_Vmax, zbin_max_Vmax, z_min_Vmax, z_max_Vmax, z_Vmax, Lbin_min_Vmax, Lbin_max_Vmax, L_min_Vmax, L_max_Vmax, L_Vmax, N_Vmax, Ndensity_Vmax, Ndensity_err_Vmax = np.loadtxt( path + 'data/Vmax_Ndensity.dat', unpack=True )

# LF grid
#redshift = np.linspace(0., 4., 10)
#luminosity = [42., 43., 44., 45., 46]

if lum == 'low':
    luminosity = [43.56, 44.15]
else:
    luminosity = [44.15, 45.1]
# model
models = ['LDDE']#, 'LADE', 'ILDE', 'PDE', 'PLE']
fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(left=0.19, top=0.94, bottom=0.10, right=0.96, wspace=0.03, hspace=0.03)   

for model in models:      
    N_interval = np.loadtxt(path + 'data/Ndensity_interval_forCosmos'+model+'.dat')
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
        try:
            N_mode3 = N_interval[LL, 12][0]
            N_mode4 = N_interval[LL, 13][0]
        except:
            pass
        
        plt.fill_between(zz, N_low, N_high, color='gray', alpha=0.15)
        plt.plot(zz, N, ls=ls.next(), color=lc.next(), label='$\mathrm{'+str(luminosity[i])+'< logL_x<'+str(luminosity[i+1])+'}$')   
        
        # best fit paramters from 2 modes of parameters distribution
        #plt.plot(zz, N_mode1, 'r--')   
        #plt.plot(zz, N_mode2, 'r:')   
        #try:
        #    plt.plot(zz, N_mode3, 'b--')   
        #    plt.plot(zz, N_mode4, 'b:')
        #except:
        #    pass
        
        #plt.yticks([-10, -8, -6, -4, -2], size='x-small', visible=True)
        #plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], size='x-small', visible=True)

         
#        plt.errorbar(Vmax_z, Vmax_dN, Vmax_dN_err, [Vmax_z_minErr, Vmax_z_maxErr], ls='', ecolor='k', color=fc.next(), marker=points.next(), markersize=12)
        if lum=='low':
            plt.errorbar(z_COSMOS, N_COSMOS, xerr= z_COSMOS_err, yerr = N_COSMOS_err, marker='o', color='r', ls = ' ', markersize=14)
            plt.errorbar(z_ikeda, N_ikeda,  yerr = N_ikeda_err, marker='s', markerfacecolor='w', markeredgecolor='black',ecolor='k',ls=' ',markersize=14)
            plt.errorbar(z_glikman, N_glikman, yerr = N_glikman_err, marker='s', color='black', ls=' ',markersize=14)
        if lum=='high':
            plt.errorbar(z_COSMOS, N_COSMOS, xerr= z_COSMOS_err, yerr = N_COSMOS_err, lolims = N_COSMOS_limit, marker='o', color='r', ls = ' ',markersize=14)
            plt.errorbar(z_XMM_COSMOS, N_XMM_COSMOS,  xerr= z_XMM_COSMOS_err, yerr = N_XMM_COSMOS_err, marker='o', color='g', ls=' ',markersize=14)

        xmin = 2.5
        if lum == 'low':
            xmax = 4.5
        else:
            xmax = 7.0
        ymin = -7.0
        ymax = -4.0

        plt.ylim([ymin, ymax])
        plt.xlim([xmin, xmax])
        
        plt.ylabel('$\mathrm{Number\,Density/(Mpc^{-3})}$')
        plt.xlabel('$\mathrm{Redshift}$')
        plt.legend(loc=3)
     
        
        plt.savefig('plots/Ndensity_onlyCOSMOS_'+lum+'_'+model+'.pdf', dpi=200)
    
    #plt.title(model)
    plt.show()