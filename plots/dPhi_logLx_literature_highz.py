import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import itertools
import sys
import copy
# Add the ptdraft folder path to the sys.path list
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/models')
from AGN_LF_config import LF_config
#import astroML as ML

from LFunctions import Ueda14, LADE, Miyaji15, highz, Fotopoulou
########################################################
class Params: pass

Ueda_params = Params()

Ueda_params.L0 = 43.97
Ueda_params.g1 = 0.96
Ueda_params.g2 = 2.71
Ueda_params.p1 = 4.78
Ueda_params.beta = 0.84
Ueda_params.Lp = 44.0
Ueda_params.p2 = -1.5
Ueda_params.p3 = -6.2
Ueda_params.zc1 = 1.86
Ueda_params.La1 = 44.61
Ueda_params.a1 = 0.29
Ueda_params.zc2 = 3.0
Ueda_params.La2 = 44.0
Ueda_params.a2 = -0.1
Ueda_params.Norm = np.log10(2.91e-6)

XMM_params = Params()

XMM_params.L0 = 43.97+0.3098
XMM_params.g1 = 2.53
XMM_params.g2 = 0.97
XMM_params.p1 = 5.72
XMM_params.p2 = -2.72
XMM_params.zc = 2.19
XMM_params.La = 44.55+0.3098
XMM_params.a = 0.26
XMM_params.Norm = -6.26


Chandra_params = Params()

Chandra_params.L0 = 43.62+0.3098
Chandra_params.g1 = 2.46
Chandra_params.g2 = 0.92
Chandra_params.p1 = 7.02
Chandra_params.p2 = -1.81
Chandra_params.zc = 1.99
Chandra_params.La = 44.48+0.3098
Chandra_params.a = 0.26
Chandra_params.Norm = -6.04

Ueda_params = Params()

Ueda_params.L0 = 43.97
Ueda_params.g1 = 0.96
Ueda_params.g2 = 2.71
Ueda_params.p1 = 4.78
Ueda_params.beta = 0.84
Ueda_params.Lp = 44.0
Ueda_params.p2 = -1.5
Ueda_params.p3 = -6.2
Ueda_params.zc1 = 1.86
Ueda_params.La1 = 44.61
Ueda_params.a1 = 0.29
Ueda_params.zc2 = 3.0
Ueda_params.La2 = 44.0
Ueda_params.a2 = -0.1
Ueda_params.Norm = np.log10(2.91e-6)


Aird_params = Params()

Aird_params.L0 = 44.09
Aird_params.g1 = 0.73
Aird_params.g2 = 2.22
Aird_params.p1 = 4.34
Aird_params.beta = -0.19
Aird_params.Lp = 44.48
Aird_params.p2 = -0.30
Aird_params.p3 = -7.33
Aird_params.zc1 = 1.85
Aird_params.La1 = 44.78
Aird_params.a1 = 0.23
Aird_params.zc2 = 3.16
Aird_params.La2 = 44.46
Aird_params.a2 = 0.13
Aird_params.Norm = -5.72


Miyaji_params = Params()

Miyaji_params.L0 = 44.04
Miyaji_params.g1 = 1.17
Miyaji_params.g2 = 2.80
Miyaji_params.p1 = 5.29
Miyaji_params.p2 = -0.35
Miyaji_params.p3 = -5.6
Miyaji_params.zb0 = 1.1
Miyaji_params.zb2 = 2.7
Miyaji_params.Lb = 44.5
Miyaji_params.a = 0.18
Miyaji_params.b1 = 1.2
Miyaji_params.b2 = 1.5
Miyaji_params.Norm = np.log10(1.56e-6)

Vito_params = Params()

Vito_params.L0 = np.log10(4.92e44)
Vito_params.g1 = 0.66
Vito_params.g2 = 3.71
Vito_params.q = -6.65
Vito_params.b = 2.40
Vito_params.Norm = np.log10(1.19e-5)

Georgakakis_params = Params()

Georgakakis_params.L0 = 44.31
Georgakakis_params.g1 = 0.21
Georgakakis_params.g2 = 2.15
Georgakakis_params.q = -7.46
Georgakakis_params.b = 2.30
Georgakakis_params.Norm = -4.79
#

path = '/home/Sotiria/workspace/Luminosity_Function/src/LF_plots/forPaper/'
param_path = '/home/Sotiria/workspace/Luminosity_Function/src/MultiNest/'


# Vmax Points
zmin, zmax, zmean, Lmin, Lmax, Lmean, N, dPhi, dPhi_err = np.loadtxt( path + 'data/Vmax_dPhi_literature.dat', unpack=True )

# Buchner15 estimation
Buchner15 = np.loadtxt('/home/Sotiria/workspace/Luminosity_Function/src/LF_plots/forPaper/plots/Buchner15_space_density.txt')
zB_min = [3.2]
zB_max = [4.0]
#Buchner_zbin = [np.where((Buchner15[:,2]>=0.1) & (Buchner15[:,3]<=0.3)),\
#                np.where((Buchner15[:,2]>=0.5) & (Buchner15[:,3]<=0.8)),\
#                np.where((Buchner15[:,2]>=1.2) & (Buchner15[:,3]<=1.5)),\
#                np.where((Buchner15[:,2]>=2.1) & (Buchner15[:,3]<=2.7)),\
#                np.where((Buchner15[:,2]>=3.2) & (Buchner15[:,3]<=4.0))]

LogL_B15 = np.unique( (list(Buchner15[:,0][:][:]) + list(Buchner15[:,1][:][:])) )
#print LogL_B15
LogL_B = np.unique( LogL_B15 )#+ 0.5*(LogL_B15[1]-LogL_B15[0]) )

# LF grid
redshift = [3.6]# np.unique(z_min_Vmax)#redshift = np.unique(z_Vmax)
#luminosity = np.linspace(LF_config.Lmin, LF_config.Lmax, 30)
print redshift
# model
models = ['LDDE']#, 'LADE', 'ILDE', 'PDE', 'PLE']
Hopkins_qlf = {0.2: '/home/Sotiria/Software/contributed/Hopkins_QLF/Hopkins07_qlf_z0.2.dat',
               0.65: '/home/Sotiria/Software/contributed/Hopkins_QLF/Hopkins07_qlf_z0.65.dat',
               2.4: '/home/Sotiria/Software/contributed/Hopkins_QLF/Hopkins07_qlf_z2.4.dat',
               3.6: '/home/Sotiria/Software/contributed/Hopkins_QLF/Hopkins07_qlf_z3.6.dat'}

for model in models:      
    LF_interval = np.loadtxt(path + 'data/dPhi_interval_'+model+'_to210.dat')
    print LF_interval
    # L z dPhi_low_90 dPhi_high_90 dPhi_low_99 dPhi_high_99 dPhi_mode dPhi_mean dPhi_median
    z0 = np.where(LF_interval[:,1]==0)
    Phi0 = LF_interval[z0, 6][0]

    L = LF_interval[z0, 0][0]

    fig = plt.figure(figsize=(10,10))
    fig.subplots_adjust(left=0.13, top=0.96, bottom=0.10, right=0.98, wspace=0.03, hspace=0.03)   
    plt_indx = 1
        
    for z in redshift:
        i = list(redshift).index(z)
        # model fit
        ax = fig.add_subplot(1, 1, plt_indx)
        #zz = np.where( LF_interval[:,1] == z_Vmax )
        zz = np.where( LF_interval[:,1] == z )
        print z
        Phi = LF_interval[zz, 6][0]
        Phi_low = LF_interval[zz, 2][0]
        Phi_high = LF_interval[zz, 3][0]
        Phi_mode1 = LF_interval[zz, 9][0]
        Phi_mode2 = LF_interval[zz, 10][0]
        #print Phi, L
        
        L_Hopkins, LF_Hopkins = np.loadtxt(Hopkins_qlf[z], unpack=True, usecols=(0,4))        
        
        LF_Ueda =  np.log10(Ueda14(np.array(L), z, Ueda_params))
        LF_Aird =  np.log10(Ueda14(np.array(L), z, Aird_params))
        LF_Miyaji =  np.log10(Miyaji15(np.array(L), z, Miyaji_params))
        
        LF_XMM =  np.log10(Fotopoulou(np.array(L), z, XMM_params))
        LF_Chandra =  np.log10(Fotopoulou(np.array(L), z, Chandra_params))

        
        LF_Vito =  np.log10(highz(np.array(L), z, Vito_params))
        LF_Georgakakis =  np.log10(highz(np.array(L), z, Georgakakis_params))

        plt.plot(L, LF_Ueda,  color='b', ls = '--',lw=3, label='Ueda+14: LDDE')
        plt.plot(L, LF_Aird,  color='r', ls = '--',lw=3, label='Aird+15: LDDE')
        plt.plot(L, LF_Miyaji,  color='g', ls = '--',lw=3, label='Miyaji+15: LDDE')
        plt.plot(L_Hopkins, np.log10(LF_Hopkins),  color='magenta', ls = '-',lw=2, label='Hopkins+07: LDDE')
        
        plt.plot(L, LF_XMM,  color='b', ls = '-.',lw=3, label='XMM: LDDE')
        plt.plot(L, LF_Chandra,  color='r', ls = '-.',lw=3, label='Chandra: LDDE')

        
        if z>3:
            plt.plot(L, LF_Vito, color='black', ls = '-',lw=2, label='Vito+14: highz')
            plt.plot(L, LF_Georgakakis, color='gray', ls = '-',lw=2, label='Georgakakis+15: highz')
        
        
        plt.fill_between(L, Phi_low, Phi_high, color='gray', alpha=0.5, label='this work')
        plt.plot(L, Phi0, 'k:',lw=2)
        plt.plot(L, Phi, 'k-', lw=4, zorder=0)   

        # find Johannes' points
        F_min = []
        F_max = []
            
        for LB in range(0, len(LogL_B15)-1):
            #print redshift[i], zB_min[i], zB_max[i]
            Buchner_Lzbin = np.where( (Buchner15[:,2]>=zB_min[i]) & (Buchner15[:,3]<=zB_max[i]) & (Buchner15[:,0]>=LogL_B15[LB]) & (Buchner15[:,1]<=LogL_B15[LB+1]) )            
            F_min.append(  np.log10( np.sum(np.power(10, Buchner15[Buchner_Lzbin, 6][:][:]) )) )
            F_max.append(  np.log10( np.sum(np.power(10, Buchner15[Buchner_Lzbin, 7][:][:]) )) )
        #plt.plot( LogL_B[:-1], F_min, 'o', color='yellow', alpha=0.3)
        #plt.plot( LogL_B[:-1], F_max, 'o', color='yellow', alpha=0.3)

        plt.fill_between(LogL_B[:-1], F_min, F_max, color='orange', alpha=0.2, label='Buchner+15')

        # best fit paramters from 2 modes of parameters distribution
        #plt.plot(L, Phi_mode1, 'r--')   
        #plt.plot(L, Phi_mode2, 'r:')   
        
        plt.yticks([-10, -8, -6, -4, -2], size='x-small')
        plt.xticks([42, 43, 44, 45], size='x-small')
        
        
        xmin = 42
        xmax = 46
        ymin = -8
        ymax = -2
        plt.ylim([ymin, ymax])
        plt.xlim([xmin, xmax])
        
        plt.ylabel('$\mathrm{Log[d\Phi/logL_x/(Mpc^{-3})]}$')
        plt.xlabel('$\mathrm{Log[L_x/(erg/sec)]}$')

    
        plt.text(42.25, -6.30, r'$\rm{this\,work}$', size='small')
        plt.text(42.25, -6.50, r'$\rm{Aird+15}$', color='red', size='small')
        plt.text(42.25, -6.70, r'$\rm{Buchner+15}$', color='orange', size='small')
        plt.text(42.25, -6.90, r'$\rm{Miyaji+15}$', color='green', size='small')
        plt.text(42.25, -7.10, r'$\rm{Ueda+14}$', color='blue', size='small')
        plt.text(42.25, -7.30, r'$\rm{Hopkins+2007}$', color='magenta', size='small')
        plt.text(42.25, -7.50, r'$\rm{Vito+2014}$', color='black',size='small')
        plt.text(42.25, -7.70, r'$\rm{Georgakakis+15}$', color='gray', size='small')
    
        ticklab = ax.yaxis.get_ticklabels()[0]
        trans = ticklab.get_transform()
        #ax.yaxis.set_label_coords(-0.125, -2, transform=trans)
        
        ticklab = ax.xaxis.get_ticklabels()[0]
        trans = ticklab.get_transform()
        #ax.xaxis.set_label_coords(46, -0.1, transform=trans)
        
        # Vmax
        #V_mask = np.where( z_Vmax==z )
        #Vmax_zmin = z_min_Vmax[V_mask][0]
        #Vmax_zmax = z_max_Vmax[V_mask][0]
        
        #Vmax_dPhi = dPhi_Vmax[V_mask]
        #Vmax_dPhi_err = dPhi_err_Vmax[V_mask]
        
        #Vmax_L = L_Vmax[V_mask]
        #Vmax_L_minErr = L_Vmax[V_mask] - L_min_Vmax[V_mask]
        #Vmax_L_maxErr = L_max_Vmax[V_mask] - L_Vmax[V_mask]
        
        #Vmax_count = N_Vmax[V_mask]
        #plt.errorbar(Vmax_L, Vmax_dPhi, Vmax_dPhi_err, [Vmax_L_minErr, Vmax_L_maxErr], 'ko', markersize=7)
        
        #for i in range(0, len(Vmax_count) ):
        #    if Vmax_count[i]>0: plt.text(Vmax_L[i], Vmax_dPhi[i]+0.6, int(Vmax_count[i]), size=13)        

        plt.text(xmax - (xmax-xmin)*0.2, ymax - (ymax-ymin)*0.15, r'z='+str(z), size='small')        

        plt_indx = plt_indx + 1

    plt.savefig('XLF_'+model+'_literature_highz.pdf', dpi=200)

    plt.show()