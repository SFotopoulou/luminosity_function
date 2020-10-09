import sys
# Append the module path to the sys.path list
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/models')

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from Source import get_flux
from LFunctions import Ueda14, LADE
import matplotlib.gridspec as gridspec
import itertools
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


Aird_params = Params()

Aird_params.L0 = 44.77
Aird_params.g1 = 0.62
Aird_params.g2 = 3.01
Aird_params.p1 = 6.36
Aird_params.p2 = -0.24
Aird_params.zc = 0.75
Aird_params.d = -0.19
Aird_params.Norm = -4.53


########################################################
# Data
########################################################

Lx = np.linspace(40.0, 47.0)
redshift = [0.104, 1.161, 2.421, 3.376]
zbin = [0.01, 0.2, 1.0, 1.2, 2.0, 3.0, 4.0]
zii = [0.1, 1.13157894737, 2.42105263158, 3.45263157895]

path = '/home/Sotiria/workspace/Luminosity_Function/src/LF_plots/forPaper/'

x_plots = 2
y_plots = 4 
gray = 'gray'#(0.85, 0.85, 0.85)

models = ['LDDE']#, 'LADE', 'ILDE', 'PDE', 'PLE']
lf_idx = itertools.cycle([0,1,4,5])
ratio_idx = itertools.cycle([2,3,6,7])
for model in models:  
    fig = plt.figure(figsize=(10,10))
    fig.subplots_adjust(hspace=0.0)
    
    gs0 = gridspec.GridSpec(2,2)
    gs0.update( left=0.11, right=0.99, top=0.99, bottom=0.10, wspace=0.20, hspace=0.20 )
        
        
    LF_interval = np.loadtxt(path + 'data/dPhi_interval_'+model+'_for_metaLF.dat')
    count1 = 1
    used = []
    for zi in zii :    
        j = zii.index(zi)
        
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[j],height_ratios=[2,1])

        ax = plt.subplot(gs[0])  
        zz = np.where( np.abs(LF_interval[:,1] - zi)<0.001 )
                
        LF_ll = LF_interval[zz, 0][0]
        LF_mode = LF_interval[zz, 6][0]

        LF_low90 = LF_interval[zz, 2][0]
        LF_high90 = LF_interval[zz, 3][0]

        LF_low99 = LF_interval[zz, 4][0]
        LF_high99 = LF_interval[zz, 5][0]
        
        LF_Ueda =  np.log10(Ueda14(np.array(LF_ll), zi, Ueda_params))
        LF_Aird =  np.log10(LADE(np.array(LF_ll), zi, Aird_params))

        plt.fill_between(x=LF_ll, y1 =LF_low99, y2=LF_high99, color=(0.85, 0.85, 0.85))
        plt.fill_between(x=LF_ll, y1 =LF_low90, y2=LF_high90, color='gray')
        
        plt.plot(LF_ll, LF_mode,  color='k', ls = '-',lw=3, label='this work')
        
        plt.plot(LF_ll, LF_Ueda,  color='b', ls = '-',lw=3, label='Ueda+14')
        plt.plot(LF_ll, LF_Aird,  color='r', ls = '--',lw=4, label='Aird+10')
        
        x_vis = False
        y_vis = True
        plt.xticks([42, 43, 44, 45, 46], visible=x_vis, size='x-small')
        plt.yticks([-10, -8, -6, -4, -2], visible=y_vis, size='x-small')
             
        if j == 1:
            plt.legend(loc=3, fontsize='xx-small') 
     
        if j == 0 or j==2:
            plt.ylabel(r'd$\Phi$/dlogLx', fontsize='medium')
            #ax.yaxis.set_label_coords(-0.25, 1.0)
     
       
        i = zii.index(zi)

        if i == 3:
            i = i +2

        if i == 2:
            i = i +2

        if i == 1 :
            i = i+1

        ax.annotate(str(zbin[i])+"$< $"+"z"+"$ < $"+str(zbin[i+1]), (0.55, 0.85) , xycoords='axes fraction', fontstyle='oblique', fontsize='x-small')
            
        plt.ylim([-12, 0.0])
        plt.xlim([41.5, 46.5 ])
        
        ax2 = plt.subplot(gs[1])  

        plt.plot(LF_ll, LF_mode/LF_mode,  color='k', ls = '-',lw=3)
        plt.plot(LF_ll, LF_mode/LF_Ueda,  color='b', ls = '-',lw=3)
        plt.plot(LF_ll, LF_mode/LF_Aird,  color='r', ls = '--',lw=4)
        plt.ylim([ 0.6, 1.4])
        plt.xlim([41.5, 46.5 ])

        x_vis = True
        y_vis = True
        plt.xticks([42, 43, 44, 45, 46], visible=x_vis, size='x-small')
        plt.yticks([0.8, 1.0, 1.2], visible=y_vis, size='x-small')
        if j == 2 or j == 3:
            plt.xlabel(r'dlogLx', fontsize='medium')
            #ax.xaxis.set_label_coords(-0.075, -0.15)
            
    #for ext in ['jpg','pdf','eps','png']:    
    plt.savefig(path + 'plots/Ueda14_Aird10_comparison_ratio_with_'+model+'.pdf')
    
    plt.show()
