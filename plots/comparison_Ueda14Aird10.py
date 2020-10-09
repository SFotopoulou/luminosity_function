import sys
# Append the module path to the sys.path list
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/models')

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from Source import get_flux
from LFunctions import Ueda14, LADE
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
zbin = [1.0, 1.2]#[0.01, 0.2, 1.0, 1.2, 2.0, 3.0, 4.0]
zii = [1.13157894737]#[0.1, 1.13157894737, 2.42105263158, 3.45263157895]

path = '/home/Sotiria/workspace/Luminosity_Function/src/LF_plots/forPaper/'

x_plots = 1
y_plots = 1 
gray = 'gray'#(0.85, 0.85, 0.85)

models = ['LDDE']#, 'LADE', 'ILDE', 'PDE', 'PLE']

for model in models:  
    fig = plt.figure(figsize=(10,10))
    fig.subplots_adjust(left=0.16, right=0.97, wspace=0.05, hspace=0.05)

    LF_interval = np.loadtxt(path + 'data/dPhi_interval_'+model+'_for_metaLF.dat')
    
    for zi in zii :
        ax = fig.add_subplot(y_plots, x_plots, zii.index(zi)+1)
    
        zz = np.where( np.abs(LF_interval[:,1] - zi)<0.001 )
                
        LF_ll = LF_interval[zz, 0][0]
        LF_mode = LF_interval[zz, 6][0]

        LF_low90 = LF_interval[zz, 2][0]
        LF_high90 = LF_interval[zz, 3][0]

        LF_low99 = LF_interval[zz, 4][0]
        LF_high99 = LF_interval[zz, 5][0]

        #fig = plt.figure(figsize=(10,10))

        f_params = [(0, 0.31, 0), (0.254, 0.068, 0.075), (0.941, 0.139, 0.225), (0.999, 0.476 ,0.55)]
        plt.plot(LF_ll, LF_mode, '-', label='ultra-hard', color='k')
        
        T = LF_ll*0
        for covering, fraction, shift in f_params:
            C = np.log10(fraction) + np.log10(Ueda14(np.array(LF_ll)+shift, zi, Ueda_params))
            T += 10**C
            plt.plot(LF_ll, C, '--', label='%d%% covered with %d%%' % (fraction * 100, covering * 100))
        
        plt.plot(LF_ll, np.log10(T), '-', label='combined, hard', color='r')
        #plt.ylabel('log[ d$\phi$/d$\log L_X$ ]')
        #plt.legend(loc='best', prop=dict(size=8))
        #plt.xlim(42, 46)
        #plt.ylim(-8, -3)
        #plt.plot(LF_ll, T / 10**LF_mode, '-', label='hard / ultra-hard', color='r')
        #plt.ylabel('ratio')
        #plt.legend(loc='best', prop=dict(size=8))
        #plt.xlabel('$\log L_X$')
        #plt.xlim(42, 46)

        #plt.show()

        LF_Ueda =  np.log10(Ueda14(np.array(Lx), zi, Ueda_params))
        LF_Aird =  np.log10(LADE(np.array(Lx), zi, Aird_params))

        #plt.fill_between(x=LF_ll, y1 =LF_low99, y2=LF_high99, color=(0.85, 0.85, 0.85))
        #plt.fill_between(x=LF_ll, y1 =LF_low90, y2=LF_high90, color='gray')
        
        #plt.plot(LF_ll, LF_mode,  color='k', ls = '-',lw=4, label='this work')
        
        #plt.plot(Lx, LF_Ueda,  color='b', ls = '-',lw=4, label='Ueda+14')
        #plt.plot(Lx, LF_Aird,  color='r', ls = '--',lw=4, label='Aird+10')
              
        plt.xticks([42, 43, 44, 45, 46], visible=False)
        plt.yticks([-10, -8, -6, -4, -2], visible=False)
                
        i = zii.index(zi)

        if i == 0 :
            plt.yticks(visible=True)
            plt.xticks(visible=False)

        if i == 3:
            i = i +2
            plt.xticks(visible=True)
            plt.xlabel(r'$\mathrm{log(\,L_x /erg\cdot s^{-1})}$', fontsize='x-large')
            ax.xaxis.set_label_coords(-0.075, -0.15)
            
        if i == 2:
            i = i +2
            plt.yticks(visible=True)
            plt.xticks(visible=True)
            plt.ylabel(r'$\mathrm{Log[d\Phi/logL_x/(Mpc^{-3})]}$', fontsize='x-large')
            ax.yaxis.set_label_coords(-0.25, 1.0)
            plt.legend(loc=3) 

        if i == 1 :
            i = i+1

            
        ax.annotate(str(zbin[i])+"$< $"+"z"+"$ < $"+str(zbin[i+1]), (0.375, 0.85) , xycoords='axes fraction', fontstyle='oblique', fontsize='medium', )
            
        plt.ylim([-12, 0.0])
        plt.xlim([41.5, 46.5 ])
    
    #for ext in ['jpg','pdf','eps','png']:    
    #plt.savefig(path + 'plots/Ueda14_Aird10_comparison_with_'+model+'.pdf')
    
    plt.show()
