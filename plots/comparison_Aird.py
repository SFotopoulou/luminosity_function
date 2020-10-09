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

#    Data from James
# 2<z<2.5, computed at 2.25, 2.5<z<3.5, computed at 3
# X-ray sample
L_Aird = itertools.cycle([[43.5625, 43.9375, 44.3125, 44.6875, 45.0625], [43.1875, 43.5625, 43.9375, 44.3125, 44.6875, 45.0625]] )
L_min_Aird = itertools.cycle([[43.3750,43.7500,44.1250,44.5000,44.8750],[43.0000,43.3750,43.7500,44.1250,44.5000, 44.8750]])
L_max_Aird = itertools.cycle([[43.7500,44.1250,44.5000,44.8750,45.2500],[43.3750,43.7500,44.1250,44.5000, 44.8750,45.2500]])

dPhi_Aird = itertools.cycle([[2.65939e-05, 1.22097e-05, 1.23578e-05, 5.67365e-06, 6.95238e-07], [3.95531e-05, 1.26670e-05, 1.34310e-05, 6.48547e-06, 5.77475e-06, 7.91706e-07]])    
dPhi_err_Aird_low = itertools.cycle([[9.07020e-06, 3.97898e-06, 3.33015e-06, 2.05293e-06, 5.67818e-07], [2.72160e-05, 6.71777e-06, 4.01875e-06, 1.94785e-06, 1.54134e-06, 4.77435e-07]])
dPhi_err_Aird_high = itertools.cycle([[1.30624e-05, 5.63148e-06, 4.42298e-06, 3.02844e-06, 1.61451e-06], [6.17042e-05, 1.21852e-05, 5.51823e-06, 2.67804e-06, 2.04131e-06, 9.55148e-07]])
# Color preselected sample
L_Aird_cl = itertools.cycle([[42.8125, 43.1875, 43.5625, 43.9375, 44.3125, 44.6875], [42.8125, 43.1875, 43.5625, 43.9375, 44.3125, 44.6875, 45.0625]], )
L_min_Aird_cl = itertools.cycle([[42.6250,43.00,43.3750,43.7500,44.1250,44.5000],[42.6250,43.0000,43.3750,43.7500,44.1250,44.5000,44.8750]])
L_max_Aird_cl = itertools.cycle([[43.00,43.3750,43.7500,44.1250,44.5000,44.8750],[43.0000,43.3750,43.7500,44.1250,44.5000,44.8750,45.2500]])

dPhi_Aird_cl = itertools.cycle([[ 6.41569e-05, 3.63649e-05, 1.79293e-05, 1.28719e-05, 6.39078e-06, 5.69329e-06], [0.000173090,6.31526e-05,2.41756e-05,1.80302e-05,7.27135e-06,5.57555e-06,6.12555e-07]])    
dPhi_err_Aird_low_cl = itertools.cycle([[4.06788e-05, 1.25321e-05, 5.33435e-06, 3.89764e-06, 2.38806e-06, 1.65616e-06], [7.81118e-05,1.94726e-05,5.17967e-06, 3.24222e-06,1.85414e-06,1.33784e-06,3.16868e-07]])
dPhi_err_Aird_high_cl = itertools.cycle([[8.50483e-05, 1.81222e-05, 7.31074e-06, 5.37364e-06, 3.57183e-06, 2.25292e-06], [0.000128217,2.70153e-05,6.47448e-06,3.90504e-06,2.42338e-06,1.71984e-06,5.65123e-07]])


# LF grid
redshift = np.array([2.25, 3.0])

# model
models = ['LDDE', 'LADE', 'ILDE', 'PDE', 'PLE']

for model in models:      
    LF_interval = np.loadtxt(path + 'data/dPhi_interval_'+model+'_for_Aird.dat')
    
    # L z dPhi_low_90 dPhi_high_90 dPhi_low_99 dPhi_high_99 dPhi_mode dPhi_mean dPhi_median

    fig = plt.figure(figsize=(10,5))
    fig.subplots_adjust(left=0.10, top=0.90, bottom=0.19, right=0.98, wspace=0.1, hspace=0.03)   
    
    for z in redshift:
        # model fit
        plt_indx = np.where(redshift==z)[0][0]+1
        print plt_indx
        ax = fig.add_subplot( 1, 2, plt_indx)
        zz = np.where( LF_interval[:,1] == z )

        L = LF_interval[zz, 0][0]
        Phi = LF_interval[zz, 6][0]
        Phi_low = LF_interval[zz, 2][0]
        Phi_high = LF_interval[zz, 3][0]
        Phi_mode1 = LF_interval[zz, 9][0]
        Phi_mode2 = LF_interval[zz, 10][0]
        
        plt.fill_between(L, Phi_low, Phi_high, color='gray', alpha=0.5)
        plt.plot(L, Phi, 'k-')   
        # best fit paramters from 2 modes of parameters distribution
        plt.plot(L, Phi_mode1, 'r--')   
        plt.plot(L, Phi_mode2, 'r:')   
        
        plt.yticks([-10, -8, -6, -4, -2], size='x-small', visible=False)
        plt.xticks([42, 43, 44, 45, 46], size='x-small', visible=True)
        if plt_indx ==1 : plt.yticks(visible=True)
        #if plt_indx in set([7, 8, 9]): plt.xticks(visible=True)
        
        xmin = 42
        xmax = 46
        ymin = -7
        ymax = -3
        plt.ylim([ymin, ymax])
        plt.xlim([xmin, xmax])
        
        if plt_indx == 1 : plt.ylabel('$\mathrm{Log[d\Phi/logL_x/(Mpc^{-3})]}$')
        if plt_indx == 2 : plt.xlabel('$\mathrm{Log[L_x/(erg/sec)]}$')

        # Vmax
        Aird_dPhi = np.array(dPhi_Aird.next())
        Aird_dPhi_err_low = 0.434*np.asarray(dPhi_err_Aird_low.next()) / Aird_dPhi
        Aird_dPhi_err_high = 0.434*np.asarray(dPhi_err_Aird_high.next()) / Aird_dPhi
        Aird_dPhi = np.log10( Aird_dPhi )
        
        Aird_L = np.asarray(L_Aird.next())
        Aird_L_minErr = np.asarray(Aird_L) - np.asarray(L_min_Aird.next())
        Aird_L_maxErr = np.asarray(L_max_Aird.next()) - np.asarray(Aird_L)
        
        plt.errorbar(Aird_L, Aird_dPhi, [Aird_dPhi_err_high, Aird_dPhi_err_low], [Aird_L_minErr, Aird_L_maxErr], 'o', ecolor='k', color='white', markersize=7)
        
        
        Aird_dPhi_cl = np.array(dPhi_Aird_cl.next())
        Aird_dPhi_err_low_cl = 0.434*np.asarray(dPhi_err_Aird_low_cl.next()) / Aird_dPhi_cl
        Aird_dPhi_err_high_cl = 0.434*np.asarray(dPhi_err_Aird_high_cl.next()) / Aird_dPhi_cl
        Aird_dPhi_cl = np.log10( Aird_dPhi_cl )
        
        Aird_L_cl = np.asarray(L_Aird_cl.next())
        Aird_L_minErr_cl = np.asarray(Aird_L_cl) - np.asarray(L_min_Aird_cl.next())
        Aird_L_maxErr_cl = np.asarray(L_max_Aird_cl.next()) - np.asarray(Aird_L_cl)

        plt.errorbar(Aird_L_cl, Aird_dPhi_cl, [Aird_dPhi_err_high_cl, Aird_dPhi_err_low_cl], [Aird_L_minErr_cl, Aird_L_maxErr_cl], 'r^', markersize=7)
        
        
        plt.text(xmin + (xmax-xmin)*0.75, ymin + (ymax-ymin)*0.85, 'z='+str(z), size='x-small')        
        ax.xaxis.set_label_coords(-0.075, -0.15)
        plt.suptitle(model)

        plt.savefig('plots/dPhi_comparison_Aird2010_'+model+'.pdf', dpi=200)


    plt.show()