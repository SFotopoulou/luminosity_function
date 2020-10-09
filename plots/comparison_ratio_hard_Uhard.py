import sys
# Append the module path to the sys.path list
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/models')

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from Source import get_flux
from LFunctions import Ueda14, LADE
from LFunctions import Fotopoulou as LDDE
import matplotlib.gridspec as gridspec
import itertools
from scipy.interpolate import interp1d
########################################################
covariance_matrix = [[0.017, 0.007, 0.05, 0.009, 0.008, -0.005, -0.0038, 0.0003, -0.04],
                     [0., 0.005, 0.017, 0.010, 0.011, -0.003, -0.0014, 0.0024, -0.019],
                     [0., 0., 0.20, 0.012, 0.0013, -0.011, -0.010, 0.007, -0.09],
                     [0., 0., 0., 0.08, 0.07, -0.029, -0.0010, 0.0015, -0.04],
                     [0., 0., 0., 0., 0.4, -0.05, -0.003, 0.014, -0.03],
                     [0., 0., 0., 0., 0., 0.19, 0.003, -0.0014, 0.015],
                     [0., 0., 0., 0., 0., 0., 0.0034, -0.0020, 0.007],
                     [0., 0., 0., 0., 0., 0., 0., 0.003, -0.007],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0.08]]

#    Scale to unity my MLE result
sigma = np.sqrt(np.diagonal(covariance_matrix))
stds = np.outer(sigma, sigma)
scaled = stds*covariance_matrix
unc_LDDE = [0.14, 0.15, 0.06, 0.34, 0.80, 0.22, 0.17, 0.03, 0.21]
LDDE_matrix = np.outer(unc_LDDE, unc_LDDE) * scaled

covariance_matrix = [[0.01, 0.008, 0.0054, -0.045, -0.013, 0.02, -0.0007, -0.0049],
                     [0., 0.016, 0.0074, -0.0036, -0.008, -0.000020, 0.0011, -0.013],
                     [0., 0., 0.0085, -0.0069, -0.010, 0.0016, 0.0015, -0.010],
                     [0., 0., 0., 0.8, 0.13, -0.20, -0.0038, 0.012],
                     [0., 0., 0., 0., 0.06, -0.05, -0.0051, 0.019],
                     [0., 0., 0., 0., 0., 0.097, -0.0028, 0.006],
                     [0., 0., 0., 0., 0., 0., 0.0020, 0.006],
                     [0., 0., 0., 0., 0., 0., 0., 0.025]]

#    Scale to unity my MLE result
sigma = np.sqrt(np.diagonal(covariance_matrix))
stds = np.outer(sigma, sigma)
scaled = stds*covariance_matrix
unc_LADE = [0.06, 0.02, 0.11, 0.4, 0.27, 0.09, 0.02, 0.07]
LADE_matrix = np.outer(unc_LADE, unc_LADE) * scaled


covariance_matrix = [[  1.01e-02, 3.64e-03, 6.16e-03, -6.48e-03, -1.06e-02, 2.15e-03, 2.8e-03, -5.6e-04, -1.52-02],
                     [ 0.0, 3.78e-03, 5.69e-03, 2.666e-03, 5.74e-03, -6.05e-04, 4.43e-04, -4.12e-04, -6.86e-03],
                     [ 0.0, 0.0, 2.27e-02, 2.306e-03, 3.16e-02, -1.157e-04, 4.01e-03, -1.75e-03,  -8.26e-03],
                     [ 0.0, 0.0, 0.0, 4.82e-02,  1.194e-02, -7.0e-03, -1.025e-03, 9.00e-05,  -2.173e-03],
                     [ 0.0, 0.0, 0.0, 0.0,  1.11e-01, -1.152e-02, -2.47e-03, -3.61e-03,  1.974e-02],
                     [ 0.0, 0.0, 0.0, 0.0, 0.0, 7.764e-03, 4.60e-03, 4.54e-04, -2.216e-03],
                     [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.01e-03, -8.91e-04, -3.895e-03],
                     [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.63e-04, 6.88e-04],
                     [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.07e-07]]

#covariance_matrix = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
##                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]


#    Scale to unity my MLE result
sigma = np.sqrt(np.diagonal(covariance_matrix))
stds = np.outer(sigma, sigma)
scaled = stds*covariance_matrix
unc_Ueda = [0.06, 0.04, 0.09, 0.16, 0.18, 0.07, 0.07, 0.02, 0.07]

Ueda_matrix = np.outer(unc_Ueda, unc_Ueda) * scaled

draws = 200

def credible_interval(distribution, level, bin=100):
    pdf, bins = np.histogram(distribution, bins = bin, normed=True)
    # credible interval
    bins = bins[:-1]
    binwidth = bins[1]-bins[0]
    idxs = pdf.argsort()
    idxs = idxs[::-1]
    credible_interval = idxs[(np.cumsum(pdf[idxs])*binwidth < level/100.).nonzero()]
    idxs = idxs[::-1] # reverse
    low = min( sorted(credible_interval) )
    high = max( sorted(credible_interval) )
    min_val = bins[low]
    max_val = bins[high]
    # mode
    dist_bin = np.array([bins[i]+(bins[1]-bins[0])/2. for i in range(0,len(bins)-1)])
    mode = dist_bin[np.where(pdf==max(pdf))][0]
    
    return min_val, max_val, mode    

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
#m.values["Lp"] = 44.0 #LF_config.Lp
#m.values["p2"] = -1.5 #LF_config.p2
#m.values["p3"] = -6.2 #LF_config.p3
#m.values["zc2"] = 3.0 #LF_config.zc2
#m.values["La2"] = 44.0 #LF_config.La2
#m.values["a2"] = -0.1 #LF_config.a2
ueda_params = [Ueda_params.L0, Ueda_params.g1, Ueda_params.g2, Ueda_params.p1, Ueda_params.beta, Ueda_params.zc1, \
               Ueda_params.La1, Ueda_params.a1, Ueda_params.Norm]
ueda_draws = np.random.multivariate_normal(ueda_params, Ueda_matrix, draws)

Aird_params = Params()

Aird_params.L0 = 44.77
Aird_params.g1 = 0.62
Aird_params.g2 = 3.01
Aird_params.p1 = 6.36
Aird_params.p2 = -0.24
Aird_params.zc = 0.75
Aird_params.d = -0.19
Aird_params.Norm = -4.53
aird_params = [Aird_params.L0, Aird_params.g1, Aird_params.g2, Aird_params.p1, Aird_params.p2, Aird_params.zc, Aird_params.d, Aird_params.Norm]
aird_draws = np.random.multivariate_normal(aird_params, LADE_matrix, draws)

Fotopoulou_params_210 = Params()

Fotopoulou_params_210.L0 = 43.87 + 0.3098
Fotopoulou_params_210.g1 = 0.88
Fotopoulou_params_210.g2 = 2.46
Fotopoulou_params_210.p1 = 5.68
Fotopoulou_params_210.p2 = -2.8
Fotopoulou_params_210.zc = 2.23
Fotopoulou_params_210.La = 44.51 + 0.3098
Fotopoulou_params_210.a = 0.26
Fotopoulou_params_210.Norm = -6.09

Fotopoulou_params_510 = Params()

Fotopoulou_params_510.L0 = 43.87 
Fotopoulou_params_510.g1 = 0.88
Fotopoulou_params_510.g2 = 2.46
Fotopoulou_params_510.p1 = 5.68
Fotopoulou_params_510.p2 = -2.8
Fotopoulou_params_510.zc = 2.23
Fotopoulou_params_510.La = 44.51 
Fotopoulou_params_510.a = 0.26
Fotopoulou_params_510.Norm = -6.09

Fotopoulou_params_cov = Params()

Fotopoulou_params_cov.g1 = 0.88
Fotopoulou_params_cov.g2 = 2.46
Fotopoulou_params_cov.p1 = 5.68
Fotopoulou_params_cov.p2 = -2.8
Fotopoulou_params_cov.zc = 2.23
Fotopoulou_params_cov.a = 0.26
Fotopoulou_params_cov.Norm = -6.09

fotop_params = [Fotopoulou_params_210.L0, Fotopoulou_params_210.g1, Fotopoulou_params_210.g2, Fotopoulou_params_210.p1, Fotopoulou_params_210.p2, Fotopoulou_params_210.zc, \
                Fotopoulou_params_210.La, Fotopoulou_params_210.a, Fotopoulou_params_210.Norm]
fotop_draws = np.random.multivariate_normal(fotop_params, LDDE_matrix, draws)

params = Params()
########################################################
# Data
########################################################

Lx = np.linspace(42.0, 46.0)
#redshift = [0.104, 1.161, 2.421, 3.376]
zbin = [1.0, 1.2]#[0.01, 0.2, 1.0, 1.2, 2.0, 3.0, 4.0]
zii =[1.13157894737] #[0.1, 1.13157894737, 2.42105263158, 3.45263157895]

path = '/home/Sotiria/workspace/Luminosity_Function/src/LF_plots/forPaper/'

gray = 'gray'

models = ['LDDE']

for model in models:  
    fig = plt.figure(figsize=(10,10))
       
    LF_interval = np.loadtxt(path + 'data/dPhi_interval_'+model+'_for_metaLF.dat')

    for zi in zii :    
        i = zii.index(zi)
        
        ax = plt.subplot(111)  
        zz = np.where( np.abs(LF_interval[:,1] - zi)<0.001 )
                
        LF_ll = LF_interval[zz, 0][0]
        LF_mode = LF_interval[zz, 6][0]
        
        LF_low90 = LF_interval[zz, 2][0]
        LF_high90 = LF_interval[zz, 3][0]

        LF_low99 = LF_interval[zz, 4][0]
        LF_high99 = LF_interval[zz, 5][0]
        
        Lx = LF_ll

        LF_Ueda =  np.log10(Ueda14(np.array(Lx), zi, Ueda_params))
        LF_Aird =  np.log10(LADE(np.array(Lx), zi, Aird_params))
        LF_Fotop_210 =  np.log10(LDDE(np.array(Lx), zi, Fotopoulou_params_210))
        LF_Fotop_510 =  np.log10(LDDE(np.array(Lx), zi, Fotopoulou_params_510))

        ax.annotate("z"+"$ = $"+str(zbin[i+1]), (0.85, 0.85) , xycoords='axes fraction', fontstyle='oblique', fontsize='small')
 
        f_params = [(0, 0.31, 0), (0.254, 0.068, 0.075), (0.941, 0.139, 0.225), (0.999, 0.476 , 0.55)]
        f_params = [(0, 0.50, 0), (0.999, 0.50 , 0.55)]
        
        T = Lx*0
        for covering, fraction, shift in f_params:
            C = np.log10(fraction) + np.log10(LDDE(np.array(Lx)+shift, zi, Fotopoulou_params_210))
            T += 10**np.array(C)
            #plt.plot(Lx, C, '--', label='%d%% covered with %d%%' % (fraction * 100, covering * 100))
        #plt.plot(Lx, np.log10(T), '-', label='combination')
        #plt.show()

        z_in = zi
        dPhi_sigma_F = []
        dPhi_mode_F = []
        for L_in in Lx:
            dPhi_dist_F = []               
            for j in range(0, draws):
                params.L0, params.g1, params.g2, params.p1, params.p2, params.zc, params.La, params.a, params.Norm = fotop_draws[j, :]
                dPhi_dist_F.append( LDDE(L_in, zi, params )) 

            dPhi_low_68, dPhi_high_68, dPhi_mode = credible_interval(dPhi_dist_F, 68)
            dPhi_sigma_F.append((dPhi_high_68-dPhi_low_68)/2.0)
            dPhi_mode_F.append( dPhi_mode )
        sigma_F = np.array(dPhi_sigma_F) / np.array(dPhi_mode_F)
        
        dPhi_sigma_A = []
        dPhi_mode_A = []   
        for L_in in Lx:
            dPhi_dist_A = []               
            for j in range(0, draws):
                params.L0, params.g1, params.g2, params.p1, params.p2, params.zc, params.d, params.Norm = aird_draws[j, :]
                dPhi_dist_A.append( LADE(L_in, zi, params )) 

            dPhi_low_68, dPhi_high_68, dPhi_mode = credible_interval(dPhi_dist_A, 68)
            dPhi_sigma_A.append((dPhi_high_68-dPhi_low_68)/2.0)
            dPhi_mode_A.append( dPhi_mode )

        sigma_A = np.array(dPhi_sigma_A) / np.array(dPhi_mode_A)

        aird_unc = np.sqrt( np.power(sigma_A, 2.0) +  np.power(sigma_F, 2.0) )
        aird_ratio = np.log10(10**(LF_Aird) /10**(LF_Fotop_210))

        aird_low = aird_ratio - 3*aird_unc
        aird_high = 3*aird_unc + aird_ratio

            
        dPhi_sigma_U = []
        dPhi_mode_U = []
        for L_in in Lx:
            dPhi_dist_U = []               
            for j in range(0, draws):
                                
                Ueda_params.Lp = 44.0
                Ueda_params.p2 = -1.5
                Ueda_params.p3 = -6.2
                Ueda_params.zc2 = 3.0
                Ueda_params.La2 = 44.0
                Ueda_params.a2 = -0.1
                Ueda_params.L0, Ueda_params.g1, Ueda_params.g2, Ueda_params.p1, Ueda_params.beta, Ueda_params.zc1, \
                Ueda_params.La1, Ueda_params.a1, Ueda_params.Norm = ueda_draws[j, :]
                dPhi_dist_U.append( Ueda14(L_in, zi, Ueda_params )) 

            dPhi_low_68, dPhi_high_68, dPhi_mode = credible_interval(dPhi_dist_U, 68)
            dPhi_sigma_U.append((dPhi_high_68-dPhi_low_68)/2.0)
            dPhi_mode_U.append( dPhi_mode )

        sigma_U = np.array(dPhi_sigma_U / np.array(dPhi_mode_U))

        ueda_unc = np.sqrt( np.power(sigma_U, 2.0) +  np.power(sigma_F, 2.0) )
        ueda_ratio = np.log10(10**(LF_Ueda) / 10**(LF_Fotop_210))

        ueda_low = ueda_ratio - 3*ueda_unc
        ueda_high = 3*ueda_unc + ueda_ratio
        
        
        #plt.plot(Lx, np.log10( T / 10**( LF_Fotop_210)), '-', label='four-population model', color='g',lw=4)
        #plt.plot(Lx, aird_ratio,  color='r', ls = '--',lw=4,label='Aird+10 / this work')
        #plt.plot(Lx, ueda_ratio,  color='b', ls = '--',lw=4,label='Ueda+14 / this work')
        plt.plot(Lx, np.log10( 10**( LF_Fotop_210)/10**( LF_Fotop_210)), '-', color='k',lw=1)

        plt.plot(Lx, np.log10( T  / 10**( LF_Fotop_210)), '-', label='four-population model', color='g',lw=4)
        plt.plot(Lx, aird_ratio,  color='r', ls = '--',lw=4,label='Aird+10 / this work')
        plt.plot(Lx, ueda_ratio,  color='b', ls = '--',lw=4,label='Ueda+14 / this work')

        plt.fill_between(x=Lx, y1 = aird_low, y2=aird_high, color='r', alpha=0.3)
        plt.fill_between(x=Lx, y1 = ueda_low, y2=ueda_high, color='b', alpha=0.3)

        plt.ylim([ -0.9, 1.1])
        plt.xlim([41.9, 46.1 ])

        #plt.xticks([42, 43, 44, 45, 46], size='x-small')
        #plt.yticks([0.0, 0.5, 1.0], size='x-small')

        plt.legend(loc=1)    
    plt.xlabel(r'$\mathrm{\log(\,L_x /erg\cdot s^{-1})}$', fontsize='x-large')
#    plt.ylabel(r'$\mathrm{\log \phi}$', fontsize='x-large')     #\log\phi_{abs} - \log \phi_{unabs}}$', fontsize='x-large')            

    plt.ylabel(r'$\mathrm{\log\phi_{abs} - \log \phi_{unabs}}$', fontsize='x-large')            
    #for ext in ['jpg','pdf','eps','png']:    
    
plt.show()
