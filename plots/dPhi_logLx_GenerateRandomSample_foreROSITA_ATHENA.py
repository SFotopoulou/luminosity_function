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
from LFunctions import PLE,PDE,ILDE,LADE,Fotopoulou
from AGN_LF_config import LF_config
from cosmology import *
from Survey import *
from Source import *
from SetUp_data import Set_up_data
import astroML as ML
from matplotlib.ticker import NullFormatter
from astropy.io import fits as pyfits

class Params: pass
params = Params()

def Phi(model, Lx, z, params_in):    
    """ 
    The luminosity function model 
    """
    if model == 'PLE':
        params.L0 = params_in[0]+0.3098
        params.g1 = params_in[1]
        params.g2 = params_in[2]
        params.p1 = params_in[3]
        params.p2 = params_in[4]
        params.zc = params_in[5]
        params.Norm = params_in[6]
        return PLE(Lx, z, params)
    
    if model == 'PDE':
        params.L0 = params_in[0]+0.3098
        params.g1 = params_in[1]
        params.g2 = params_in[2]
        params.p1 = params_in[3]
        params.p2 = params_in[4]
        params.zc = params_in[5]
        params.Norm = params_in[6]
        return PDE(Lx, z, params)                          

    if model == 'ILDE':
        params.L0 = params_in[0]+0.3098
        params.g1 = params_in[1]
        params.g2 = params_in[2]
        params.p1 = params_in[3]
        params.p2 = params_in[4]
        params.Norm = params_in[5]        
        return ILDE(Lx, z, params)
    
    if model == 'LADE':
        params.L0 = params_in[0]+0.3098
        params.g1 = params_in[1]
        params.g2 = params_in[2]
        params.p1 = params_in[3]
        params.p2 = params_in[4]
        params.zc = params_in[5]      
        params.d = params_in[6]      
        params.Norm = params_in[7]             
        return LADE(Lx, z, params)                          

    if model == 'LDDE':
        params.L0 = params_in[0]+0.3098
        params.g1 = params_in[1]
        params.g2 = params_in[2]
        params.p1 = params_in[3]
        params.p2 = params_in[4]
        params.zc = params_in[5]
        params.La = params_in[6]+0.3098
        params.a = params_in[7]
        params.Norm = params_in[8]       
        return Fotopoulou(Lx, z, params)



def rebin_2darray(a, xbin, ybin):

    """
    input:  2d numpy array
    output: 2d array with shape (x_bin, y_bin) with elements summed of the merged bin
    """
    #print np.shape(a)
    new_xstep = int( len(a[:,0])/xbin )
    #print new_xbin, new_xstep, len(a[:,0]) 
    
    
    new_ystep = int( len(a[0,:])/ybin )
    #print ybin, new_ystep, len(a[0,:]) 
    
    b = np.zeros((xbin, ybin))
    k = 0
    
    for i in range(0, len(a[:,0]), new_xstep ):
        l = 0
        for j in range(0, len(a[0,:]), new_ystep ):
       #     print i, j, k, l
      #      print a[i:i+new_xstep,j:j+new_ystep]
     #       print
            b[k, l] = a[i:i+new_xstep,j:j+new_ystep].sum()
            l = l + 1
        k = k + 1
    #print np.shape(b)
    return b

def credible_interval(distribution, level, bin=100000, plot=False):
    pdf, bins = np.histogram(distribution, bins = bin, normed=True)
    if plot==True:
        plt.clf()
        plt.plot(bins[:-1], pdf)
        plt.show()
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

path = '/home/Sotiria/workspace/Luminosity_Function/src/LF_plots/forPaper/'
param_path = '/home/Sotiria/workspace/Luminosity_Function/src/MultiNest/new/'

import time
zbin = 1001
Lbin = 1001
zlim = 4.0
data_zbin = 51
data_Lbin = 51
# plot LF grid
ZZ_plot = np.linspace(LF_config.zmin, zlim, data_zbin)
LL_plot = np.linspace(LF_config.Lmin, LF_config.Lmax, data_Lbin)
zbin_width_plot = (ZZ_plot[1] - ZZ_plot[0])/2.0
Lbin_width_plot = (LL_plot[1] - LL_plot[0])/2.0

print ZZ_plot
print LL_plot
# LF grid
ZZ = np.linspace(LF_config.zmin, zlim, zbin)
LL = np.linspace(LF_config.Lmin, LF_config.Lmax, Lbin)
zbin_width = (ZZ[1] - ZZ[0])/2.0
Lbin_width = (LL[1] - LL[0])/2.0

def evol(z ,L, p1, p2, zc, La, a):
    L = np.power(10.0, L)
    La10 = np.power(10.0, La)
    zc_case1 = zc
    zc_case2 = zc * np.power( L/La10, a )
    zc = np.where(L >= La, zc_case1, zc_case2)

    norm = np.power( (1.0 + zc), p1) + np.power( (1.0 + zc), p2)
    ez = norm/(np.power( (1. + z) / (1. + zc), -p1) + np.power( (1. + z) / (1. + zc), -p2) )
    return ez

def zcrit(L, zc, La, a):
    L = np.power(10.0, L)
    La10 = np.power(10.0, La)
    zc_case1 = zc
    zc_case2 = zc * np.power( L/La10, a )
    zc = np.where(L >= La, zc_case1, zc_case2)
    return zc

#lum_data = pyfits.open(LF_config.data_out_name)[1].data

# model
models = ['LDDE']#, 'LADE', 'ILDE', 'PDE', 'PLE']
c_low_lim, c_up_lim=0, 100
for model in models:
    
    fig = plt.figure(figsize=(10,10))

#    fig.add_subplot(2,3,models.index(model)+1)

    nullfmt   = NullFormatter()         # no labels
    # definitions for the axes
    left, width = 0.11, 0.60
    bottom, height = 0.11, 0.60
    bottom_h = left_h = left+width
    
    rect_2dhist = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.18]
    rect_histy = [left_h, bottom, 0.18, height]
    
    # start with a rectangular Figure
    #plt.figure(1, figsize=(8,8))
    
    ax2Dhist = plt.axes(rect_2dhist)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    folder_name = '_'.join(LF_config.fields) + '_' +'_ztype_'+LF_config.ztype+"_"+ model + '_pl' + str(LF_config.pl) + '_zunc_' + str(LF_config.z_unc)
    parameters = np.loadtxt(param_path + folder_name + '/1-.txt')
    params_in = parameters[:, 2:]
    n_params = len(params_in[0,:])
    #print len(params_in)
    
    LF_modes = np.loadtxt(param_path + folder_name + '/1-summary.txt')
    n_modes = len(LF_modes)
    mean_params = LF_modes[:, 0:n_params]
    mean_err_params = LF_modes[:, n_params:2*n_params]
    MLE_params = LF_modes[0, 2*n_params:3*n_params]
    MAP_params = LF_modes[:, 3*n_params:4*n_params]
    #print MLE_params
       
    modes = []
    for i in range(1, n_modes+1):
        modes.append('mode'+str(i))
    
#    N = np.loadtxt( path + 'data/Expected_N_'+model+'.dat')
#    plt.imshow(N)
#    plt.show()
    vol = np.loadtxt( path + 'data/Volume_for_eROSITA.dat')
    vol_future = np.loadtxt( path + 'data/Volume_for_futureSurvey.dat')
    vol_perfect = np.loadtxt( path + 'data/Volume_for_perfectSurvey.dat')

    P = []
    P_min = []
    P_max = []

    P_future = []
    P_min_future = []
    P_max_future = []
    
    P_perfect = []
    P_min_perfect = []
    P_max_perfect = []

    
    evolution = []
    zc = []
    
    flux_perfect = 2.5e-16
    
    perfect_luminosity = []
    for z in ZZ:
        perfect_luminosity.append( get_luminosity(flux_perfect, flux_perfect, z)[0] )
 
 
    e_flux, e_curve = np.loadtxt('/home/Sotiria/Documents/Luminosity_Function/data/eROSITA/eROSITA_acurve_hard_band.txt', unpack=True)
    flux_future = np.power(10., e_flux-2)
#
    future_luminosity = []
    for z in ZZ:
        future_luminosity.append( get_luminosity(np.min(flux_future), np.min(flux_future), z)[0] )
    
    L_break = []
    print model, len(params_in)
    for j in range(0, len(LL)-1):
        #print LL[j]
        L_star = []
        temp_p = []
        temp_p_min = []
        temp_p_max = []
        
        temp_p_future = []
        temp_p_min_future = []
        temp_p_max_future = []
        
        temp_p_perfect = []
        temp_p_min_perfect = []
        temp_p_max_perfect = []
        #t1 = time.time()
        ev = []
        

#        for i in range(0, len(ZZ)-1):     
#            ev = 

#            #print model, LL[j]+Lbin_width, ZZ[i]+zbin_width, params_in
#            n_distr = Phi(model, LL[j]+Lbin_width, ZZ[i]+zbin_width, params_in)
#            
#            # eROSITA
#            if vol[j,i]>0:
#                n_min, n_max, n_mode = credible_interval(n_distr, 99.) 
#            else:
#                n_min, n_max, n_mode = 0, 0, 0
#
#            temp_p.append(n_mode)
#            temp_p_min.append(n_min)
#            temp_p_max.append(n_max)
#
#            # future
#            if vol_future[j,i]>0:
#                n_min_future, n_max_future, n_mode_future = credible_interval(n_distr, 99.) 
#            else:
#                n_min_future, n_max_future, n_mode_future = 0, 0, 0
#
#            temp_p_future.append(n_mode_future)
#            temp_p_min_future.append(n_min_future)
#            temp_p_max_future.append(n_max_future)
#
#            # perfect
#            if vol_perfect[j,i]>0:
#                n_min_perfect, n_max_perfect, n_mode_perfect = credible_interval(n_distr, 99.) 
#            else:
#                n_min_perfect, n_max_perfect, n_mode_perfect = 0, 0, 0
#
#            temp_p_perfect.append(n_mode_perfect)
#            temp_p_min_perfect.append(n_min_perfect)
#            temp_p_max_perfect.append(n_max_perfect)
#
#
#        P.append(temp_p)
#        P_min.append(temp_p_min)
#        P_max.append(temp_p_max)
#        
#        P_future.append(temp_p_future)
#        P_min_future.append(temp_p_min_future)
#        P_max_future.append(temp_p_max_future)
#        
#        P_perfect.append(temp_p_perfect)
#        P_min_perfect.append(temp_p_min_perfect)
#        P_max_perfect.append(temp_p_max_perfect)
        params.L0 = params_in[0]+0.3098
        params.g1 = params_in[1]
        params.g2 = params_in[2]
        params.p1 = params_in[3]
        params.p2 = params_in[4]
        params.zc = params_in[5]
        params.La = params_in[6]+0.3098
        params.a = params_in[7]
        params.Norm = params_in[8]  
        #evolution.append(ev)   
        
        for z in ZZ:
            evolution = evol(z ,LL[j]+Lbin_width, MLE_params[3], MLE_params[4], MLE_params[5], MLE_params[6]+0.3098, MLE_params[7])
            L_star.append( MLE_params[0]+0.3098/np.power(evolution,MLE_params[1]) )
         
        P.append(list(Phi(model, LL[j]+Lbin_width, np.array(ZZ[:-1])+zbin_width, MLE_params))) 
        
        #    L break    
        L_break.append( L_star  )

    #print np.shape(evolution), np.min(evolution), np.max(evolution)
    N = vol * P
    N_future = vol_future * P
    N_perfect = vol_perfect * P
    #===========================================================================
#    N_min = vol * P_min
#    N_max = vol * P_max
#    N_min_future = vol_future * P_min_future
#    N_max_future = vol_future * P_max_future
#    N_min_perfect = vol_perfect * P_min_perfect
#    N_max_perfect = vol_perfect * P_max_perfect
    #===========================================================================
    
#    try:
#        np.savetxt( path + 'data/Expected_eROSITA_N_'+model+'.dat', N)
#        #=======================================================================
#        np.savetxt( path + 'data/Expected_eROSITA_N_'+model+'.dat', N_min)
#        np.savetxt( path + 'data/Expected_eROSITA_N_'+model+'.dat', N_max)
#        #=======================================================================
#        np.savetxt( path + 'data/Expected_future_N_'+model+'.dat', N)
#        #=======================================================================
#        np.savetxt( path + 'data/Expected_future_N_'+model+'.dat', N_min)
#        np.savetxt( path + 'data/Expected_future_N_'+model+'.dat', N_max)
#        #=======================================================================
#        np.savetxt( path + 'data/Expected_perfect_N_'+model+'.dat', N)
#        #=======================================================================
#        np.savetxt( path + 'data/Expected_perfect_N_'+model+'.dat', N_min)
#        np.savetxt( path + 'data/Expected_perfect_N_'+model+'.dat', N_max)
#        #=======================================================================
#    except:
#        pass
    #print np.shape(N)
    N = rebin_2darray(N, data_Lbin-1, data_zbin-1) 
    N_future = rebin_2darray(N_future, data_Lbin-1, data_zbin-1) 
    N_perfect = rebin_2darray(N_perfect, data_Lbin-1, data_zbin-1) 

    #print np.shape(N)
    im = ax2Dhist.imshow(N, origin='low', extent=[0.01, zlim, 41,46], cmap='binary', aspect='auto')
    ax2Dhist.plot(ZZ, future_luminosity, 'r--',lw=3)
    ax2Dhist.plot(ZZ, perfect_luminosity, 'b:',lw=3)
    #im = ax2Dhist.imshow(N_future, origin='low', extent=[0.01, zlim, 41,46], cmap='Reds', aspect='auto', alpha=0.4)
    #im = ax2Dhist.imshow(N_perfect, origin='low', extent=[0.01, zlim, 41,46], cmap='Blues', aspect='auto', alpha=0.4)
    ax2Dhist.set_xlim( (0.0, zlim) )
    ax2Dhist.set_ylim( (41, 46) )
    im.set_clim(c_low_lim, c_up_lim)
    
  
    ax2Dhist.set_ylabel('$\mathrm{log(\,L_x /erg\cdot s^{-1})}$', fontsize='large')
    ax2Dhist.set_xlabel('$\mathrm{z}$', fontsize='large')
    
    ax2Dhist.text(1.75, 42.0, '$\mathrm{eROSITA\,all\,sky}$', fontsize=20, color='black')
    ax2Dhist.text(1.75, 41.7, '$\mathrm{100\,deg^2\,deep\,field}$', fontsize=20, color='red')
    ax2Dhist.text(1.75, 41.4, '$\mathrm{100\,deg^2\,perfect\,efficiency}$', fontsize=20, color='blue')
    ax2Dhist.text(2.5, 43.6, '$\mathrm{2.5\,10^{-16}erg/s/cm^2}$', fontsize=20, color='blue',rotation=13)
    
    
    number = int(np.sum(N))
    print int(np.sum(N)),int(np.sum(N_future)),int(np.sum(N_perfect))
    print '1<z<2:',int(np.sum(N[:30,13:25])),int(np.sum(N_future[:30,13:25])),int(np.sum(N_perfect[:30,13:25]))
    #===========================================================================
#    up_lim = int(np.sum(N_max)) - int(np.sum(N))
#    low_lim = int(np.sum(N)) - int(np.sum(N_min))
#    print 'eROSITA', int(np.sum(N_min)), int(np.sum(N)), int(np.sum(N_max))
#    print 'future',  int(np.sum(N_min_future)), int(np.sum(N_future)), int(np.sum(N_max_future))
#    print 'perfect', int(np.sum(N_min_perfect)), int(np.sum(N_perfect)), int(np.sum(N_max_perfect))
    #ax2Dhist.text(4.15, 46.5, '$\mathrm{N='+str(number)+'^{+'+str(up_lim)+'}_{-'+str(low_lim)+'} }$')

    #===========================================================================
    #ax2Dhist.text(4.15, 46.5, '$\mathrm{N='+str(number)+'}$')
    
    #zz_test = np.sum(N, axis=0)
    #print np.shape(zz_test), len(ZZ)
    #print model, np.sum(N)

    axHistx.plot(ZZ_plot[:-1]+zbin_width_plot, np.sum(N, axis=0), color='k',lw=3)
    axHistx.plot(ZZ_plot[:-1]+zbin_width_plot, np.sum(N_future, axis=0), 'r--',lw=3)
    axHistx.plot(ZZ_plot[:-1]+zbin_width_plot, np.sum(N_perfect, axis=0), 'b:',lw=3)
    axHistx.set_xlim( ax2Dhist.get_xlim() )
    #axHistx.set_ylim((0,1.8))
    #axHistx.set_ylim((0,120))
    axHistx.set_yticks([])
    
    axHisty.plot(np.sum(N, axis=1), LL_plot[:-1]+Lbin_width_plot, color='k',lw=3)
    axHisty.plot(np.sum(N_future, axis=1), LL_plot[:-1]+Lbin_width_plot, 'r--',lw=3)
    axHisty.plot(np.sum(N_perfect, axis=1), LL_plot[:-1]+Lbin_width_plot, 'b:',lw=3)
    axHisty.set_ylim( ax2Dhist.get_ylim() )
    #axHisty.set_xlim((0,0.8))
    #axHisty.set_xlim((0,150))

    axHisty.set_xticks([])
    tick = [0, 50, 100]
    cbar = plt.colorbar(im, ticks=tick)
    cbar.ax.tick_params( labelsize='x-small')
    cbar.set_label('$\mathrm{N}$', fontsize=26)
    plt.savefig(path + 'plots/expectation_N_'+str(data_zbin-1)+'x'+str(data_Lbin-1)+'_'+model+'.pdf', dpi=300)
    #print model, (time.time()-t1)/60, "min"
    plt.show()
    #plt.close()