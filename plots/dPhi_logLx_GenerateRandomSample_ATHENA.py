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

    if model == 'Ueda14':
        params.L0  =  params_in[0]+0.3098
        params.g1  =  params_in[1]
        params.g2  =  params_in[2]
        params.p1  =  params_in[3]
        params.beta = params_in[4]
        params.Lp = params_in[5]+0.3098
        params.p2  =  params_in[6]
        params.p3  =  params_in[7]
        params.zc1  = params_in[8]
        params.zc2  = params_in[9]
        params.La1  = params_in[10]+0.3098
        params.La2  = params_in[11]+0.3098
        params.a1  =  params_in[12]
        params.a2  =  params_in[13]
        params.Norm  = params_in[14]
              

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

#print ZZ_plot
#print LL_plot
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

from matplotlib.colors import colorConverter
import matplotlib as mpl
# generate the colors for your colormap
color1a = colorConverter.to_rgba('white')
color2a = colorConverter.to_rgba('red')

cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',[color1a,color2a],1024)
cmap1._init() # create the _lut array, with rgba values
# create your alpha array and fill the colormap with them.
# here it is progressive, but you can create whathever you want
alphas = np.linspace(0, 1, cmap1.N+3)
cmap1._lut[:,-1] = alphas


color1b = colorConverter.to_rgba('white')
color2b = colorConverter.to_rgba('black')
# make the colormaps
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1b,color2b],1024)
cmap2._init() # create the _lut array, with rgba values
# create your alpha array and fill the colormap with them.
# here it is progressive, but you can create whathever you want
alphas = np.linspace(0, 1, cmap2.N+3)
cmap2._lut[:,-1] = alphas

# model
models = ['LDDE']#, 'LADE', 'ILDE', 'PDE', 'PLE']

c_low_lim1, c_up_lim1=0, 100
c_low_lim2, c_up_lim2=0, 100

for model in models:
    
    fig = plt.figure(figsize=(10,10))

#    fig.add_subplot(2,3,models.index(model)+1)

    nullfmt   = NullFormatter() # no labels
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
    vol_athena = np.loadtxt( path + 'data/Volume_for_ATHENA.dat')
    #vol_perfect = np.loadtxt( path + 'data/Volume_for_perfectSurvey.dat')

    P = []
    P_min = []
    P_max = []

    P_athena = []
    P_min_athena = []
    P_max_athena = []
    
    evolution = []
    zc = []
    
 
    e_flux, e_curve = np.loadtxt('/home/Sotiria/Documents/Luminosity_Function/data/ATHENA/area_curves/total_ATHENA_curve.txt', unpack=True)
    flux_athena = e_flux[e_curve>0]
#
    athena_luminosity = []
    for z in ZZ:
        
        
        athena_luminosity.append( get_luminosity(np.min(flux_athena), np.min(flux_athena), z)[0] )
    
    L_break = []
    
    print model#, len(params_in)
    
    for j in range(0, len(LL)-1):
        #print LL[j]
        L_star = []
        temp_p = []
        temp_p_min = []
        temp_p_max = []
        
        temp_p_athena = []
        temp_p_min_athena = []
        temp_p_max_athena = []

        #t1 = time.time()
        ev = []
        

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
            L_star.append( (MLE_params[0]+0.3098)/np.power(evolution,MLE_params[1]) )
         
        P.append(list(Phi(model, LL[j]+Lbin_width, np.array(ZZ[:-1])+zbin_width, MLE_params))) 
        
        #    L break    
        L_break.append( L_star  )

    N = vol * P
    N_athena = vol_athena * P * 0.00030461742 

    N = rebin_2darray(N, data_Lbin-1, data_zbin-1) 
    N_athena = rebin_2darray(N_athena, data_Lbin-1, data_zbin-1) 
    N_athena_max = int( np.max(N_athena) )
    N_max = int( np.max(N) )
    
    print np.shape(N), np.shape(N_athena)
    
    ax2Dhist.set_xlim( (0.0, zlim) )
    ax2Dhist.set_ylim( (41, 46) )
    X, Y = np.meshgrid(ZZ_plot[:-1]+zbin_width_plot, LL_plot[:-1]+Lbin_width_plot)
    print np.shape(X), np.shape(Y), np.shape(N)
    
    cs1 = ax2Dhist.contour(X, Y, N, 6, colors='k', lw=3)
    #ax2Dhist.clabel(cs1,  inline=1,  fmt='%d',  fontsize=12)
    
    cs2 = ax2Dhist.contour(ZZ_plot[:-1]+zbin_width_plot, LL_plot[:-1]+Lbin_width_plot, N_athena, 6, colors='r', lw=3)
    #ax2Dhist.clabel(cs2,  inline=1,  fmt='%d',  fontsize=12)
    
#    im1 = ax2Dhist.imshow(N_athena, origin='low', extent=[0.01, zlim, 41,46], cmap=cmap1, aspect='auto')
#    im1.set_clim(c_low_lim1, c_up_lim1)
    
#    im2 = ax2Dhist.imshow(N, origin='low', extent=[0.01, zlim, 41, 46], cmap=cmap2, aspect='auto')
#    im2.set_clim(c_low_lim2, c_up_lim2)
 
      
    ax2Dhist.set_ylabel('$\mathrm{log(\,L_x /erg\cdot s^{-1})}$', fontsize='large')
    ax2Dhist.set_xlabel('$\mathrm{z}$', fontsize='large')
    ax2Dhist.set_xticks([0, 1 ,2 ,3 ,4])
    ax2Dhist.text(2.2, 42.0, '$\mathrm{eROSITA\,all\,sky}$', fontsize=20, color='black')
    ax2Dhist.text(2.2, 41.7, '$\mathrm{ATHENA\,deep\,field}$', fontsize=20, color='red')



    number = int(np.sum(N))
    print int(np.sum(N)), int(np.sum(N_athena))
    print '1<z<2:', int(np.sum(N[:30,13:25])), int(np.sum(N_athena[:30,13:25]))

    axHistx.plot(ZZ_plot[:-1]+zbin_width_plot, np.sum(N, axis=0), color='k',lw=3)
    axHistx.plot(ZZ_plot[:-1]+zbin_width_plot, np.sum(N_athena, axis=0), 'r--',lw=3)
    axHistx.set_xlim( ax2Dhist.get_xlim() )
    axHistx.set_yticks([])

    axHisty.plot(np.sum(N, axis=1), LL_plot[:-1]+Lbin_width_plot, color='k',lw=3)
    axHisty.plot(np.sum(N_athena, axis=1), LL_plot[:-1]+Lbin_width_plot, 'r--',lw=3)
    axHisty.set_ylim( ax2Dhist.get_ylim() )
    axHisty.set_xticks([])
    
    #tick1 = [0, c_up_lim1/2.0, c_up_lim1]
    #cbar1 = plt.colorbar(im1, ticks=tick1)
    #cbar1.ax.tick_params( labelsize='x-small')
    #cbar.set_label('$\mathrm{N}$', fontsize=26)
    
    #tick2 = [0, c_up_lim2/2.0, c_up_lim2]
    #cbar2 = plt.colorbar(im2, ticks=tick2)
    #cbar2.ax.tick_params( labelsize='x-small')
    #cbar2.set_label('$\mathrm{N}$', fontsize=26)
    
    plt.savefig(path + 'plots/eROSITA_ATHENA_expectation_LDDE.pdf', dpi=300)
    plt.show()
    #plt.close()