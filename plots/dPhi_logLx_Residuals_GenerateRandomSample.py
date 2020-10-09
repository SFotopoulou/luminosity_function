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
from LFunctions import PLE,PDE,ILDE,LADE,Fotopoulou, Fotopoulou2, Fotopoulou3, Ueda14
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
        params.L0 = params_in[:,0]
        params.g1 = params_in[:,1]
        params.g2 = params_in[:,2]
        params.p1 = params_in[:,3]
        params.p2 = params_in[:,4]
        params.zc = params_in[:,5]
        params.Norm = params_in[:,6]
        return PLE(Lx, z, params)
    
    if model == 'PDE':
        params.L0 = params_in[:,0]
        params.g1 = params_in[:,1]
        params.g2 = params_in[:,2]
        params.p1 = params_in[:,3]
        params.p2 = params_in[:,4]
        params.zc = params_in[:,5]
        params.Norm = params_in[:,6]
        return PDE(Lx, z, params)                          

    if model == 'ILDE':
        params.L0 = params_in[:,0]
        params.g1 = params_in[:,1]
        params.g2 = params_in[:,2]
        params.p1 = params_in[:,3]
        params.p2 = params_in[:,4]
        params.Norm = params_in[:,5]        
        return ILDE(Lx, z, params)
    
    if model == 'LADE':
        params.L0 = params_in[:,0]
        params.g1 = params_in[:,1]
        params.g2 = params_in[:,2]
        params.p1 = params_in[:,3]
        params.p2 = params_in[:,4]
        params.zc = params_in[:,5]      
        params.d = params_in[:,6]      
        params.Norm = params_in[:,7]             
        return LADE(Lx, z, params)                          

    if model == 'LDDE':
        params.L0 = params_in[:,0]
        params.g1 = params_in[:,1]
        params.g2 = params_in[:,2]
        params.p1 = params_in[:,3]
        params.p2 = params_in[:,4]
        params.zc = params_in[:,5]
        params.La = params_in[:,6]
        params.a = params_in[:,7]
        params.Norm = params_in[:,8]       
        return Fotopoulou(Lx, z, params)

    if model == 'Fotopoulou2':
        params.L0 = params_in[:,0]
        params.g1 = params_in[:,1]
        params.g2 = params_in[:,2]
        params.p1 = params_in[:,3]
        params.p2 = params_in[:,4]
        params.zc = params_in[:,5]
        params.La = params_in[:,6]
        params.a = params_in[:,7]
        params.Norm = params_in[:,8]
        return Fotopoulou2(Lx, z, params)
    
    if model == 'Fotopoulou3':
        params.L0 = params_in[:,0]
        params.g1 = params_in[:,1]
        params.g2 = params_in[:,2]
        params.p1 = params_in[:,3]
        params.p2 = params_in[:,4]
        params.zc = params_in[:,5]
        params.a = params_in[:,6]
        params.Norm = params_in[:,7]
        return Fotopoulou3(Lx, z, params)

    if model == 'Ueda14':
        params.L0 = params_in[:,0]
        params.g1 = params_in[:,1]
        params.g2 = params_in[:,2]
        params.p1 = params_in[:,3]
        params.beta = params_in[:,4]
        params.Lp = params_in[:,5]
        params.p2 = params_in[:,6]
        params.p3 = params_in[:,7]
        params.zc1 = params_in[:,8]
        params.zc2 = params_in[:,9]
        params.La1 = params_in[:,10]
        params.La2 = params_in[:,11]
        params.a1 = params_in[:,12]
        params.a2 = params_in[:,13]
        params.Norm = params_in[:,14]
        return Ueda14(Lx, z, params)
    
    
def calc_Vol(Lmin, Lmax, zmin, zmax, zpoints=10, Lpoints=10):
    LL = np.array([np.ones( (zpoints), float )*item for item in 
                   np.linspace(Lmin, Lmax, Lpoints)])
    # make LL 1D
    L = LL.ravel()
    # repeat as many times as Lpoints
    Z = np.tile(np.logspace(np.log10(zmin), np.log10(zmax), zpoints), Lpoints) 

# Set up grid for survey integral
    vecFlux = np.vectorize(get_flux)
    temp_Fx = vecFlux(L, Z)
    area = get_area(temp_Fx)   
   
    vecDifVol = np.vectorize(dif_comoving_Vol) 
    DVc = np.where( area>0, vecDifVol(Z, area), 0) 
    DVcA = DVc*3.4036771e-74 # vol in Mpc^3

        
    Redshift_int = Z[0:zpoints]
    Luminosity_int = np.linspace(Lmin, Lmax, Lpoints)
    
    y = []
    
    count_r = xrange(0, Lpoints)
    for count in count_r:
        startz = count * zpoints
        endz = startz + zpoints
        x = DVcA[startz:endz]
        
        int1 = simps(x, Redshift_int, even='last')
        y.append(int1)
    
    DV_int = simps(y, Luminosity_int, even='last')
    return DV_int

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

def credible_interval(distribution, level, bin=100, plot=False):
    pdf, bins = np.histogram(distribution, bins = bin, normed=True)
#    if plot==True:
#        plt.plot(bins[:-1], pdf)
#        plt.show()
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

import time
t1 = time.time()
path = '/home/Sotiria/workspace/Luminosity_Function/src/LF_plots/forPaper/'
#param_path = '/home/Sotiria/workspace/Luminosity_Function/src/MultiNest/'
param_path = '/home/Sotiria/Dropbox/transfer/XLF_output_files/combination_fields/'
#get_data()
#import time
zbin = 1001
Lbin = 1001
zlim = 4.0
data_zbin = 26
data_Lbin = 26
# plot LF grid
ZZ_plot = np.linspace(LF_config.zmin, zlim, data_zbin)
LL_plot = np.linspace(LF_config.Lmin, LF_config.Lmax, data_Lbin)
zbin_width_plot = (ZZ_plot[1] - ZZ_plot[0])/2.0
Lbin_width_plot = (LL_plot[1] - LL_plot[0])/2.0


# LF grid
ZZ = np.linspace(LF_config.zmin, zlim, zbin)
LL = np.linspace(LF_config.Lmin, LF_config.Lmax, Lbin)
zbin_width = (ZZ[1] - ZZ[0])/2.0
Lbin_width = (LL[1] - LL[0])/2.0

lum_data = pyfits.open(LF_config.data_out_name)[1].data

data_Z = []
data_logL = []
for fld in ['MAXI', 'HBSS', 'XMM_COSMOS', 'Chandra_COSMOS','LH',  'AEGIS','XMM_CDFS',  'Chandra_CDFS']: #LF_config.fields:
    z_data = lum_data[lum_data.field('Z_'+fld) > 0]
    use_data = z_data[z_data.field('Z_'+fld) <= zlim]
    data_logL.extend( use_data.field( 'Lum_'+fld ) )
    data_Z.extend( use_data.field( 'Z_'+fld ) ) 

data_N, yedges, xedges = np.histogram2d(data_logL, data_Z, bins=(LL_plot, ZZ_plot))

# model
models = ['Ueda14', 'LDDE', 'LADE', 'ILDE', 'PDE', 'PLE']#, 'Fotopoulou2', 'Fotopoulou3']
c_low_lim, c_up_lim=-2, 2
for model in models:
    
    fig = plt.figure(figsize=(10,10))

#    fig.add_subplot(2,3,models.index(model)+1)

    nullfmt   = NullFormatter()         # no labels
    # definitions for the axes
    left, width = 0.15, 0.60
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
    print

    folder_name = 'All_' + model
    parameters = np.loadtxt(param_path + folder_name + '/1-.txt')
    params_in = parameters[::10, 2:]
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

#    plt.imshow(N)
#    plt.show()

    vol = np.loadtxt( path + 'data/Volume_for_N_1000.dat')
    
    P = []
    P_min = []
    P_max = []
    print model, len(params_in)
    for j in range(0, len(LL)-1):
        print LL[j]
        temp_p = []
        temp_p_min = []
        temp_p_max = []
        #tt1 = time.time()
        for i in range(0, len(ZZ)-1): 
            tt2  = time.time() 
            n_distr = Phi(model, LL[j]+Lbin_width, ZZ[i]+zbin_width, params_in)
           # print 'Phi distr. time', time.time() - tt2         
            #print LL[j], ZZ[i]+zbin_width
            #print LL[i], ZZ[i], np.min(n_distr), np.max(n_distr), vol[j,i]
            
            #if ZZ[i]>1.29:
            #    n_min, n_max, n_mode = credible_interval(n_distr, 90., plot=True)
            #else:
            tt3 = time.time()    
            if vol[j,i]>0:
                n_min, n_max, n_mode = credible_interval(n_distr, 90.)
 
            else:
                n_min, n_max, n_mode = 0, 0, 0
          #  print 'credible interval time', time.time() - tt3
            #print n_min, n_max, n_mode, vol[j,i]
            #print
            tt4 = time.time()
            temp_p.append(n_mode)
            temp_p_min.append(n_min)
            temp_p_max.append(n_max)
         #   print 'small append time', time.time() - tt4
        #print
        #tt5 = time.time()
        P.append(temp_p)
        P_min.append(temp_p_min)
        P_max.append(temp_p_max)
        #print 'append time', time.time() - tt5    
        #P.append(list(Phi(model, LL[j]+Lbin_width, np.array(ZZ[:-1])+zbin_width, MLE_params))) 
        #print 'all z', time.time() - tt1
        
    N = vol * P    
    N_min = vol * P_min
    N_max = vol * P_max
    
    try:
        np.savetxt( path + 'data/20151103_Expected_N_'+model+'.dat', N)
        np.savetxt( path + 'data/20151103_Expected_N_'+model+'.dat', N_min)
        np.savetxt( path + 'data/20151103_Expected_N_'+model+'.dat', N_max)
    except:
        pass
    #print np.shape(N)

    
    #N = np.loadtxt( path + 'data/20151103_Expected_N_'+model+'.dat')

    N = rebin_2darray(N, data_Lbin-1, data_zbin-1) 
    
    #print np.shape(N)
    im = ax2Dhist.imshow((N-data_N)/data_N, origin='low', extent=[min(xedges), max(xedges), min(yedges), max(yedges)], cmap='RdGy', aspect='auto')
    ax2Dhist.set_xlim( (0.0, zlim) )
    ax2Dhist.set_ylim( (41, 46) )
    im.set_clim(c_low_lim, c_up_lim)

    ax2Dhist.set_ylabel('$\mathrm{log(\,L_x /erg\cdot s^{-1})}$', fontsize='large')
    ax2Dhist.set_xlabel('$\mathrm{z}$', fontsize='large')
    #ax2Dhist.xticks([])
    ax2Dhist.text(2.75, 41.6, '$\mathrm{'+model+'}$', fontsize=34)
    number = int(np.sum(N))
    up_lim = int(np.sum(N_max)) - int(np.sum(N))
    low_lim = int(np.sum(N)) - int(np.sum(N_min))
    print int(np.sum(N_min)), int(np.sum(N)), int(np.sum(N_max))
    ax2Dhist.text(4.15, 46.5, '$\mathrm{N='+str(number)+'^{+'+str(up_lim)+'}_{-'+str(low_lim)+'} }$')
    
    #zz_test = np.sum(N, axis=0)
    #print np.shape(zz_test), len(ZZ)
    #print "histogram zz_plot: ", ZZ_plot[:-1]
    #print "histogram zz_plot++zbin_width_plot", ZZ_plot[:-1]+zbin_width_plot
    #print 'data xedges: ', xedges
    #print 'LF bins: ', ZZ[:-1]+zbin_width
    
    axHistx.plot(ZZ_plot[:-1]+zbin_width_plot, np.sum(N, axis=0), color='k', lw=4)
    axHistx.hist(data_Z, bins = data_zbin-1, lw=4, histtype='step', color='gray', normed=False)
    axHistx.set_xlim( ax2Dhist.get_xlim() )
    #axHistx.set_ylim((0,1.8))
    axHistx.set_ylim((0,120))
    axHistx.set_yticks([])
    
    axHisty.plot(np.sum(N, axis=1), LL_plot[:-1]+Lbin_width_plot, color='k', lw=4)
    axHisty.hist(data_logL, bins = data_Lbin-1, lw=4, orientation='horizontal', histtype='step', color='gray', normed=False)
    axHisty.set_ylim( ax2Dhist.get_ylim() )
    #axHisty.set_xlim((0,0.8))
    axHisty.set_xlim((0,150))

    axHisty.set_xticks([])
    tick = [-2, -1, 0, 1, 2]
    cbar = plt.colorbar(im)
    cbar.ax.tick_params( labelsize='x-small')
    cbar.set_label('$\mathrm{(N_{model}-N_{data})/N_{data}}$', fontsize=26)
    plt.savefig(path + 'plots/20151103_Residuals_N_'+str(data_zbin-1)+'x'+str(data_Lbin-1)+'_'+model+'.pdf', dpi=300)
    print model, (time.time()-t1)/60, "min"
    plt.show()
    plt.close()