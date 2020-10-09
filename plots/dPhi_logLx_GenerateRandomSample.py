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
        params.L0 = params_in[0]
        params.g1 = params_in[1]
        params.g2 = params_in[2]
        params.p1 = params_in[3]
        params.p2 = params_in[4]
        params.zc = params_in[5]
        params.Norm = params_in[6]
        return PLE(Lx, z, params)
    
    if model == 'PDE':
        params.L0 = params_in[0]
        params.g1 = params_in[1]
        params.g2 = params_in[2]
        params.p1 = params_in[3]
        params.p2 = params_in[4]
        params.zc = params_in[5]
        params.Norm = params_in[6]
        return PDE(Lx, z, params)                          

    if model == 'ILDE':
        params.L0 = params_in[0]
        params.g1 = params_in[1]
        params.g2 = params_in[2]
        params.p1 = params_in[3]
        params.p2 = params_in[4]
        params.Norm = params_in[5]        
        return ILDE(Lx, z, params)
    
    if model == 'LADE':
        params.L0 = params_in[0]
        params.g1 = params_in[1]
        params.g2 = params_in[2]
        params.p1 = params_in[3]
        params.p2 = params_in[4]
        params.zc = params_in[5]      
        params.d = params_in[6]      
        params.Norm = params_in[7]             
        return LADE(Lx, z, params)                          

    if model == 'LDDE':
        params.L0 = params_in[0]
        params.g1 = params_in[1]
        params.g2 = params_in[2]
        params.p1 = params_in[3]
        params.p2 = params_in[4]
        params.zc = params_in[5]
        params.La = params_in[6]
        params.a = params_in[7]
        params.Norm = params_in[8]       
        return Fotopoulou(Lx, z, params)

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


path = '/home/Sotiria/workspace/Luminosity_Function/src/LF_plots/forPaper/'
param_path = '/home/Sotiria/workspace/Luminosity_Function/src/MultiNest/safe_keep/'

get_data()
import time
zbin = 501
Lbin = 501
zlim = 4.5
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
for fld in ['MAXI', 'HBSS', 'COSMOS', 'LH', 'X_CDFS', 'AEGIS']: #LF_config.fields:
    z_data = lum_data[lum_data.field('Z_'+fld) > 0]
    use_data = z_data[z_data.field('Z_'+fld) <= zlim]
    data_logL.extend( use_data.field( 'Lum_'+fld ) )
    data_Z.extend( use_data.field( 'Z_'+fld ) ) 

# model
models = ['data', 'LDDE', 'PLE', 'LADE', 'PDE', 'ILDE']

c_low_lim, c_up_lim=0, 25

for model in models:
    
    fig = plt.figure(figsize=(10,10))
    
    if model == 'data':
        nullfmt   = NullFormatter()         # no labels
        # definitions for the axes
        left, width = 0.11, 0.65
        bottom, height = 0.11, 0.65
        bottom_h = left_h = left+width
        
        rect_2dhist = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]
        
        # start with a rectangular Figure
        #plt.figure(1, figsize=(8,8))
        
        ax2Dhist = plt.axes(rect_2dhist)
        axHistx = plt.axes(rect_histx)
        axHisty = plt.axes(rect_histy)
        # no labels
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)
        
        
        lum_data = pyfits.open(LF_config.data_out_name)[1].data
        
        data_Z = []
        data_logL = []
        for fld in ['MAXI', 'HBSS', 'COSMOS', 'LH', 'X_CDFS', 'AEGIS']: #LF_config.fields:
            z_data = lum_data[lum_data.field('Z_'+fld)>0]
            use_data = z_data[z_data.field('Z_'+fld)<=4.5]
            data_logL.extend( use_data.field( 'Lum_'+fld ) )
            data_Z.extend( use_data.field( 'Z_'+fld ) ) 
            
        zbin = 25
        Lbin = 25
        N, yedges, xedges = np.histogram2d(data_logL, data_Z, bins=(Lbin, zbin))
        im = ax2Dhist.imshow(N, origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='bone_r', aspect='auto')
        
        ax2Dhist.set_xlim( (0.00, 4.5) )
        ax2Dhist.set_ylim( (41, 46) )
        im.set_clim(c_low_lim, c_up_lim)
        
        
        ax2Dhist.set_ylabel('$\mathrm{log(\,L_x /erg\cdot s^{-1})}$', fontsize='large')
        ax2Dhist.set_xlabel('$\mathrm{z}$', fontsize='large')
        ax2Dhist.text(3.5, 41.5, '$\mathrm{data}$')
        ax2Dhist.text(4.85, 46.5, '$\mathrm{N='+str(len(data_Z))+'}$')
        
        
        axHistx.hist(data_Z, bins = zbin, histtype='step', color='k', normed=False)
        axHistx.set_xlim( ax2Dhist.get_xlim() )
        axHistx.set_ylim((0,120))
        axHistx.set_yticks([])
        
        axHisty.hist(data_logL, bins = Lbin, orientation='horizontal', histtype='step', color='k', normed=False)
        axHisty.set_ylim( ax2Dhist.get_ylim() )
        axHisty.set_xlim((0,150))
        axHisty.set_xticks([])

    else:    
        nullfmt   = NullFormatter()         # no labels
        # definitions for the axes
        left, width = 0.11, 0.65
        bottom, height = 0.11, 0.65
        bottom_h = left_h = left+width
        
        rect_2dhist = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]
        
        # start with a rectangular Figure
#        plt.figure(1, figsize=(8,8))
        
        ax2Dhist = plt.axes(rect_2dhist)
        axHistx = plt.axes(rect_histx)
        axHisty = plt.axes(rect_histy)
        # no labels
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)
        print
    
        folder_name = '_'.join(LF_config.fields) + '_' +'_ztype_'+LF_config.ztype+"_"+ model + '_pl' + str(LF_config.pl) + '_zunc_' + str(LF_config.z_unc)
        parameters = np.loadtxt(param_path + folder_name + '/1-.txt')
        params_in = parameters[:, 2:]
        n_params = len(params_in[0,:])
        
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
        vol = np.loadtxt( path + 'data/Volume_for_N.dat')
        P = []
        for j in range(0, len(LL)-1):
            #print model, LL[j]
            #vol = []
            #for i in range(0, len(ZZ)-1):    
                #print round(LL[j],3), round(ZZ[i],3), ph 
                #vol.append(calc_Vol(LL[j], LL[j+1], ZZ[i], ZZ[i+1]))
                
            #vol = np.array(vol)
            #N_z = np.where(vol>0, vol*Phi(model, LL[j]+Lbin_width, np.array(ZZ[:-1])+zbin_width, MLE_params), 0 )
            
            P.append(list(Phi(model, LL[j]+Lbin_width, np.array(ZZ[:-1])+zbin_width, MLE_params))) 
               
        N = vol * P    
        
        try:
            np.savetxt( path + 'data/Expected_N_'+model+'.dat', N)
        except:
            pass
        #print np.shape(N)
        N = rebin_2darray(N, data_Lbin-1, data_zbin-1) 
        
        #print np.shape(N)
        im = ax2Dhist.imshow(N, origin='low', extent=[0.01, zlim, 41,46], cmap='bone_r', aspect='auto')
        ax2Dhist.set_xlim( (0.0, zlim) )
        ax2Dhist.set_ylim( (41, 46) )
        im.set_clim(c_low_lim, c_up_lim)
        
        ax2Dhist.set_ylabel('$\mathrm{log(\,L_x /erg\cdot s^{-1})}$', fontsize='large')
        ax2Dhist.set_xlabel('$\mathrm{z}$', fontsize='large')
        ax2Dhist.text(3.5, 41.5, '$\mathrm{'+model+'}$')
        ax2Dhist.text(4.85, 46.5, '$\mathrm{N='+str(int(np.sum(N)))+'}$')
        #zz_test = np.sum(N, axis=0)
        #print np.shape(zz_test), len(ZZ)
    
        axHistx.plot(ZZ_plot[:-1]+zbin_width_plot, np.sum(N, axis=0), color='k')
        axHistx.hist(data_Z, bins = data_zbin-1, histtype='step', color='gray', normed=False)
        axHistx.set_xlim( ax2Dhist.get_xlim() )
        #axHistx.set_ylim((0,1.8))
        axHistx.set_ylim((0,120))
        axHistx.set_yticks([])
        
        axHisty.plot(np.sum(N, axis=1), LL_plot[:-1]+Lbin_width_plot, color='k')
        axHisty.hist(data_logL, bins = data_Lbin-1, orientation='horizontal', histtype='step', color='gray', normed=False)
        axHisty.set_ylim( ax2Dhist.get_ylim() )
        #axHisty.set_xlim((0,0.8))
        axHisty.set_xlim((0,150))
    
        axHisty.set_xticks([])
        #CB = plt.colorbar(im, orientation='horizontal')
        #CB.ax.set_position([0.125, -0.45, 0.5, 0.7])

    #plt.savefig(path + 'plots/Expected_N_'+str(data_zbin-1)+'x'+str(data_Lbin-1)+model+'.pdf', dpi=300)
    #print model, (time.time()-t1)/60, "min"
plt.show()
    #plt.close()