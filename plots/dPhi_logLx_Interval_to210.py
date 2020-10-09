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
#import astroML as ML


class Params: pass
params = Params()

def Phi(model, Lx, z, params_in):    
    """ 
    The luminosity function model 
    """
    if model == 'PLE':
        params.L0 = params_in[:, 0]
        params.g1 = params_in[:, 1]
        params.g2 = params_in[:, 2]
        params.p1 = params_in[:, 3]
        params.p2 = params_in[:, 4]
        params.zc = params_in[:, 5]
        params.Norm = params_in[:, 6]
        return PLE(Lx, z, params)
    
    if model == 'PDE':
        params.L0 = params_in[:, 0]
        params.g1 = params_in[:, 1]
        params.g2 = params_in[:, 2]
        params.p1 = params_in[:, 3]
        params.p2 = params_in[:, 4]
        params.zc = params_in[:, 5]
        params.Norm = params_in[:, 6]
        return PDE(Lx, z, params)                          

    if model == 'ILDE':
        params.L0 = params_in[:, 0]
        params.g1 = params_in[:, 1]
        params.g2 = params_in[:, 2]
        params.p1 = params_in[:, 3]
        params.p2 = params_in[:, 4]
        params.Norm = params_in[:, 5]        
        return ILDE(Lx, z, params)
    
    if model == 'LADE':
        params.L0 = params_in[:, 0]
        params.g1 = params_in[:, 1]
        params.g2 = params_in[:, 2]
        params.p1 = params_in[:, 3]
        params.p2 = params_in[:, 4]
        params.zc = params_in[:, 5]      
        params.d = params_in[:, 6]      
        params.Norm = params_in[:, 7]             
        return LADE(Lx, z, params)                          

    if model == 'LDDE':
        params.L0 = params_in[:, 0]
        params.g1 = params_in[:, 1]
        params.g2 = params_in[:, 2]
        params.p1 = params_in[:, 3]
        params.p2 = params_in[:, 4]
        params.zc = params_in[:, 5]
        params.La = params_in[:, 6]
        params.a = params_in[:, 7]
        params.Norm = params_in[:, 8]       
        return Fotopoulou(Lx, z, params)

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

path = '/home/Sotiria/workspace/Luminosity_Function/src/LF_plots/forPaper/'
param_path = '/home/Sotiria/workspace/Luminosity_Function/src/MultiNest/'

# Vmax Points
z_min_Vmax, z_max_Vmax, z_Vmax, L_min_Vmax, L_max_Vmax, L_Vmax, N_Vmax, dPhi_Vmax, dPhi_err_Vmax = np.loadtxt( path + 'data/Vmax_dPhi_literature.dat', unpack=True )

# LF grid
redshift = np.append(np.array(0), np.unique(z_Vmax))
luminosity = np.linspace(LF_config.Lmin, LF_config.Lmax, 30)

# model
models = ['LDDE']#, 'LADE', 'ILDE', 'PDE', 'PLE']

for model in models:
    out_file = open(path + 'data/dPhi_interval_'+model+'_to210.dat', 'w')
    #folder_name = '_'.join(LF_config.fields) + '_' +'_ztype_'+LF_config.ztype+"_"+ model + '_pl' + str(LF_config.pl) + '_zunc_' + str(LF_config.z_unc)
    folder_name = '/home/Sotiria/Dropbox/transfer/XLF_output_files/combination_fields/All_Coherent'

    
    parameters = np.loadtxt( folder_name + '/1-.txt')
    print parameters[:,2:]
    factor = np.zeros_like(parameters[:,2:])
    factor[:,0] = factor[:,0] + 0.3098
    factor[:,6] = factor[:,6] + 0.3098
    
    params_in = parameters[:, 2:] + factor 
    print params_in
    n_params = len(params_in[0,:])
    
    LF_modes = np.loadtxt(folder_name + '/1-summary.txt')
    n_modes = len(LF_modes)
    mean_params = LF_modes[:, 0:n_params]
    mean_err_params = LF_modes[:, n_params:2*n_params]
    MLE_params = LF_modes[:, 2*n_params:3*n_params]
    MAP_params = LF_modes[:, 3*n_params:4*n_params]

    modes = []
    for i in range(1, n_modes+1):
        modes.append('mode'+str(i))
        
    out_file.write('#L z dPhi_low_90 dPhi_high_90 dPhi_low_99 dPhi_high_99 dPhi_mode dPhi_mean dPhi_median '+' '.join(modes)+'\n')

    for z_in in redshift:
        for L_in in luminosity:
            # full dPhi distribution
            dPhi_dist = np.log10(Phi(model, L_in, z_in, params_in))
            # calculate 90%,99% interval and dPhi at the highest probability mode of the parameter distribution.
            dPhi_low_90, dPhi_high_90, dPhi_mode = credible_interval(dPhi_dist, 90)
            dPhi_low_99, dPhi_high_99, dPhi_mode = credible_interval(dPhi_dist, 99)
            # MLE parameters, for each mode
            #dPhi_modes = []
            #for i in range(0, n_modes):
                #print MLE_params[i,:]
            dPhi_modes = np.log10(Phi(model, L_in, z_in, MLE_params))
                #dPhi_modes.append(Phi_m)
            dPhi_modes = [str(P) for P in dPhi_modes]
            out_file.write( str(L_in) +' '+ str(z_in) +' '+ str(dPhi_low_90) +' '+str(dPhi_high_90) +' '+str(dPhi_low_99)+\
                            ' '+str(dPhi_high_99)+' '+str(dPhi_mode)+' '+str(np.mean(dPhi_dist))+' '+str(np.median(dPhi_dist))+' '+str(' '.join(dPhi_modes))+'\n')
    out_file.close()