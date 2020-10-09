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


class Params: pass
params = Params()

def Phi(model, Lx, z, params_in):    
    """ 
    The luminosity function model 
    """
    if model == 'PLE':
        params.L0 = params_in[:, 0]+ 0.3098 
        params.g1 = params_in[:, 1]
        params.g2 = params_in[:, 2]
        params.p1 = params_in[:, 3]
        params.p2 = params_in[:, 4]
        params.zc = params_in[:, 5]
        params.Norm = params_in[:, 6]
        return PLE(Lx, z, params)
    
    if model == 'PDE':
        params.L0 = params_in[:, 0]+ 0.3098 
        params.g1 = params_in[:, 1]
        params.g2 = params_in[:, 2]
        params.p1 = params_in[:, 3]
        params.p2 = params_in[:, 4]
        params.zc = params_in[:, 5]
        params.Norm = params_in[:, 6]
        return PDE(Lx, z, params)                          

    if model == 'ILDE':
        params.L0 = params_in[:, 0]+ 0.3098 
        params.g1 = params_in[:, 1]
        params.g2 = params_in[:, 2]
        params.p1 = params_in[:, 3]
        params.p2 = params_in[:, 4]
        params.Norm = params_in[:, 5]        
        return ILDE(Lx, z, params)
    
    if model == 'LADE':
        params.L0 = params_in[:, 0]+ 0.3098 
        params.g1 = params_in[:, 1]
        params.g2 = params_in[:, 2]
        params.p1 = params_in[:, 3]
        params.p2 = params_in[:, 4]
        params.zc = params_in[:, 5]      
        params.d = params_in[:, 6]      
        params.Norm = params_in[:, 7]             
        return LADE(Lx, z, params)                          

    if model == 'LDDE':
        params.L0 = params_in[:, 0]+ 0.3098 
        params.g1 = params_in[:, 1]
        params.g2 = params_in[:, 2]
        params.p1 = params_in[:, 3]
        params.p2 = params_in[:, 4]
        params.zc = params_in[:, 5]
        params.La = params_in[:, 6]+ 0.3098 
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


# LF grid
redshift = np.linspace(2.5, 7., 10)
luminosity = [43.56, 44.15]
#luminosity = [44.15, 45.1]
# model
models = ['LDDE', 'LADE', 'ILDE', 'PDE', 'PLE']

for model in models:
    out_file = open(path + 'data/Ndensity_low_interval_onlyCOSMOS_'+model+'.dat', 'w')
    folder_name = 'COSMOS' + '_' +'_ztype_'+LF_config.ztype+"_"+ model + '_pl' + str(LF_config.pl) + '_zunc_' + str(LF_config.z_unc)

    
    parameters = np.loadtxt(param_path + folder_name + '/1-.txt')
    params_in = parameters[:, 2:]
    
    n_params = len(params_in[0,:])
    
    LF_modes = np.loadtxt(param_path + folder_name + '/1-summary.txt')
    n_modes = len(LF_modes)
    mean_params = LF_modes[:, 0:n_params]
    mean_err_params = LF_modes[:, n_params:2*n_params]
    MLE_params = LF_modes[:, 2*n_params:3*n_params]
    MAP_params = LF_modes[:, 3*n_params:4*n_params]

    modes = []
    for i in range(1, n_modes+1):
        modes.append('mode'+str(i))
        
    out_file.write('#Lmin Lmax z Ndensity_low_90 Ndensity_high_90 Ndensity_low_99 Ndensity_high_99 Ndensity_mode Ndensity_mean Ndensity_median '+' '.join(modes)+'\n')
    print model
    for i in range(0, len(luminosity)-1): 
        L_bin = np.linspace(luminosity[i], luminosity[i+1], 10)
        print luminosity[i]
        for z_in in redshift:
            print z_in
            # full Ndensity distribution
            Ndensity_dist = []
            for k in range(0, len(params_in)):
                arr = np.array([params_in[k,:]])
                Phi_dist = []
                for L_in in L_bin:
                    #print L_in, z_in, np.array([params_in[k,:]])
                    Phi_dist.extend( Phi(model, L_in, z_in, np.array([params_in[k,:]]) )) 
                
                Ndensity_dist.append( np.log10( simps(Phi_dist, L_bin) ))
            
            
            # calculate 90%,99% interval and Ndensity at the highest probability mode of the parameter distribution.
            Ndensity_low_90, Ndensity_high_90, Ndensity_mode = credible_interval(Ndensity_dist, 90)
            Ndensity_low_99, Ndensity_high_99, Ndensity_mode = credible_interval(Ndensity_dist, 99)
            
            # MLE parameters, for each mode
            Ndensity_mode_dist = []
            for j in range(0, n_modes):
                Phi_dist = []
                for L_in in L_bin:
                    Phi_dist.extend( Phi(model, L_in, z_in, np.array([MLE_params[j]]) )) 
                Ndensity_mode_dist.append( np.log10( simps(Phi_dist, L_bin) ))
               
            Ndensity_modes = [str(P) for P in Ndensity_mode_dist]
            
            
            out_file.write( str(min(L_bin)) +' '+str(max(L_bin)) +' '+ str(z_in) +' '+ str(Ndensity_low_90) +' '+str(Ndensity_high_90) +' '+str(Ndensity_low_99)+\
                            ' '+str(Ndensity_high_99)+' '+str(Ndensity_mode)+' '+str(np.mean(Ndensity_dist))+' '+str(np.median(Ndensity_dist))+' '+str(' '.join(Ndensity_modes))+'\n')
    out_file.close()