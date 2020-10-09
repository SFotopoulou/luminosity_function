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
from Likelihood import Marshall_Likelihood
import time

t1 = time.time()
class Params: pass
params = Params()

def set_params(model, params_in):
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
        return params
    
    if model == 'PDE':
        params.L0 = params_in[0]
        params.g1 = params_in[1]
        params.g2 = params_in[2]
        params.p1 = params_in[3]
        params.p2 = params_in[4]
        params.zc = params_in[5]
        params.Norm = params_in[6]
        return params                    

    if model == 'ILDE':
        params.L0 = params_in[0]
        params.g1 = params_in[1]
        params.g2 = params_in[2]
        params.p1 = params_in[3]
        params.p2 = params_in[4]
        params.Norm = params_in[5]        
        return params
    
    if model == 'LADE':
        params.L0 = params_in[0]
        params.g1 = params_in[1]
        params.g2 = params_in[2]
        params.p1 = params_in[3]
        params.p2 = params_in[4]
        params.zc = params_in[5]      
        params.d = params_in[6]      
        params.Norm = params_in[7]             
        return params
    
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
        return params

path = '/home/Sotiria/workspace/Luminosity_Function/src/LF_plots/forPaper/'
param_path = '/home/Sotiria/workspace/Luminosity_Function/src/MultiNest/new/'

# model
fields = ['MAXI', 'HBSS','COSMOS','AEGIS', 'LH', 'X_CDFS']
model = 'LDDE'

#for i in range(0, len(LF_config.fields)):
comp_fld = LF_config.fields[0]
print
print comp_fld
print
prior_vol = 1.0
for key in LF_config.Prior:
    dp = LF_config.Prior[key][1] - LF_config.Prior[key][0]
    prior_vol = prior_vol * dp 

for ref_fld in fields:
    print ref_fld
    # the reference field provides the posterior draws (importance sampling)
    reference_folder_name = ref_fld + '_' +'_ztype_'+LF_config.ztype+"_"+ model + '_pl' + str(LF_config.pl) + '_zunc_' + str(LF_config.z_unc)    
    parameters = np.loadtxt(param_path + reference_folder_name + '/1-.txt')
    params_in = parameters[:, 2:]
    

    # for the comparison field we calculate the prior*like on the posterior draws of the reference field    
    comparison_folder_name = comp_fld + '_' +'_ztype_'+LF_config.ztype+"_"+ model + '_pl' + str(LF_config.pl) + '_zunc_' + str(LF_config.z_unc)
    comparison_stats = open(param_path + comparison_folder_name + '/1-stats.dat').readlines()[0].split()
    comparison_evidence = float(comparison_stats[2])
    
    comp_Like = []
    for row in range(0, len(params_in[:,0])):
        params = set_params(model, params_in[row,:])
        Likelihood = -Marshall_Likelihood(params)/2.0
        comp_Like.append( Likelihood )
        
    comp_Like = ( np.array(comp_Like) - np.log( prior_vol ) )/comparison_evidence

    np.savetxt(param_path+ 'KLD/' + ref_fld+ '_' + comp_fld + '_' + model + '_loglikelihood.dat', zip( np.array(comp_Like)) )
print "time lapsed:", time.time() - t1