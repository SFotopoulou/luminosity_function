import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/models')
from scipy.stats import beta
from scipy.integrate import simps
import numpy as np
import math
from AGN_LF_config import LF_config
import sys
import matplotlib.pyplot as plt
import Likelihood as lk
import pymultinest
import time
from scipy.interpolate import interp1d

if LF_config.model == 'PLE':
    parameters = [ "L0", "g1", "g2", "p1", "p2", "zc", "Norm"]
    Phi = lk.PLE_Likelihood

if LF_config.model == 'PDE':
    parameters = [ "L0", "g1", "g2", "p1", "p2", "zc", "Norm"]
    Phi = lk.PDE_Likelihood
    
if LF_config.model == 'ILDE':
    parameters = [ "L0", "g1", "g2", "p1", "p2", "Norm"]
    Phi = lk.ILDE_Likelihood
    
if LF_config.model == 'LADE':
    parameters = [ "L0", "g1", "g2", "p1", "p2", "zc", "d", "Norm"]
    Phi = lk.LADE_Likelihood

if LF_config.model == 'LDDE':
    parameters = [ "L0", "g1", "g2", "p1", "p2", "zc", "La", "a", "Norm"]
    Phi = lk.Fotopoulou_Likelihood
    
if LF_config.model == 'Fotopoulou2':
    parameters = [ "L0", "g1", "g2", "p1", "p2", "zc", "La", "a", "Norm"]
    Phi = lk.Fotopoulou2_Likelihood

if LF_config.model == 'Fotopoulou3':
    parameters = [ "L0", "g1", "g2", "p1", "p2", "zc", "a", "Norm"]
    Phi = lk.Fotopoulou3_Likelihood

if LF_config.model == 'Ueda':
    parameters = [ "L0", "g1", "g2", "p1", "p2", "zc", "La", "a", "Norm"]
    Phi = lk.Ueda_Likelihood
    
if LF_config.model == 'Ueda14':
    parameters = ['L0', 'g1', 'g2', 'p1', 'beta', 'Lp', 'p2', 'p3', 'zc1', 'zc2', 'La1', 'La2', 'a1', 'a2', 'Norm']
                    #3    4     5    6      7      8     9      10   11      12     13    14     15    16     17  
    Phi = lk.Ueda14_Likelihood
if LF_config.model == 'FDPL':    
    parameters = ['K0','K1','L0','L1','L2','g1','g2']
    Phi = lk.FDPL_Likelihood
    
if LF_config.model == 'Schechter':
    parameters = ['A', 'Lx', 'a', 'b']
    Phi = lk.Schechter_Likelihood
    
n_params = len(parameters)

def Uniform(r,x1,x2):
    return x1+r*(x2-x1)

# First run with uniform prior      
def myUniformPrior(cube, ndim, nparams):
    
    if LF_config.model != 'FDPL'and LF_config.model != 'Schechter':
        cube[0] = Uniform(cube[0], 42.0, 46.0) #L0
        cube[1] = Uniform(cube[1], -0.5, 1.5)  # g1
        cube[2] = Uniform(cube[2], 1.5, 4.0)  # g2
        cube[3] = Uniform(cube[3], 0.0, 6.0) # p1
        
        if LF_config.model == 'PLE':
            cube[4] = Uniform(cube[4], -6.0, 2.0) # p2
            cube[5] = Uniform(cube[5], 0.01, 4.0)  # zc
            cube[6] = Uniform(cube[6], -11.0, -2.0) # Norm
    
        if LF_config.model == 'PDE':
            cube[4] = Uniform(cube[4], -6.0, 2.0) # p2
            cube[5] = Uniform(cube[5], 0.01, 4.0)  # zc
            cube[6] = Uniform(cube[6], -11.0, -2.0) # Norm
    
        if LF_config.model == 'ILDE':
            cube[4] = Uniform(cube[4], -6.0, 2.0) # p2
            cube[5] = Uniform(cube[5], -11.0, -2.0) # Norm
               
        if LF_config.model == 'LADE':
            cube[4] = Uniform(cube[4], -6.0, 2.0) # p2        
            cube[5] = Uniform(cube[5], 0.01, 4.0)  # zc
            cube[6] = Uniform(cube[6], -1.0, 1.0) # d
            cube[7] = Uniform(cube[7], -11.0, -2.0) # Norm
    
        if LF_config.model == 'LDDE':
            cube[4] = Uniform(cube[4], -5.0, 1.0) # p2        
            cube[5] = Uniform(cube[5], 1.0, 4.0)  # zc
            cube[6] = Uniform(cube[6], 42.0, 46.0) # La
            cube[7] = Uniform(cube[7], 0.0, 1.0) # a
            cube[8] = Uniform(cube[8], -8.0, -4.0) # Norm
    
        if LF_config.model == 'Fotopoulou2':
            cube[4] = Uniform(cube[4], -10.0, 3.0) # p2        
            cube[5] = Uniform(cube[5], 0.01, 4.0)  # zc
            cube[6] = Uniform(cube[6], 41.0, 46.0) # La
            cube[7] = Uniform(cube[7], 0.0, 1.0) # a
            cube[8] = Uniform(cube[8], -10.0, -2.0) # Norm
    
        if LF_config.model == 'Fotopoulou3':
            cube[4] = Uniform(cube[4], -10.0, 3.0) # p2        
            cube[5] = Uniform(cube[5], 0.01, 4.0)  # zc
            cube[6] = Uniform(cube[6], 0.0, 1.0) # a
            cube[7] = Uniform(cube[7], -10.0, -2.0) # Norm
    
        if LF_config.model == 'Ueda14':
            cube[4] = Uniform(cube[4],-5.0, 5.0) # beta
            cube[5] = Uniform(cube[5], 41.0, 46.0) # Lp
            cube[6] = Uniform(cube[6],-10.0, 3.0) # p2
            cube[7] = Uniform(cube[7],-10.0, 3.0) # p2                
            cube[8] = Uniform(cube[8],0.01, 4.0)  # zc1
            cube[9] = Uniform(cube[9],0.01, 4.0)  # zc2
            cube[10] = Uniform(cube[10],41.0, 46.0) # La1
            cube[11] = Uniform(cube[11],41.0, 46.0) # La2
            cube[12] = Uniform(cube[12],-0.7, 0.7) # a1
            cube[13] = Uniform(cube[13],-0.7, 0.7) # a2
            cube[14] = Uniform(cube[14], -10.0, -2.0) # Norm
    elif LF_config.model=='FDPL':
        cube[0] = Uniform(cube[0], -7.0, -3.0) # K1
        cube[1] = Uniform(cube[1], -7.0, -3.0)  # K2
        cube[2] = Uniform(cube[2], 42.0, 46.0)  # L1
        cube[3] = Uniform(cube[3], 42.0, 46.0) # L2
        cube[4] = Uniform(cube[4], 42.0, 46.0) # L3
        cube[5] = Uniform(cube[5], 0.01, 1.5) # g1
        cube[6] = Uniform(cube[6], 1.5, 4.0) # g2
    elif LF_config.model == 'Schechter':
        cube[0] = Uniform(cube[0], -7.0, -3.0) # A
        cube[1] = Uniform(cube[1], 42.0, 46.0)  # Lx
        cube[2] = Uniform(cube[2], -3.0, 3.0)  # a
        cube[3] = Uniform(cube[3], -10.0, 10.0) # b
        
#===============================================================================
# # Subsequent runs with last result as prior
# #param_p = np.loadtxt("/run/media/Sotiria/Data/Luminosity_Function/src/MultiNest/LDDE_M_H_C_L_XC/1-ev.dat")
# #
# #prior = {}
# #for param in parameters:
# #    pdf, bins, patches = plt.hist(param_p[:,parameters.index(param)], bins=len(param_p[:,parameters.index(param)]), normed=1, histtype='step', cumulative=True)
# #    plt.clf()
# #    pdf = np.insert(pdf, 0, 0)
# #    pr = interp1d(pdf, bins)
# #    prior[param] = pr
# #
# #def myPrior(cube, ndim, nparams):
# ##    print [cube[i] for i in range(0,nparams)]   
# #    for param in parameters:
# #        prior_pp = prior[param]
# #        cube[parameters.index(param)] = prior_pp(cube[parameters.index(param)])
# #    print [cube[i] for i in range(0,nparams)]
# #    print 
#    
#===============================================================================

def myLogLike(cube, ndim, nparams):
    params = [cube[i] for i in range(0, nparams)]
    LogL = Phi(*params)
    return -0.5*LogL

##########################################################################
#Create output folder name
folder_name = '_'.join(LF_config.fields) + '_' +'_ztype_'+LF_config.ztype+"_"+ LF_config.model + '_pl' + str(LF_config.pl) + '_zunc_' + str(LF_config.z_unc)
print 'results in: ', folder_name
import os, threading, subprocess
if not os.path.exists(folder_name): os.mkdir(folder_name)
def show(filepath):
    """ open the output (pdf) file for the user """
    if os.name == 'mac': subprocess.call(('open', filepath))
    elif os.name == 'nt': os.startfile(filepath)
    elif os.name == 'posix': subprocess.call(('xdg-open', filepath))

start_time = time.time()

#progress = pymultinest.ProgressPlotter(n_params = n_params, interval_ms = 20000000)
#progress.start()

##threading.Timer(2, show, ["/chains/1-phys_live.points.pdf"]).start() # delayed opening

pymultinest.run(myLogLike,
                myUniformPrior,
                n_dims = n_params,
                n_params = n_params,
                importance_nested_sampling=False,
                multimodal=True, const_efficiency_mode=False, n_live_points=400,
                evidence_tolerance=0.5,
                n_iter_before_update=100, null_log_evidence=-1e90, 
                max_modes=100, mode_tolerance=-1e90, 
                seed=-1, 
                context=0, write_output=True, log_zero=-1e100, 
                max_iter=0, init_MPI=False,
                outputfiles_basename = folder_name+"/1-",
                resume = True,
                verbose = True,
                sampling_efficiency = 'model')
    
#progress.stop()

# lets analyse the results
a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename = folder_name+"/1-")
s = a.get_stats()

import json
json.dump(s, file('%s.json' % a.outputfiles_basename, 'w'), indent=2)
print
print "-" * 30, 'ANALYSIS', "-" * 30
print "Global Evidence:\n\t%.15e +- %.15e" % ( s['global evidence'], s['global evidence error'] )

import matplotlib.pyplot as plt
plt.clf()

# Here we will plot all the marginals and whatnot, just to show off
# You may configure the format of the output here, or in matplotlibrc
# All pymultinest does is filling in the data of the plot.

# Copy and edit this file, and play with it.

p = pymultinest.PlotMarginalModes(a)
plt.figure(figsize=(5*n_params, 5*n_params))
#plt.subplots_adjust(wspace=0, hspace=0)
for i in range(n_params):
    plt.subplot(n_params, n_params, n_params * i + i + 1)
    p.plot_marginal(i, with_ellipses = True, with_points = False, grid_points=50)
    plt.ylabel("Probability")
    plt.xlabel(parameters[i])

    for j in range(i):
        plt.subplot(n_params, n_params, n_params * j + i + 1)
        #plt.subplots_adjust(left=0, bottom=0, right=0, top=0, wspace=0, hspace=0)
        p.plot_conditional(i, j, with_ellipses = False, with_points = True, grid_points=30)
        plt.xlabel(parameters[i])
        plt.ylabel(parameters[j])

plt.savefig(folder_name+"/marginals_multinest.pdf") #, bbox_inches='tight')
#show(folder_name+"/marginals_multinest.pdf")

for i in range(n_params):
    outfile = '%s-mode-marginal-%d.pdf' % (a.outputfiles_basename,i)
    p.plot_modes_marginal(i, with_ellipses = True, with_points = False)
    plt.ylabel("Probability")
    plt.xlabel(parameters[i])
    plt.savefig(outfile, format='pdf', bbox_inches='tight')
    plt.close()

    outfile = '%s-mode-marginal-cumulative-%d.pdf' % (a.outputfiles_basename,i)
    p.plot_modes_marginal(i, cumulative = True, with_ellipses = True, with_points = False)
    plt.ylabel("Cumulative probability")
    plt.xlabel(parameters[i])
    plt.savefig(outfile, format='pdf', bbox_inches='tight')
    plt.close()

print "take a look at the pdf files in chains/" 
print "Runtime: ",(time.time()-start_time)/3600., "hours"

