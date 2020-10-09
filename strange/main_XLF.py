import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/LF_modules/models')

import numpy as np
from AGN_LF_config import LF_config
from cosmology import Cosmo
from Survey import Survey
from Source import Source
from LFunctions import Models
from Likelihood import Marshall_Likelihood
from SetUp_grid import set_up_grid
from SetUp_data import set_up_data 
from scipy.integrate import simps
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
import time

class Params: pass
params = Params()

def PLE_Likelihood(L0, g1, g2, p1, p2, zc, Norm):
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.zc = zc
    params.Norm = -5.0
    return Marshall_Likelihood(params)

def PDE_Likelihood(L0, g1, g2, p1, p2, zc, Norm):
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.zc = zc
    params.Norm = -5.0
    return Marshall_Likelihood(params)

def ILDE_Likelihood(L0, g1, g2, p1, p2, Norm):
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.Norm = -5.0
    return Marshall_Likelihood(params)

def Hasinger_Likelihood(L0, g1, g2, p1, p2, zc, La, a, b1, b2):
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.zc = zc
    params.La = La
    params.a = a
    params.b1 = b1
    params.b2 = b2
    params.Norm = -5.0
    return Marshall_Likelihood(params)
    
def Ueda_Likelihood(L0, g1, g2, p1, p2, zc, La, a, Norm):
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.zc = zc
    params.La = La
    params.a = a
    params.Norm = -5.0
    return Marshall_Likelihood(params)
    
def LADE_Likelihood(L0, g1, g2, p1, p2, zc, d, Norm):                          
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.zc = zc
    params.d = d
    params.Norm = -5.0
    return Marshall_Likelihood(params)
    
def Miyaji_Likelihood(L0, g1, g2, p1, p2, zc, La, a, pmin, Norm):
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.zc = zc
    params.La = La
    params.a = a
    params.pmin = pmin
    params.Norm = -5.0
    return Marshall_Likelihood(params)
    
def Fotopoulou_Likelihood(L0, g1, g2, p1, p2, zc, La, a, Norm):
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.zc = zc
    params.La = La
    params.a = a
    params.Norm = -5.0
    return Marshall_Likelihood(params)



start_time = time.time()
############### Observations ###############
# Prepare individual grid from each datum ##
############################################
    if LF_config.model == 'PLE':
        LogL = PLE_Likelihood(*params)
    
    elif LF_config.model == 'PDE':
        LogL = PDE_Likelihood(*params)                          
    
    elif LF_config.model == 'ILDE':
        LogL = ILDE_Likelihood(*params)
    
    elif LF_config.model == 'Hasinger':
        LogL = Hasinger_Likelihood(*params)
    
    elif LF_config.model == 'Ueda':
        LogL = Ueda_Likelihood(*params)
    
    elif LF_config.model == 'LADE':
        LogL = LADE_Likelihood(*params)                          
    
    elif LF_config.model == 'Miyaji':
        LogL = Miyaji_Likelihood(*params)
    
    elif LF_config.model == 'LDDE':
        LogL = Fotopoulou_Likelihood(*params)


##########################################################################################
def myprior(mp, oldparams):
    return 0.

def myloglike(mp, oldparams):
    m = mp.contents
    params = [m.params.contents.data[i] for i in range(m.n_params)]
    LogL = Marshall_Likelihood(*params)
    return -LogL
       
###########################################################################
import scipy
import numpy
import math
import sys
import matplotlib.pyplot as plt
import pyapemost
outpath = '/home/sotiria/workspace/Luminosity_Function/output_files/'
def show(filepath):
    """ open the output (pdf) file for the user """
    import subprocess, os
    if os.name == 'mac': subprocess.call(('open', filepath))
    elif os.name == 'nt': os.startfile(filepath)
    elif os.name == 'posix': subprocess.call(('xdg-open', filepath))

pyapemost.set_function(myloglike, myprior)

if len(sys.argv) < 2:
    #print "SYNOPSIS: %s [calibrate|run|analyse]" % sys.argv[0]
    cmd = ["calibrate", "run", "analyse"]
else:
    cmd = sys.argv[1:]

if "calibrate" in cmd:
    pyapemost.calibrate()
# run APEMoST
if "run" in cmd:
    #w = pyapemost.watch.ProgressWatcher()
    #w.start()
    pyapemost.run(max_iterations = 100000, append = False)
    #w.stop()

# lets analyse the results
if "analyse" in cmd:
    plotter = pyapemost.analyse.VisitedAllPlotter()
    plotter.plot()
    show(outpath+LF_config.model+"_chain0.pdf")
    histograms = pyapemost.create_histograms()
    i = 1
    plt.clf()
    plt.figure(figsize=(7, 4 * len(histograms)))
    for k,(v,stats) in histograms.iteritems():
        plt.subplot(len(histograms), 1, i)
        plt.plot(v[:,0], v[:,2], ls='steps--', label=k)
        plt.legend()
        print k, stats
        i = i + 1
    plt.savefig(outpath+LF_config.model+"_marginals.pdf")
    show(outpath+LF_config.model+"_marginals.pdf")
    
    print pyapemost.model_probability()
