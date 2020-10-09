import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/models')

import scipy
import numpy
import math
from AGN_LF_config import LF_config
import sys
import matplotlib.pyplot as plt
import pyapemost
import Likelihood as lk
import time
#print "I am Bayes"

############################################################################
def myprior(mp, oldparams):
    
    return 0.

def myloglike(mp, oldparams):
    m = mp.contents
    params = [m.params.contents.data[i] for i in range(m.n_params)]  
    LogL = lk.Fotopoulou_Likelihood(*params)
    
    return -LogL
       
###########################################################################

outpath = '/home/Sotiria/workspace/Luminosity_Function/output_files/'
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
    #cmd = ["run", "analyse"]
    #cmd = ["calibrate"]
else:
    cmd = sys.argv[1:]

if "calibrate" in cmd:
    t_cal = time.time()
    pyapemost.calibrate()
    print "Calibration time:", time.time() - t_cal
# run APEMoST
if "run" in cmd:
    #w = pyapemost.watch.ProgressWatcher()
    #w.start()
    t_run = time.time()
    pyapemost.run(max_iterations = 400000, append = True)
    print "Run time:", time.time() - t_run
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
