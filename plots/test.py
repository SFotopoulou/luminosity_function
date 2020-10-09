import sys
# Add the module path to the sys.path list
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/LF_modules/models')
sys.path.append('/home/sotiria/Documents/AEGIS/James/metalf-code')
import numpy as np
import time, itertools
from numpy import hstack,vstack,arange,savetxt, linspace,logspace,sqrt,log10,array,ones,tile, median, asarray, genfromtxt, power, column_stack
from Source import *
from expectation_Survey import get_data, get_area
from cosmology import dif_comoving_Vol
import matplotlib.pyplot as plt
import scipy.integrate
from LFunctions import *
from scipy.integrate import simps
from AGN_LF_config import LF_config
import LFunctions as lf
import pyfits
def get_catalog_data(fdata, key='redshift', select_min=-99, select_max=1000):
    """ returns fits subcatalog according to key selection"""
    tdata = fdata[fdata.field(key) >= select_min ]
    data = tdata[tdata.field(key) <= select_max ]
    return data
get_data()
#   Compute at:
#Lbin = [40.0, 40.5, 41.0, 41.5, 42.0, 42.5, 43.0, 43.5, 44.0, 44.5]
#Lmean = [40.25, 40.75, 41.25, 41.75, 42.25, 42.75, 43.25, 43.75, 44.25]
#zspace = linspace(0.01, 1.2 , 20)

Lbin = [42.5, 43.0, 43.5, 44.0, 44.5, 45.0, 45.5]
Lmean = [42.75, 43.25, 43.75, 44.25, 44.75, 45.25]
zspace = linspace(1.2, 3.5, 20)

# Output plot

fig_size = [10, 10]
fig = plt.figure(figsize=fig_size)
fig.subplots_adjust(left=0.175,  right=0.95, bottom=0.10, top=0.95, wspace=0.0, hspace=0.0)


##################################################################################    
# metaLF
# read distribution
meta_path = '/home/sotiria/workspace/Luminosity_Function/input_files/'
meta_name = 'meta_LF_distribution.fits'
meta_filename = meta_path + meta_name
meta_in = pyfits.open(meta_filename)
column = meta_in[1].columns
# select z outside loop, select L inside loop

meta_draws = get_catalog_data(meta_in[1].data, 'col2', min(zspace), max(zspace))

L = np.unique(meta_draws.field('col1'))
Z = np.unique(meta_draws.field('col2'))
dL = abs(L[0]-L[1])
dz = abs(Z[0]-Z[1])
N = np.logspace(-20,20,1000)
M = N[1:]
def plot_metaLF_dist(color='gray'):   
    from metalf import get_percentiles
    from scipy.stats import scoreatpercentile, percentileofscore
    N_mean = []
    N_min = []
    N_max = []
    for i in xrange(0, len(Lbin)-1): 
        Number = np.array(np.zeros(len(N[:-1])))
        print i
        Lin_data = get_catalog_data(meta_draws, 'col1', Lbin[i], Lbin[i+1])
        meta_l = Lin_data.field('col1')
        meta_z = Lin_data.field('col2')   
        for lx, z in zip(meta_l, meta_z):
            area = get_area(get_flux(lx, z))
            Dv = dif_comoving_Vol(z, area)*3.40367719e-74 # Mpc^3
            NF = np.power(10.0, get_percentiles(lx,z))*Dv*dz*dL
            x = np.arange(0.0, 1.01, 0.01)
            perc = [scoreatpercentile(NF, i) for i in x]
            Number = Number + np.array([m*(percentileofscore(perc, m)/100. -percentileofscore(perc, n)/100.) for n,m in zip(N[:-1], M)])
            
        plt.plot(N[:-1], Number)
        plt.show()
#            plt.plot(perc, x)
#            plt.show()
#            print perc
        N_mean.append(np.mean(Number)/len(meta_z))
    print N_mean

#########################################################################
print 'plotting meta_LF_dist'
plot_metaLF_dist()
plt.show()
