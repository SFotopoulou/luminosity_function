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

#   Compute at:
#Lbin = [40.0, 40.5, 41.0, 41.5, 42.0, 42.5, 43.0, 43.5, 44.0, 44.5]
#Lmean = [40.25, 40.75, 41.25, 41.75, 42.25, 42.75, 43.25, 43.75, 44.25]
#zspace = linspace(0.01, 1.2 ,20)

Lbin = [42.5, 43.0, 43.5, 44.0, 44.5, 45.0, 45.5]
Lmean = [42.75, 43.25, 43.75, 44.25, 44.75, 45.25]
zspace = linspace(1.2, 3.5, 20)

# Output plot

fig_size = [10, 10]
fig = plt.figure(figsize=fig_size)
fig.subplots_adjust(left=0.175,  right=0.95, bottom=0.10, top=0.95, wspace=0.0, hspace=0.0)

linecolor = (0., 0., 0.)
fillcolor = (0.75, 0.75, 0.75)
pointcolor = (0., 0., 0.)
linestyles = itertools.cycle( ['-', '--',':','-.','steps'] )
colors = itertools.cycle( ['k', 'k','k','k','gray'] )


# LF fit
p2 = -1.34853227641
a = 0.260307788878
p1 = 5.8361720265
g2 = 2.55460679566
g1 = 1.03913647028
La = 44.3258637967 + 0.3098 
zc = 1.87730303823
L0 = 43.9402810729 + 0.3098
Norm = -6.34232103471

parameters = [L0, g1, g2, p1, p2, zc, La, a, Norm]

matrix = [[ 0.027414483358256400, 0.009185811147251570,  0.024585585436562200, -0.025438119503221600, -0.027392420918882700,  0.007192235482021060,   0.002963160876719450, -0.002909163707161460,-0.041877891411498200],
          [0.009185811147251570,  0.006546785883021330,  0.008425511473072540,  0.003897095265996940,  0.002554832567327960, -0.001369163682634570,  -0.000723386505863384,  0.000095508653477705,-0.017369968324576700],
          [0.024585585436562200,  0.008425511473072540,  0.030818578707093500, -0.014678165197323700, -0.014910924910161700,  0.008228073619160770,   0.007683225069501030, -0.002233582892853550,-0.038059997041001100],
          [-0.025438119503221600, 0.003897095265996940, -0.014678165197323700,  0.155518507775386000,  0.151164555001805000, -0.057815279141344300,  -0.004301123991898550,  0.004649599528615900, 0.013981354317924200],
          [-0.027392420918882700, 0.002554832567327960, -0.014910924910161700,  0.151164555001805000,  0.324456001631511000, -0.081635399889806100,  -0.002211087030002840,  0.007139118187159160, 0.026655099680892600],
          [0.007192235482021060, -0.001369163682634570,  0.008228073619160770, -0.057815279141344300, -0.081635399889806100,  0.035496658032260300,   0.007746782469778650, -0.001329407316093770,-0.003593660623643860],
          [0.002963160876719450, -0.000723386505863384,  0.007683225069501030, -0.004301123991898550, -0.002211087030002840,  0.007746782469778650,   0.010339733433261000, -0.001343902504603640,-0.003176008379971750],
          [-0.002909163707161460, 0.000095508653477705, -0.002233582892853550,  0.004649599528615900,  0.007139118187159160, -0.001329407316093770,  -0.001343902504603640,  0.001019416244804420, 0.003972578046047490],
          [-0.041877891411498200,-0.017369968324576700, -0.038059997041001100,  0.013981354317924200,  0.026655099680892600, -0.003593660623643860,  -0.003176008379971750,  0.003972578046047490, 0.070355261229252000]]

#    3-sigma area
sigma = 3.0
draws = np.random.multivariate_normal(parameters, matrix, 10) # draw random values using the covariance matrix

class Params():
    pass
params = Params()


def plot_model(color='red'):     
    min_Numbers = []
    max_Numbers = []
    mean_Numbers = []
    for i in xrange(0, len(Lbin)-1):
        Numbers = []
        ll = linspace(Lbin[i], Lbin[i+1], 10)
        for k in arange(0, len(draws[:,0])):        
            mle_Nmodel = []        
            for z in zspace:
                LF_model = []
                for lx in ll:
#               Luminosity Function to use
#               Fotopoulou et al., best fit
                    params.L0 = draws[k,:][0]
                    params.g1 = draws[k,:][1]
                    params.g2 = draws[k,:][2]
                    params.p1 = draws[k,:][3]
                    params.p2 = draws[k,:][4]
                    params.zc = draws[k,:][5]
                    params.La = draws[k,:][6]
                    params.a = draws[k,:][7]
                    params.Norm = draws[k,:][8]
                    PF = lf.Fotopoulou(lx, z, params) 

#               Fold with survey's sensitivity                    
                    area = get_area(get_flux(lx, z))
                    Dv = dif_comoving_Vol(z, area)*3.40367719e-74 # Mpc^3
                    LF_model.append(PF*Dv)                

                integral = simps(LF_model, ll)
                mle_Nmodel.append(integral)
            Numbers.append(simps(mle_Nmodel, zspace))

        min_Numbers.append(np.mean(Numbers) - np.std(Numbers))
        max_Numbers.append(np.mean(Numbers) + np.std(Numbers))
        mean_Numbers.append(np.mean(Numbers))
    plt.fill_between(Lmean, min_Numbers, max_Numbers, color=color, alpha=0.2)
    plt.plot(Lmean, mean_Numbers, color=color)       
    plt.draw()
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


#def plot_metaLF(color='gray'):   
#    from metalf import get_at, get_percentiles
#    N = []
#    for i in xrange(0, len(Lbin)-1):
#        Numbers = []    
#        
#        Lin_data = get_catalog_data(meta_draws, 'col1', Lbin[i], Lbin[i+1])
#        meta_l = Lin_data.field('col1')
#        meta_z = Lin_data.field('col2')   
#
#        for lx, z in zip(meta_l, meta_z):
#            area = get_area(get_flux(lx, z))
#            Dv = dif_comoving_Vol(z, area)*3.40367719e-74 # Mpc^3
#            PF = np.power(10.0, get_percentiles(lx,z))
#            dist = PF*Dv*dz*dL
#            plt.hist(dist)
#            plt.show()
##            
##            
##            Numbers.append(dist)                        
##            hist, bin_edges = np.histogram(dist, 10, normed=True)
##    
##    plt.plot(bin_edges[:-1], hist)
##            plt.show()
    

def plot_metaLF(color='gray'):   
    from metalf import get_at
    min_Numbers = []
    max_Numbers = []
    mean_Numbers = []
    for i in xrange(0, len(Lbin)-1):
        Lin_data = get_catalog_data(meta_draws, 'col1', min(Lbin), max(Lbin))
        meta_l = Lin_data.field('col1')
        meta_z = Lin_data.field('col2')
        Numbers = []
        for k in arange(2, len(column)):
            mle_Nmodel = []        
            for z in meta_z:
                LF_model = []
                for lx in meta_l:
            # Luminosity Function to use
            # meta-analysis     
                    PF = power(10.0, Lin_data.field(k))            
            # Fold with survey's sensitivity                    
                    area = get_area(get_flux(lx, z))
                    Dv = dif_comoving_Vol(z, area)*3.40367719e-74 # Mpc^3
                    LF_model.append(PF*Dv)
                
                integral = simps((LF_model), meta_l)
                mle_Nmodel.append(integral)
                
            Numbers.append(simps(mle_Nmodel, meta_z))
        min_Numbers.append(np.mean(Numbers) - np.std(Numbers))
        max_Numbers.append(np.mean(Numbers) + np.std(Numbers))
        mean_Numbers.append(np.mean(Numbers))
    plt.plot(Lmean, Numbers,color=color)       
    plt.draw()    
#################### Plot number found in AEGIS #########################
#   read survey data for expected number of sources
get_data()
#   load luminosities
path = '/home/sotiria/Documents/Luminosity_Function/data/AEGIS/'
name = 'master_aegis_lum.fits'
filename = path+name
fin = pyfits.open(filename)

def plot_data():
    Ldat = []
    Ndat = []
    for lum in Lbin[:-1]:
        #print lum, lum+0.5
        Lin_data = get_catalog_data(fin[1].data, 'L_2_10', lum, lum+0.5)
        Zin_data = get_catalog_data(Lin_data, 'redshift', min(zspace), max(zspace))
        Prin_data = get_catalog_data(Zin_data, 'HARD_PROB', 1.0e-8)
        Ldat.append(lum+0.25)
        Ndat.append(len(Prin_data))
    print Ldat, Ndat
    plt.errorbar(Ldat, Ndat, yerr = sqrt(Ndat), xerr=0.25, marker='o', color='black', ls=' ',markersize=14)
    #plt.legend(loc=4)
#    plt.xlim([3,4.5])
#    plt.ylim([-7.0,-4.0])
    plt.xlabel('$logLx$', fontsize='x-large')
    plt.ylabel('$Number$', fontsize='x-large')
    plt.draw()
#########################################################################
print 'plotting meta_LF'
plot_metaLF()
print 'plotting model'
plot_model()
print 'plotting data'
plot_data()

for extension in ['.eps', '.jpg', '.pdf', '.png']:
    plt.savefig('/home/sotiria/workspace/Luminosity_Function/output_files/plots/expected_number_AEGIS_high_z'+extension)

#plt.show()
