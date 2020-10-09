import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/Bayesian/Multinest/Multinest_modules/')
#
import time
import math   
from numpy import array,log10,savetxt,loadtxt,log,sqrt,linspace,vectorize,ones,tile
from Source import Source
from parameters import Parameters
from LFunctions import Models
from scipy.integrate import simps
import matplotlib.pyplot as plt
from make_PDFz import Spectrum
from multiprocessing import Process, Queue

params = Parameters()
Lmin, Lmax = Parameters.L(params)
zmin, zmax = Parameters.z(params)

def PDFgauss(x, mu, sigma):
    s2 = sigma*sigma
    t2 = (x-mu)*(x-mu)
    return (math.exp(-t2/(2.0*s2)))/math.sqrt(2.0*math.pi*s2)

vPDFgauss = vectorize(PDFgauss)

model = Models()

def Phi(l,z,L0,g1,g2,p1,p2,zc,La,a):    
    """ 
    The luminosity function model 
    """
    model.Ueda_init(l,z,L0,g1,g2,p1,p2,zc,La,a)
    Phi = model.Phi_UedaLDDE()
    return Phi

vPhi = vectorize(Phi)
#
def Survey_prob(L0,g1,g2,p1,p2,zc,La,a):
    """
        Survey detection probability Phi*dV*Area_curve
    """
    y = []
    Luminosity = []
    for count in range(0, Lpoints):
        
        startz = count*zpoints
        endz = (count + 1)*zpoints
        
        Lum = L[startz:endz]
        redshift = Z[startz:endz]
        V = DVcA[startz:endz].ravel()      
        
        phi = array(vPhi(Lum,redshift,L0,g1,g2,p1,p2,zc,La,a))
        x = V*phi 
        
        int1 = simps(x,redshift, even='last')
        Luminosity.append(Lum[0])
        
        y.append(int1)
        
    int2 = simps(y,Luminosity, even='last')
    return int2

def Source_prob(L0,g1,g2,p1,p2,zc,La,a):
    """Source detection probability phi*dVc*Source_prob"""
    sum = 0.
    phi = array(vPhi(array(L_data),array(z_data),L0,g1,g2,p1,p2,zc,La,a))
    V = array(DV)
    x = list(V*phi)
    for item in x:
        sum = sum + log(item)         
    return sum

def Normalization(L0,g1,g2,p1,p2,zc,La,a):
    int = Survey_prob(L0,g1,g2,p1,p2,zc,La,a)
    count = 0.
    for l,z in zip(L_data,z_data):
        if Lmin<=l<=Lmax and zmin<=z<=zmax:
            count = count + 1.
    A = count/int
    dA = 1.7*A/sqrt(count)
    return A, dA


def Miyaji_Likelihood(L0,g1,g2,p1,p2,zc,La,a):
    """
    Likelihood for MLE L = -2.0*ln(L)
    """
    survey_d = Survey_prob(L0,g1,g2,p1,p2,zc,La,a)
    source_d = Source_prob(L0,g1,g2,p1,p2,zc,La,a)
    
    L = 2.0*log(survey_d)*n - 2.0*source_d
    return L

start_time = time.time()
#
############### Observations ###############
# Prepare individual grid from each datum ##
############################################

### Set up grid for data
### L sample
Lpoints = 40 # no need to go too thin
### z sample
zpoints = 400 # governed by the photoz step
##
LL = array([ones( (zpoints), float )*item for item in linspace(Lmin,Lmax,Lpoints)])
L = LL.ravel() #    make LL 1D
Z = tile(linspace(zmin, zmax, zpoints), Lpoints) # repeat as many times as Lpoints

d = Source('data')
s = Spectrum()
e_zspec = 0.01
ID, Fx, e_Fx, z, z_flag, field = d.get_data()
L_data, z_data = d.get_luminosity(Fx, e_Fx, z)
DV = d.Dz_area(L_data, z_data)

n = len(ID)

# Set up grid for integral
g = Source('grid')
int_grid = g.Dz_area(L,Z)
DVcA = int_grid

###########################################################################################    
cube=[ 44.5, 1.3, 2.97,4.59, -2.52, 1.8, 43.8, 0.45]  
    # L0  # g1  # g3  # p1 # p2 # zc # La  # a


import pymultinest

def UniformPrior(r,x1,x2):
    return x1+r*(x2-x1)

def LogUniformPrior(r,x1,x2):
    if (r<=0.0):
        LogPrior=-1.0e32
    else:
        lx1=log10(x1)
        lx2=log10(x2)
        LogPrior=10.**(lx1+r*(lx2-lx1))
    return LogPrior
    
def myprior(cube, ndim, nparams):
    cube[0] = LogUniformPrior(cube[0],42,46) #L0
    cube[1] = UniformPrior(cube[1],0,10)  # g1
    cube[2] = UniformPrior(cube[2],0,10)  # g2
    cube[3] = UniformPrior(cube[3],2,6) # p1
    cube[4] = UniformPrior(cube[4],-3,0) # p2
    cube[5] = UniformPrior(cube[5],1.,2)  # zc
    cube[6] = LogUniformPrior(cube[6],42,46) # La
    cube[7] = UniformPrior(cube[7],0.1,0.5) # a
    
    
def myloglike(cube, ndim, nparams):
    LogL = Miyaji_Likelihood(cube[0],cube[1],cube[2],cube[3],cube[4],cube[5],cube[6],cube[7])
    return LogL

n_params = 8
ndim = len(cube)
#progress = pymultinest.ProgressWatcher(n_params = n_params,
#                                       outputfiles_basename = "output_files/1-")
#progress.start()

#pymultinest.run(myloglike,
#                myprior,
#                n_dims = n_params,
#                n_params = n_params,
#                n_clustering_params = None, 
#                wrapped_params = None, 
#                multimodal = True, 
#                const_efficiency_mode = False, 
#                n_live_points = 1000,
#                evidence_tolerance = 0.5, 
#                n_iter_before_update = 100,
#                null_log_evidence = -1e90,
#                max_modes = 100,
#                seed = -1,
#                context = 0,
#                outputfiles_basename = "output_files/1-", 
#                resume = True,
#                verbose = False,
#                sampling_efficiency = 'model')

a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename = "output_files/1-")
s = a.get_stats()

#progress.running = False
import json
json.dump(s, file('%s.json' % a.outputfiles_basename, 'w'), indent=2)
print
print "Global Evidence:\n\t%.15e +- %.15e" % ( s['global evidence'], s['global evidence error'] )

plt.clf()

labels = ["L0", "g1","g2","p1","p2","zc","La","a","N"]

p = pymultinest.PlotMarginal(a)
for i in range(n_params):
    outfile = '%s-marginal-%d.pdf' % (a.outputfiles_basename,i)
    p.plot_conditional(i, None, with_ellipses = True, with_points = False, use_log_values=True)
    plt.xlabel(labels[i])
    plt.savefig(outfile, format='pdf', bbox_inches='tight')
    plt.close()
    
    outfile = '%s-mode-marginal-%d.pdf' % (a.outputfiles_basename,i)
    p.plot_modes_marginal(i, with_ellipses = False, with_points = True)
    plt.xlabel(labels[i])
    plt.savefig(outfile, format='pdf', bbox_inches='tight')
    plt.close()
    
    outfile = '%s-mode-marginal-cumulative-%d.pdf' % (a.outputfiles_basename,i)
    p.plot_modes_marginal(i, cumulative = True, with_ellipses = True, with_points = False)
    
    plt.savefig(outfile, format='pdf', bbox_inches='tight')
    plt.close()
    
    for j in range(i):
        print i,j 
        p.plot_conditional(i, j, with_ellipses = False, with_points = True, only_interpolate=False, use_log_values=True)
        outfile = '%s-conditional-interpolate-%d-%d.pdf' % (a.outputfiles_basename,i,j)
        plt.savefig(outfile, format='pdf', bbox_inches='tight')
        plt.close()

        p.plot_conditional(i, j, with_ellipses = True, with_points = False)
        outfile = '%s-conditional-%d-%d.pdf' % (a.outputfiles_basename,i,j)
        plt.savefig(outfile, format='pdf', bbox_inches='tight')
        plt.close()