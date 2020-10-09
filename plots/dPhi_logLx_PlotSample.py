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


def rebin_sum(A, x, y):
    """
    input:  np.histogram2d object = A[0] counts, A[1] rows, A[2] columns
    output: array b(x, y) with new dimensions x<=x_in, y<=y_in
    with elements summed of the merged bin
    """
    b = np.zeros((y, x))
    
    for i in range(0, len(A[1])-1):
        for j in range(0, len(A[2])-1):
            b[i, j] = np.sum( A[0][ np.where( x[i] < A[1] < x[i+1] ) ][ np.where( y[i] < A[2] < y[i+1]) ] )
    return b


path = '/home/Sotiria/workspace/Luminosity_Function/src/LF_plots/forPaper/'
param_path = '/home/Sotiria/workspace/Luminosity_Function/src/MultiNest/safe_keep/'

get_data()
import time
zbin = 4490
Lbin = 500
zlim = 4.5
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
    
#N, yedges, xedges = np.histogram2d(data_logL, data_Z, bins=(Lbin, zbin))

# model
models = ['LDDE', 'LADE', 'ILDE', 'PDE', 'PLE']

for model in models:
    
    fig = plt.figure(figsize=(10,10))
    nullfmt   = NullFormatter()         # no labels
    # definitions for the axes
    left, width = 0.11, 0.65
    bottom, height = 0.11, 0.65
    bottom_h = left_h = left+width
    
    rect_2dhist = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    # start with a rectangular Figure
    plt.figure(1, figsize=(8,8))
    
    ax2Dhist = plt.axes(rect_2dhist)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    print


    
    N = np.loadtxt( path + 'data/Expected_N_'+model+'.dat')
    plt.imshow(N)
    plt.show()

    
    N_binned = rebin_sum(N, zbin, Lbin)

    ax2Dhist.imshow(N_binned, origin='low', extent=[0.01, zlim, 41,46], cmap='bone_r', aspect='auto')
    ax2Dhist.set_xlim( (0.0, zlim) )
    ax2Dhist.set_ylim( (41, 46) )
    
    ax2Dhist.set_ylabel('$\mathrm{log(\,L_x /erg\cdot s^{-1})}$', fontsize='large')
    ax2Dhist.set_xlabel('$\mathrm{log\, z}$', fontsize='large')
    ax2Dhist.text(3.5, 41.5, '$\mathrm{'+model+'}$')
    ax2Dhist.text(4.85, 46.5, '$\mathrm{N='+str(int(np.sum(N)))+'}$')
    
    #zz_test = np.sum(N, axis=0)
    #print np.shape(zz_test), len(ZZ)

    axHistx.plot(ZZ[:-1]+zbin_width, np.sum(N, axis=0), color='k')
    axHistx.hist(data_Z, bins = zbin, histtype='step', color='gray', normed=False)
    axHistx.set_xlim( ax2Dhist.get_xlim() )
    #axHistx.set_ylim((0,1.8))
    axHistx.set_yticks([])
    
    axHisty.plot(np.sum(N,axis=1), LL[:-1]+Lbin_width, color='k')
    axHisty.hist(data_logL, bins = Lbin, orientation='horizontal', histtype='step', color='gray', normed=False)
    axHisty.set_ylim( ax2Dhist.get_ylim() )
    #axHisty.set_xlim((0,0.8))
    axHisty.set_xticks([])
    
    #plt.savefig(path + 'plots/Expected_N_'+model+'.pdf', dpi=300)
    plt.show()
