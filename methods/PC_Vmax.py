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
from AGN_LF_config import LF_config
from cosmology import *
from Survey import *
from Source import *
from SetUp_data import Set_up_data
from LFunctions import *
#import astroML as ML


setup_data = Set_up_data()
data = setup_data.get_data()[1]

def histo2d(Z, L, zbins):
#    Z, L data
#    zbins: 1-d array
    H = []
    Summary = []
    Ledges = []
    for i in range( 0, len(zbins)-1 ):
        
        zbin_count = 0
        z_temp = []
        L_temp = []
        
        for redshift, luminosity in zip(Z,L):
            if zbins[i] <= redshift < zbins[i+1] :
                zbin_count += 1
                z_temp.append( redshift )
                L_temp.append( luminosity )
        if zbin_count == 0 :
            z_temp = [0]
            L_temp = [0]
        
        if min(L_temp) >0 and max(L_temp)>0:
            print zbin_count 
            nbins = 10
            Lbins = np.linspace(min(L_temp, LF_config.Lmin), max(L_temp), nbins)
            Ledges.append(Lbins)
            
            Counter = []
            for j in range( 0, len(Lbins)-1 ):
                Lbin_count = 0
                for redshift, luminosity in zip(z_temp,L_temp):
                    if Lbins[j] <= luminosity < Lbins[j+1] :
                        Lbin_count += 1
                Counter.append(Lbin_count)
            H.append(Counter)   
        else:
            H.append( [0]*nbins )   
            Ledges.append(np.linspace(42,46,nbins))

        Summary.append( [zbin_count, zbins[i], zbins[i+1], min(z_temp), max(z_temp), Ledges, min(L_temp), max(L_temp)] )
#    print Summary
#    ofile = open( LF_config.outpath+'results/Vmax_summary.txt', 'w')
#    ofile.write(str(Summary))
    return H, Ledges  

def calc_Vol(Lmin, Lmax, zmin, zmax, zpoints=25, Lpoints=25):
    LL = np.array([np.ones( (zpoints), float )*item for item in 
                   np.linspace(Lmin, Lmax, Lpoints)])
    # make LL 1D
    L = LL.ravel()
    # repeat as many times as Lpoints
    Z = np.tile(np.logspace(np.log10(zmin), np.log10(zmax), zpoints), Lpoints) 

# Set up grid for survey integral
#    The grid includes the area curve, gives the survey efficiency
    vecFlux = np.vectorize(get_flux)
    temp_Fx = vecFlux(L, Z)
    area = get_area(temp_Fx)   
   
    vecDifVol = np.vectorize(dif_comoving_Vol) 
    DVc = np.where( area>0, vecDifVol(Z, area), 0) 
    DVcA = DVc*3.4036771e-74 # vol in Mpc^3, per unit area

    # save to file
    #integr = zip(L, Z, DVcA, temp_Fx, area)
    
#    np.savetxt(LF_config.inpath+'area_curves/PC_Vmax_integral_grid.dat', integr)    
        
    Redshift_int = Z[0:zpoints]
    Luminosity_int = np.linspace(Lmin, Lmax, Lpoints)
    
    y = []
    
    count_r = xrange(0, Lpoints)
    for count in count_r:
        startz = count * zpoints
        endz = startz + zpoints
        x = DVcA[startz:endz]
        
        int1 = simps(x, Redshift_int, even='last')
        y.append(int1)
    
    DV_int = simps(y, Luminosity_int, even='last')
    return DV_int

def bin_middle(array):
    middle = []
    for i in range(0, len(array)-1):
        middle.append( (array[i]+array[i+1])/2.0 )
    return np.array(middle)

L = []
Z = []
for field in LF_config.fields:
    L.extend(data['Lum_'+field])
    Z.extend(data['Z_'+field])

L = np.array(L)
Z = np.array(Z)

#zedges = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0,  3.0, 4.0]
zedges = [0.01, 0.5, 1.0]

H, Ledges = histo2d(Z, L, zedges) 

H = np.array(H)

L_middle = []
for i in range(0, len(Ledges)):
    L_middle.append(bin_middle(Ledges[i]))

z_middle = bin_middle(zedges)

dPhi = []
for i in range(0, len(zedges)-1):
    Vol = []
    Ledge = Ledges[i]
    for j in range(0, len(Ledge)-1):
        print i,j
        if 0 < H[i][j] < 10:
            Vol.append( H[i][j] / calc_Vol(Ledge[j], Ledge[j+1], zedges[i], zedges[i+1], zpoints = 50, Lpoints=50 ))
        elif H[i][j] >= 10 :
            Vol.append( H[i][j] / calc_Vol(Ledge[j], Ledge[j+1], zedges[i], zedges[i+1], zpoints = 50, Lpoints=50 ) )
        else: 
            Vol.append(0.)
    dPhi.append(Vol)
dPhi = np.array(dPhi)

class Params: pass
params = Params()

params.L0 = LF_config.L0
params.g1 = LF_config.g1
params.g2 = LF_config.g2
params.p1 = LF_config.p1
params.Norm = LF_config.Norm
if LF_config.model == 'LDDE' or LF_config.model == 'halted_LDDE':
    params.La = LF_config.La
    params.a = LF_config.a
if LF_config.model == 'LADE':
    params.d = LF_config.d
if LF_config.model != 'ILDE':
    params.zc = LF_config.zc
if LF_config.model not in ['halted_PLE', 'halted_LDDE', 'halted_PDE', 'halted_LADE']:
    params.p2 = LF_config.p2

def Phi(Lx, z, params):    
    """ 
    The luminosity function model 
    """
    if LF_config.model == 'PLE':
        return PLE(Lx, z, params)
    
    elif LF_config.model == 'halted_PLE':
        return halted_PLE(Lx, z, params)
    
    elif LF_config.model == 'PDE':
        return PDE(Lx, z, params)                          
    
    elif LF_config.model == 'halted_PDE':
        return halted_PDE(Lx, z, params)
    
    elif LF_config.model == 'ILDE':
        return ILDE(Lx, z, params)
    
    elif LF_config.model == 'halted_ILDE':
        return halted_ILDE(Lx, z, params)
    
    elif LF_config.model == 'Hasinger':
        return Hasinger(Lx, z, params)
    
    elif LF_config.model == 'Ueda':
        return Ueda(Lx, z, params)
    
    elif LF_config.model == 'LADE':
        return LADE(Lx, z, params)                          
    
    elif LF_config.model == 'halted_LADE':
        return halted_LADE(Lx, z, params)
    
    elif LF_config.model == 'Miyaji':
        return Miyaji(Lx, z, params)
    
    elif LF_config.model == 'LDDE':
        return Fotopoulou(Lx, z, params)

    elif LF_config.model == 'halted_LDDE':
        return halted_Fotopoulou(Lx, z, params)
 
#############################################3
color = itertools.cycle(['olive', 'black', 'magenta', 'cyan', 'green', 'red', 'blue'])
for i in range(0, len(dPhi[:,0])):
    clr = color.next()
    plt.plot(L_middle[i], dPhi[i,:], '-o', label=str(z_middle[i]), color=clr )
    phi = Phi(L_middle[i], z_middle[i], params)
    plt.plot(L_middle[i], phi, color=clr)
plt.yscale('log')
plt.xscale('linear')
plt.ylim([1e-9, 5e-3])
plt.xlim([42.0, 46.0])
plt.legend()
#plt.savefig(LF_config.outpath+"/plots/"+LF_config.outname+'.pdf')
plt.show()
