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
        nbins = 10

        if min(L_temp) >0 and max(L_temp)>0:
            print zbin_count 
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
            Ledges.append(np.linspace(41,46,nbins))

        Summary.append( [zbin_count, zbins[i], zbins[i+1], min(z_temp), max(z_temp), Ledges, min(L_temp), max(L_temp)] )
#    print Summary
    ofile = open('/home/Sotiria/workspace/Luminosity_Function/src/LF_plots/XXL_full/data/Vmax_summary.txt', 'w')
    ofile.write('# zbin_count, zbin_low, zbin_high, z_min, z_max, Ledges, L_min, L_max\n')
    ofile.write(str(Summary))
    ofile.close()
    return H, Ledges  

def calc_Vol(Lmin, Lmax, zmin, zmax, zpoints=25, Lpoints=25):
    LL = np.array([np.ones( (zpoints), float )*item for item in 
                   np.linspace(Lmin, Lmax, Lpoints)])
    # make LL 1D
    L = LL.ravel()
    # repeat as many times as Lpoints
    Z = np.tile(np.logspace(np.log10(zmin), np.log10(zmax), zpoints), Lpoints) 

# Set up grid for survey integral
    vecFlux = np.vectorize(get_flux)
    temp_Fx = vecFlux(L, Z)
    area = get_area(temp_Fx)   
   
    vecDifVol = np.vectorize(dif_comoving_Vol) 
    DVc = np.where( area>0, vecDifVol(Z, area), 0) 
    DVcA = DVc*3.4036771e-74 # vol in Mpc^3

        
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

def Vmax_Phi():
    
    #zedges = [0.01, 0.2, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 3.0, 4.0]
    #zedges = [0.01, 0.5, 1.0,  2.0,  3.0, 4.0]
    #zedges = [0.01, 0.5, 0.75, 1.0]
    zedges = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0 , 1.2, 1.5, 1.8, 2.1, 2.7, 3.2, 4.0, 7.0]
    H, Ledges = histo2d(Z, L, zedges) 
    
    H = np.array(H)
    
    L_middle = []
    for i in range(0, len(Ledges)):
        L_middle.append(bin_middle(Ledges[i]))
    
    Phi_Points = open('/home/Sotiria/workspace/Luminosity_Function/src/LF_plots/XXL_full/data/Vmax_dPhi_literature.dat', 'w') 
    Phi_Points.write('# min zmax zmean Lmin Lmax Lmean N dPhi dPhi_err\n')
    
    for i in range(0, len(zedges)-1):
        Ledge = Ledges[i]
        for j in range(0, len(Ledge)-1):
            print i,j
            Vol = calc_Vol(Ledge[j], Ledge[j+1], zedges[i], zedges[i+1], zpoints = 50, Lpoints=50 )
            if 0 < H[i][j] < 10:
                Phi = H[i][j] / Vol
                err = np.sqrt(H[i][j])/ Vol
                dPhi_err = 0.434*err/ Phi
                dPhi = np.log10(Phi)
                
            elif H[i][j] >= 10 :
                Phi = H[i][j] / Vol
                err= np.sqrt(H[i][j])/ Vol
                dPhi_err = 0.434*err/ Phi
                dPhi = np.log10(Phi)
                
            else: 
                dPhi =0.
                dPhi_err = 0.
            
            Phi_Points.write( str( zedges[i] ) + ' ' +str( zedges[i+1]) + ' ' +str( (zedges[i]+zedges[i+1])/2.0 ) + ' ' + str( Ledge[j] ) +' ' + str( Ledge[j+1] ) +' ' + str( (Ledge[j]+Ledge[j+1])/2.0 ) + ' ' +str(H[i][j]) +' ' +str(dPhi) + ' ' +str(dPhi_err) + '\n' )
    
    Phi_Points.close()
    
    
def Vmax_Ndensity():
    zbin = [0.01, 0.2, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 3.0, 4.0]
    Lbin = [41.0, 42.0, 43.0, 44.0, 45.0, 46.0]

    Density_Points = open('/home/Sotiria/workspace/Luminosity_Function/src/LF_plots/XXL_full/data/Vmax_Ndensity.dat', 'w') 
    Density_Points.write('# zbin_min zbin_max zmin zmax zmedian Lbin_min Lbin_max Lmin Lmax Lmedian N logNdensity logNdensity_err \n' )
    
    for j in np.arange(0,len(Lbin)-1):    
        print Lbin[j]
        dL = Lbin[j+1] - Lbin[j]
        for i in np.arange(0,len(zbin)-1):
            print zbin[i]
            count = 0
            dN_dV = 0.0
            err_dN_dV = 0.0
            Ll = []
            Zz = []
            count = 0
            for Lx, z in zip(L, Z):
                if zbin[i] <= z < zbin[i+1] and Lbin[j] <= Lx < Lbin[j+1]:
                    count = count + 1
                    Ll.append(Lx)        
                    Zz.append(z)
            if count>0:
                    
                Vmax =  calc_Vol(Lbin[j], Lbin[j+1], zbin[i], zbin[i+1], zpoints = 50, Lpoints=50) 
                dN_dV = count/Vmax
                err_dN_dV = np.sqrt(count) /Vmax
                #print np.median(Zz), np.median(Ll), count, np.log10(dN_dV), 0.434*err_dN_dV/dN_dV, np.median(Ll)-Lbin[j], Lbin[j+1]-np.median(Ll)
                Density_Points.write( str(zbin[i]) +' '+str(zbin[i+1]) +' '+str(np.min(Zz)) + ' ' + str(np.max(Zz)) + ' ' + str(np.median(Zz)) + ' ' + str(Lbin[j]) + ' ' + str(Lbin[j+1]) +' '+ str(np.min(Ll)) + ' ' + str(np.max(Ll)) + ' ' +str(np.median(Ll)) + ' ' +str(count) + ' ' + str(np.log10(dN_dV)) + ' ' + str(0.434*err_dN_dV/dN_dV) + '\n' )
    Density_Points.close()
    
L = []
Z = []
for field in LF_config.fields:
    L.extend(data['Lum_'+field])
    Z.extend(data['Z_'+field])

L = np.array(L)
Z = np.array(Z)

Vmax_Phi()
Vmax_Ndensity()



