import sys
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
from AGN_LF_config import LF_config
from cosmology import *
from Source import *
from Survey import *
import numpy as np

Lpoints = LF_config.Lpoints
zpoints = LF_config.zpoints
path_in = LF_config.inpath

def set_up_grid():
    #print "I am set_up_grid!"
    if LF_config.reset_grid == False:
    
        L, Z, DVcA, temp_Fx, area  = np.genfromtxt(path_in+'area_curves/input_integral_grid.dat', unpack=True)
    
    else:
        
        print 'Oh dear. Generating data file 1, be patient...'   
        
#        LL = np.array([np.ones( (zpoints), float )*item for item in 
#                       np.linspace(LF_config.Lmin, LF_config.Lmax, Lpoints)])
        #Lspace = np.logspace(np.log10(LF_config.Lmin), np.log10(LF_config.Lmax), Lpoints)
        Lspace = np.linspace(LF_config.Lmin, LF_config.Lmax, Lpoints)

        LL = np.array([np.ones( (zpoints), float )*item for item in Lspace])    
  
        # make LL 1D
        L = LL.ravel()

        # repeat as many times as Lpoints
        Z = np.tile(np.logspace(np.log10(LF_config.zmin), np.log10(LF_config.zmax), zpoints), Lpoints) 
        #Z = np.tile(np.linspace((LF_config.zmin), (LF_config.zmax), zpoints), Lpoints)
    # Set up grid for survey integral
    #    The grid includes the area curve, gives the survey efficiency
        vecFlux = np.vectorize(get_flux)
        temp_Fx = vecFlux(L, Z)
#        print temp_Fx[0], temp_Fx[1]
#        print min(temp_Fx), max(temp_Fx)
        area = get_area(temp_Fx)   
#        plt.plot(temp_Fx, area, 'o')
#        plt.show()
        vecDifVol = np.vectorize(dif_comoving_Vol) 
        DVc = vecDifVol(Z, area) 
        DVcA = DVc*3.4036771e-74 # vol in Mpc^3, per unit area
        
        integr = zip(L, Z, DVcA, temp_Fx, area)
        
        # save to file
        np.savetxt(path_in+'input_integral_grid.dat', integr)    
        
    Redshift_int = Z[0:LF_config.zpoints]
    Luminosity_int = np.linspace(LF_config.Lmin, LF_config.Lmax, Lpoints)
    
#    print "grid ok"
    return L, Z, DVcA, temp_Fx, area, Redshift_int, Luminosity_int
    
if __name__ == "__main__":
    import timeit
    print(timeit.timeit( "set_up_grid()", setup="from __main__ import set_up_grid" , number=1))    