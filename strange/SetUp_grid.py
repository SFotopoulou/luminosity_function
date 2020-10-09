import numpy as np
from cosmology import dif_comoving_Vol
from Source import get_flux
from Survey import get_area
from AGN_LF_config import LF_config

def set_up_grid():
    Lpoints = LF_config.Lpoints
    zpoints = LF_config.zpoints
    path_in = LF_config.inpath

    if LF_config.reset_grid == False:
    
        L, Z, DVcA, temp_Fx, area  = np.genfromtxt(path_in+'input_integral_grid.dat', unpack=True)
    
    else:
        
        print( 'Oh dear. Generating data file 1, be patient...')   
        Lspace = np.linspace(LF_config.Lmin, LF_config.Lmax, Lpoints)
        LL = np.array([np.ones( (zpoints), float )*item for item in Lspace])    
        L = LL.ravel()

        # repeat as many times as Lpoints
        Z = np.tile(np.logspace(np.log10(LF_config.zmin), np.log10(LF_config.zmax), zpoints), Lpoints) 
        vecFlux = np.vectorize(get_flux)
        temp_Fx = vecFlux(L, Z, LF_config.pl)
        area = get_area(temp_Fx)   
        vecDifVol = np.vectorize(dif_comoving_Vol) 
        DVc = vecDifVol(Z, area) 
        DVcA = DVc*3.4036771e-74 # vol in Mpc^3, per unit area
        
        integr = [L, Z, DVcA, temp_Fx, area]
        
        # save to file
        np.savetxt(path_in+'input_integral_grid.dat', integr)    
        
    Redshift_int = Z[0:LF_config.zpoints]
    Luminosity_int = np.linspace(LF_config.Lmin, LF_config.Lmax, Lpoints)
    
#    print "grid ok"
    return L, Z, DVcA, temp_Fx, area, Redshift_int, Luminosity_int
    
if __name__ == "__main__":
    import AGN_LF_config
    LF_config = AGN_LF_config.LF_config()
    set_up_grid(LF_config)
