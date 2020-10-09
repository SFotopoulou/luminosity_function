#!/home/Sotiria/Software/anaconda3/bin/python
    
def run_Vmax(LF_config):
    import os
    import sys
    import numpy as np
    from scipy.integrate import simps
    import matplotlib.pyplot as plt
    
    sys.path.append('/home/Sotiria/Dropbox/AGNinCL/src')
    sys.path.append('/home/Sotiria/Dropbox/AGNinCL/plots/scripts')
    from SetUp_data import Set_up_data
    from Source import get_flux, get_area
    from cosmology import dif_comoving_Vol
    from Vmax_plots import plot_Vmax, plot_Vmaxdensity
    
    ##
    path = LF_config.outpath + "Vmax/"
    if not os.path.exists(path): os.mkdir(path)
    print('results in: ', path)
    
    setup_data = Set_up_data()
    data = setup_data.get_data()[1]
    
    def define_grid(Z, L, zedges, Ledges, show=True, save=True):
        plt.plot(Z, L, 'k.')
        for l in Ledges: plt.axhline(l,linestyle='--',color='gray')
        for z in zedges: plt.axvline(z,linestyle='--',color='gray')
        plt.xlabel('redshift')
        plt.ylabel('log(Lx / erg/s)')
        plt.draw()
        if save==True: plt.savefig(path+'Lz_grid.png') 
        if show==True: plt.show()
        return
    
     
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
        print(zmin, Lmin)
        vecDifVol = np.vectorize(dif_comoving_Vol) 
        DVc = np.where( area>0, vecDifVol(Z, area), 0) 
        DVcA = DVc*3.4036771e-74 # vol in Mpc^3
    
        Redshift_int = Z[0:zpoints]
        Luminosity_int = np.linspace(Lmin, Lmax, Lpoints)
        
        y = []
        
        count_r = range(0, Lpoints)
        for count in count_r:
            startz = count * zpoints
            endz = startz + zpoints
            x = DVcA[startz:endz]
            int1 = simps(x, Redshift_int, even='last')
            y.append(int1)
        
        DV_int = simps(y, Luminosity_int, even='last')
        return DV_int
    
    def Vmax_Phi(Z, L, zedges, Ledges, path):
    #    input: L, Z (data), zedges, Ledges
    #    output: N_bin, Lmean_bin, zmean_bin, Phi in file
        Vmax_filename = path+'Vmax_dPhi.dat'
        Phi_Points = open(Vmax_filename, 'w') 
        Phi_Points.write('# zBinmin zBinmax zmin zmax zmean zmedian LBinmin LBinmax Lmin Lmax Lmean Lmedian N dPhi dPhi_err\n')  
    
        N_bin = []
        zmean_bin = []
        Lmean_bin = []
        zmedian_bin = []
        Lmedian_bin = []
        
        for i in range(0, len(Ledges)-1):
            for j in range( 0, len(zedges)-1 ):
                
                N_count = 0
                z_temp = []
                L_temp = []
                
                for redshift, luminosity in zip(Z,L):
                    if Ledges[i] <= luminosity < Ledges[i+1] and zedges[j] <= redshift < zedges[j+1] :
                        N_count += 1
                        z_temp.append( redshift )
                        L_temp.append( luminosity )
    
                if N_count == 0 :
                    z_temp = [0]
                    L_temp = [0]
                
                N_bin.append(N_count)
                zmean_bin.append(np.mean(z_temp)) 
                Lmean_bin.append(np.mean(L_temp))
                zmedian_bin.append(np.median(z_temp)) 
                Lmedian_bin.append(np.median(L_temp))
                
                if N_count>0:
                    Vol = calc_Vol(Ledges[i], Ledges[i+1], zedges[j], zedges[j+1], zpoints = 50, Lpoints=50 )
                    Phi = N_count / Vol
                    err= np.sqrt(N_count)/ Vol
                    dPhi_err = 0.434*err/ Phi
                    dPhi = np.log10(Phi)
                else: 
                    dPhi =0.
                    dPhi_err = 0.
                #
                Phi_Points.write( str( round(zedges[j],5) ) + ' ' +str( round( zedges[j+1],5)) + ' ' +str( round( np.min(z_temp) ,5)) + ' ' +str( round( np.max(z_temp) ,5)) + ' ' +str( round( np.mean(z_temp) ,5)) + ' ' +str( round( np.median(z_temp) ,5)) + ' ' + str( round( Ledges[i] ,5)) + ' ' + str( round( Ledges[i+1] ,5)) + ' ' +str( round( np.min(L_temp) ,5)) + ' ' +str( round( np.max(L_temp) ,5)) + ' ' + str( round( np.mean(L_temp) ,5)) + ' ' + str( round( np.median(L_temp) ,5)) + ' ' +str( round(N_count,5) ) + ' ' +str( round(dPhi,5) ) + ' ' +str( round(dPhi_err,5)) + '\n' )
        
        Phi_Points.close()
                   
        return Vmax_filename
    
       
    def Vmax_Ndensity(Z, L, zbin, Lbin, path):
        Vmaxdensity_filename = path+'Vmax_Ndensity.dat' 
        Density_Points = open(Vmaxdensity_filename, 'w') 
        Density_Points.write('# zBinmin zBinmax zmin zmax zmean zmedian LBinmin LBinmax Lmin Lmax Lmean Lmedian N logNdensity logNdensity_err \n' )
        
        for j in np.arange(0,len(Lbin)-1):    
            for i in np.arange(0,len(zbin)-1):
                print( zbin[i] )
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
                    Density_Points.write( str( round(zbin[i],5)) + ' '+str( round(zbin[i+1],5)) + ' '+str( round(np.min(Zz),5)) + ' ' + str( round(np.max(Zz),5)) + ' ' + str( round(np.mean(Zz),5)) + ' ' + str( round(np.median(Zz),5)) + ' ' + str( round(Lbin[j],5)) + ' ' + str( round(Lbin[j+1],5)) + ' '+ str( round(np.min(Ll),5)) + ' ' + str( round(np.max(Ll),5)) + ' ' +str( round(np.mean(Ll),5)) + ' ' +str( round(np.median(Ll),5)) + ' ' +str( round(count,5)) + ' ' + str( round(np.log10(dN_dV),5)) + ' ' + str( round(0.434*err_dN_dV/dN_dV,5)) + '\n' )
        Density_Points.close()
        return Vmaxdensity_filename
        
     
    #zedges = np.logspace(np.log10(LF_config.zmin), np.log10(LF_config.zmax), 10, 6)
    #Ledges = np.logspace(np.log10(LF_config.Lmin), np.log10(LF_config.Lmax), 10, 6)
      
    #zedges = np.linspace(LF_config.zmin, 2, 5)
    #Ledges = np.linspace(LF_config.Lmin, LF_config.Lmax, 12)  
      
    Z = []
    L = []
    for field in LF_config.fields:
        Z.extend(data['Z_'+field])
        L.extend(data['Lum_'+field])
        
    Z = np.array(Z)
    L = np.array(L)
    
    
    # dPhi/dlogLx
    zedges = [0.01, 0.5, 1.0, 1.5, 2.0]
    Ledges = [42.0, 42.5, 43.0, 43.5, 44.0, 44.5, 45.0, 45.5, 46.0]
    define_grid(Z, L, zedges, Ledges)
    Vmax_filename = Vmax_Phi(Z, L, zedges, Ledges, path)
    plot_Vmax(Vmax_filename)
    
    #Ndensity
    zedges = [0.01, 0.2, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0]
    Ledges = [42.0, 43.0, 44.0, 45.0, 46.0]
    Vmaxdensity_filename = Vmax_Ndensity(Z, L, zedges, Ledges, path)
    plot_Vmaxdensity(Vmaxdensity_filename)
    
if __name__ == "__main__":
    import AGN_LF_config
    LF_config = AGN_LF_config.LF_config()
    run_Vmax(LF_config)
