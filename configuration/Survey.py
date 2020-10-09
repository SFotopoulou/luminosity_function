import sys
# Add the module path to the sys.path list
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
from AGN_LF_config import LF_config
import numpy as np

import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
from scipy import interpolate   

def read_data():
    # Read data
    path_out = LF_config.outpath
    path_in = LF_config.inpath
    a_curve = {}
    Flux_min = []
    Flux_max = []
    
    global data_in 
    data_in = {}
    
    for fld in LF_config.fields:
        print fld
        # read each file for data
        hdulist_data = pyfits.open(path_in+fld+LF_config.fname)
        fdata = hdulist_data[1].data
    
        # Read each Area Curves
        hdulist_acurve = pyfits.open(path_in+fld+LF_config.acname)
        acurve = hdulist_acurve[1].data
    
    
        #       Select data
        if LF_config.ztype == 'measured':
            fielddata = fdata[fdata.field('redshift_flag_'+fld) > 0 ]    
        elif LF_config.ztype == 'zspec':
            fielddata = fdata[fdata.field('redshift_flag_'+fld) == 1 ]
        elif LF_config.ztype == 'zphot':
            fielddata = fdata[fdata.field('redshift_flag_'+fld) == 2 ]
        elif LF_config.ztype == 'all':
            fielddata = fdata
        
        if LF_config.ztype != 'all':
            mask_zmin = fielddata[fielddata.field('redshift_'+fld) >= LF_config.zmin]
            mask_zmax = mask_zmin[mask_zmin.field('redshift_'+fld) <= LF_config.zmax]
        elif LF_config.ztype == 'all':
            mask_zmin = fielddata
            mask_zmax = fielddata
            
        #       Store data
        
        data_in["ID_"+fld] = mask_zmax.field('ID_'+fld)
        # control difference between XMM and chandra: Tsujimoto+2010
        # ACIS can be 5-10% higher than Epic pn.
        # Applied also to the area curve, below
        
        factor = LF_config.scaling[fld]
        
        data_in["F_"+fld] = factor * mask_zmax.field('Flux_'+fld)
        data_in["F_err_"+fld] = factor * mask_zmax.field('e_Flux_'+fld)
        Flux_min.append( min(data_in["F_"+fld]) )
        Flux_max.append( min(data_in["F_"+fld]) )
        
        data_in["Z_"+fld] = mask_zmax.field('redshift_'+fld)
        data_in["Z_flag_"+fld] = mask_zmax.field('redshift_flag_'+fld)
        data_in["Counts_"+fld] = mask_zmax.field('Counts_'+fld)
        data_in["e_Counts_"+fld] = mask_zmax.field('e_Counts_'+fld)
        data_in["mag_"+fld] = mask_zmax.field('mag_'+fld)
#===============================================================================
#        Filtering according to data
#        Fmin = np.min(data_in["F_"+fld])*0.95
#        Fmax = np.max(data_in["F_"+fld])*1.05 # 5% below and above the extreme values
#        
#        Flux_min.append(Fmin)
#        Flux_max.append(Fmax)
#        
#        Fname = acurve[ acurve.field(fld+'_FLUX')>0]
#        fx = Fname.field(fld+'_FLUX')
#        ar = Fname.field(fld+'_AREA')*0.00030461742  
#        
#        in_flux = np.logspace(np.log10(Fmin), np.log10(Fmax), 100)
#        
#        area_out = []
#        for flux in in_flux:
#            
#            for i in np.arange(0,len(fx)-1):
#                if flux > max(fx) or flux < min(fx):
#                    field_area = 0.
#                    break
#                if fx[i] < flux < fx[i+1]:
#                    field_area = ( ar[i] + ar[i+1] ) / 2.
#                    break
#            
#            area_out.append( field_area )
#            
#        a_curve["aFlux"+fld] = in_flux
#        a_curve["aArea"+fld] = area_out
# 
# #            plt.plot(in_flux, area_out, 'ko')
# 
#        if LF_config.plot_Acurves == True: 
#            plt.plot(a_curve["aFlux"+fld],a_curve["aArea"+fld],'-',label=fld)
# 
# 
#    total_flux = np.logspace(np.log10(min(Flux_min)), np.log10(max(Flux_max)), 250)
#===============================================================================
        # Johannes' advice
        Fname = acurve[ acurve.field(fld+'_FLUX')>0]
        fx = factor * Fname.field(fld+'_FLUX')
        complete = LF_config.completeness[fld] # to correct for redshift completeness
        ar =  complete * Fname.field(fld+'_AREA')*0.00030461742  
#        Amin = min(ar)
#        Amax = max(ar)       
#        Fmin = 1e-16
#        Fmax = 1e-8
#
#        in_flux = np.logspace(np.log10(Fmin), np.log10(Fmax), 250)
#        area_out = []
#        for flux in in_flux:
#            if flux >= max(fx): 
#                field_area = Amax
#            elif flux < min(fx):
#                field_area = 0.0
#            else:    
#                for i in np.arange(0,len(fx)-1):                    
#                    if fx[i] <= flux <= fx[i+1]:
#                        A = ( ar[i] + ar[i+1] ) / 2.
#                        field_area = Amax * (A - Amin) / (Amax - Amin)
#                        
#                        break
#                    
#            area_out.append( field_area )
#            
#        a_curve["aFlux"+fld] = in_flux
#        a_curve["aArea"+fld] = area_out

##### Try interpolation
        #print fx, ar
        a_curve["aFlux"+fld] = fx
        a_curve["aArea"+fld] = interpolate.interp1d(fx, ar, bounds_error=False, fill_value=0.0)

#        print min(area_out), max(area_out) 
#        plt.plot(in_flux, area_out, 'ko')
 
        if LF_config.plot_Acurves == True: 
            plt.plot(fx, ar,'-',label=fld)
#            plt.plot(a_curve["aFlux"+fld],a_curve["aArea"+fld],'-',label=fld)
               
#                    
#    Fmin = 6e-18# min( Flux_min )
#    Fmax = 1e-7#max( Flux_max )
#    total_flux = np.logspace(np.log10(Fmin), np.log10(Fmax), 300)
#    print total_flux[0], total_flux[1]
###########################################################################################
#    total_area = []
#    
#    for flux in total_flux:
#        area = 0.0
#        
#        for fld in LF_config.fields:
#            
#            fx = a_curve["aFlux"+fld]
#            ar = a_curve["aArea"+fld]
#            
#            if flux >= max(fx) :
#                field_area = ar(max(fx))
#            elif flux < min(fx):
#                field_area = 0.0
#            else:
#                field_area = ar(flux)
##                for i in np.arange(0,len(fx)-1):
##                    if fx[i] <= flux <= fx[i+1]:
##                        field_area = ( ar[i] + ar[i+1] ) / 2.
##                        break
#            
#            area = area + field_area               
#        
#        total_area.append( area )
#
#    global area_curve
#    area_curve = interpolate.interp1d(total_flux, total_area)
#    
#    save_area = total_area
#    save_flux = total_flux
#    
#    save_total = np.column_stack([save_flux, save_area])
#    np.savetxt(path_out+"results/Total_a_curve.txt", save_total)    
#
#
#    if LF_config.plot_Acurves or LF_config.save_Acurves:
#        plt.plot(save_flux, save_area,'-')
#        plt.yscale('log')
#        plt.xscale('log')
        
        
        
    Fmin = 1e-16# min( Flux_min )
    Fmax = 1e-10#max( Flux_max )
    total_flux = np.logspace(np.log10(Fmin), np.log10(Fmax), 10000)

##########################################################################################
    total_area = []
    
    for flux in total_flux:
        area = 0.0
        
        for fld in LF_config.fields:
            
            fx = a_curve["aFlux"+fld]
            ar = a_curve["aArea"+fld]
            
            if flux >= max(fx) :
                field_area = ar(max(fx))
            elif flux < min(fx):
                field_area = 0.0
            else:
                field_area = ar(flux)
                #for i in np.arange(0,len(fx)-1):
                #    if fx[i] <= flux <= fx[i+1]:
                #        field_area = ( ar[i] + ar[i+1] ) / 2.
                #        break
            
            area = area + field_area               
        
        total_area.append( area )

    global area_curve
    area_curve = interpolate.interp1d(total_flux, total_area, bounds_error=False, fill_value=0.)
    
    save_area = total_area
    save_flux = total_flux
    # print total_flux[0], total_flux[1]


    if LF_config.plot_Acurves or LF_config.save_Acurves:
        plt.plot(save_flux, save_area,':',color='red')
        plt.legend(loc=2)
        plt.ylabel('area/deg$^{-2}$')
        plt.xlabel('Flux/erg/sec/cm$^{-2}$')
        plt.yscale('log')
        plt.xscale('log')
        
    #plt.show()            
    if LF_config.save_Acurves: 
        plt.draw()
        for ext in ['pdf','eps','jpg']:
            plt.savefig(path_out+'plots/acurve.'+ext)
        save_total = np.column_stack([save_flux, save_area])
        np.savetxt(LF_config.inpath+"Total_acurve.txt", save_total)    
            
    if LF_config.plot_Acurves:
        plt.show()
        plt.close()
        
    return data_in
    
def get_data():
    global data_in
    try:
        return data_in
    except NameError:
        data_in = read_data()
        return data_in
  
def get_area(search_flux):
    global area_curve
    #print search_flux[1]/search_flux[0]
       
    return area_curve(search_flux) # in steradians
#
if __name__ == "__main__":
    import time

    in_flux = np.array([1e-10,1e-15,1e-16])
    get_data()
    start = time.time()

    print get_area(in_flux)
    print time.time()-start

#    Fmaxi = data_in['F_MAXI']
#    Amaxi = get_area(Fmaxi)
#    plt.plot(Fmaxi,Amaxi,'ko')
#    plt.show()