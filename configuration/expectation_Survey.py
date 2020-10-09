import sys
# Add the module path to the sys.path list
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
from AGN_LF_config import LF_config
import numpy as np
import matplotlib.pyplot as plt
import pyfits
   

def read_data():
    # Read data
    path_in = LF_config.inpath
    path_out = LF_config.outpath
    hdulist_data = pyfits.open(path_in+"catalogs/temp_above42.fits")
    fdata = hdulist_data[1].data

    global data_in 
    data_in = {}

    # Read Area Curves
    hdulist_acurve = pyfits.open(path_in+"area_curves/test_Acurve.fits")
    acurve = hdulist_acurve[1].data
    a_curve = {}
    Flux_min = []
    Flux_max = []
    
    for fld in LF_config.expectation:

        #       Select data
        if LF_config.ztype == 'all':
            fielddata = fdata[fdata.field('redshift_flag_'+fld) > 0 ]    
        elif LF_config.ztype == 'zspec':
            fielddata = fdata[fdata.field('redshift_flag_'+fld) == 1 ]
        elif LF_config.ztype == 'zphot':
            fielddata = fdata[fdata.field('redshift_flag_'+fld) == 2 ]
            
        mask_zmin = fielddata[fielddata.field('redshift_'+fld) >= LF_config.zmin]
        mask_zmax = mask_zmin[mask_zmin.field('redshift_'+fld) <= LF_config.zmax]
        
        #       Store data
        data_in["ID_"+fld] = mask_zmax.field('ID_'+fld)
        data_in["F_"+fld] = mask_zmax.field('Flux_'+fld)
        data_in["F_err_"+fld] = mask_zmax.field('e_Flux_'+fld)
        data_in["Z_"+fld] = mask_zmax.field('redshift_'+fld)
        data_in["Z_flag_"+fld] = mask_zmax.field('redshift_flag_'+fld)
        
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
        fx = Fname.field(fld+'_FLUX')
        ar = Fname.field(fld+'_AREA')*0.00030461742  
        Amin = min(ar)
        Amax = max(ar)
        
        Fmin = 5e-17
        Fmax = 1e-8

        in_flux = np.logspace(np.log10(Fmin), np.log10(Fmax), 250)
        area_out = []
        for flux in in_flux:
            if flux >= max(fx): 
                field_area = Amax
            elif flux < min(fx):
                field_area = 0.0
            else:    
                for i in np.arange(0,len(fx)-1):                    
                    if fx[i] <= flux <= fx[i+1]:
                        A = ( ar[i] + ar[i+1] ) / 2.
                        field_area = Amax * (A - Amin) / (Amax - Amin)
                        break
                    
            area_out.append( field_area )
            
        a_curve["aFlux"+fld] = in_flux
        a_curve["aArea"+fld] = area_out
#        print min(area_out), max(area_out) 
#        plt.plot(in_flux, area_out, 'ko')
 
        if LF_config.plot_Acurves == True: 
            plt.plot(a_curve["aFlux"+fld],a_curve["aArea"+fld],'-',label=fld)
                      
                    
    Fmin = 1e-17
    Fmax = 1e-8
    total_flux = np.logspace(np.log10(Fmin), np.log10(Fmax), 300)
##########################################################################################
    total_area = []
    
    for flux in total_flux:
        area = 0.0
        
        for fld in LF_config.expectation:
            
            fx = a_curve["aFlux"+fld]
            ar = a_curve["aArea"+fld]
            
            if flux >= max(fx) :
                field_area = max(ar)
            elif flux < min(fx):
                field_area = 0.0
            else:
                for i in np.arange(0,len(fx)-1):
                    if fx[i] <= flux <= fx[i+1]:
                        field_area = ( ar[i] + ar[i+1] ) / 2.
                        break
            
            area = area + field_area               
        
        total_area.append( area )

    global save_area, save_flux
    save_area = total_area
    save_flux = total_flux
    
    
    save_total = np.column_stack([save_flux, save_area])
    np.savetxt(path_out+"results/Expectation_a_curve.txt", save_total)    

    if LF_config.plot_Acurves or LF_config.save_Acurves:
        plt.plot(save_flux, save_area,'-')
        plt.yscale('log')
        plt.xscale('log')
        
    if LF_config.save_Acurves: 
        plt.draw()
        for ext in ['pdf','eps','jpg']:
            plt.savefig(path_out+'plots/expectation_acurve.'+ext)
            
    if LF_config.plot_Acurves:
        plt.show()
        
    return data_in
    
def get_data():
    global data_in
    try:
        return data_in
    except NameError:
        data_in = read_data()
        return data_in
  
def get_area(search_flux):
    global save_flux, save_area
        
    fx = save_flux
    ar = save_area
    
    if isinstance(search_flux, float):
        if search_flux >= max(fx):
            return max(ar)
        elif search_flux < min(fx):
            return 0.0
        else:
            for i in np.arange(0,len(fx)-1):            
                if fx[i] <= search_flux <= fx[i+1]:
                    return ( ar[i] + ar[i+1] ) / 2.

    else:
        
        sorted = np.argsort(search_flux)
        original = np.argsort(sorted)
        
        area_out = []
        start = 0
        
        for flux in search_flux[sorted]:
            
            if flux >= max(fx):
                area_out.append(max(ar))
            elif flux < min(fx):
                area_out.append(0.0)
            else:    
                for i in np.arange(start,len(fx)-1):
                    if fx[i] <= flux <= fx[i+1]:
                        area_out.append(( ar[i] + ar[i+1] ) / 2.)
                        start = i
                        break

        area = np.array(area_out)
        return area[original] # in steradians
#
if __name__ == "__main__":

    in_flux = [1e-10,1e-15,1e-16]
    get_data()
    #print s.return_area(in_flux)
#    Fmaxi = data_in['F_MAXI']
#    Amaxi = get_area(Fmaxi)
#    plt.plot(Fmaxi,Amaxi,'ko')
#    plt.show()
