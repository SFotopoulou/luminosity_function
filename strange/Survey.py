import numpy as np
from scipy import interpolate   
import matplotlib.pyplot as plt
from astropy.io import fits as fits
from AGN_LF_config import LF_config

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
        print(fld)
        # read each file for data
        hdulist_data = fits.open(path_in+fld+LF_config.fname)
        fdata = hdulist_data[1].data
    
        # Read each Area Curves
        hdulist_acurve = fits.open(path_in+fld+LF_config.acname)
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
            
        # Store data
        
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
        # Johannes' advice
        Fname = acurve[ acurve.field(fld+'_FLUX')>0]
        fx = factor * Fname.field(fld+'_FLUX')
        complete = LF_config.completeness[fld] # to correct for redshift completeness
        ar =  complete * Fname.field(fld+'_AREA')*0.00030461742  
##### interpolation
        a_curve["aFlux"+fld] = fx
        a_curve["aArea"+fld] = interpolate.interp1d(fx, ar, bounds_error=False, fill_value=0.0)
 
        if LF_config.plot_Acurves == True: 
            plt.plot(fx, ar,'-',label=fld)

    Fmin = 1e-16
    Fmax = 1e-10
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
            
            area = area + field_area               
        
        total_area.append( area )

    global area_curve
    area_curve = interpolate.interp1d(total_flux, total_area, bounds_error=False, fill_value=0.)
    
    save_area = total_area
    save_flux = total_flux

    if LF_config.plot_Acurves or LF_config.save_Acurves:
        plt.plot(save_flux, save_area,':',color='red')
        plt.legend(loc=2)
        plt.ylabel('area/deg$^{-2}$')
        plt.xlabel('Flux/erg/sec/cm$^{-2}$')
        plt.yscale('log')
        plt.xscale('log')
        
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
    return area_curve(search_flux) # in steradians

if __name__ == "__main__":
    import time
    import AGN_LF_config
    LF_config = AGN_LF_config.LF_config()

    in_flux = np.array([1e-10,1e-15,1e-16])
    get_data()
    start = time.time()

    print(get_area(in_flux))
    print(time.time()-start)
