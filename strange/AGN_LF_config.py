#!/usr/bin/python
import os
""" 
Compute AGN Luminosity Function. 
"""
class LF_config:
    # restore computation?
    reset_grid = True # volume computed over L-z range
    depickle_data = False # source class objects for last fields used
    
    # model set up
    model = 'LDDE'
    
    # X-ray spectrum
    pl = 0.7
    
    # Optical Magnitudes
    Ootmag_min = -100.0
    Ootmag_max = 100.0

    # Redshift type = ('measured', 'zspec', 'zphot', 'all')
    ztype = 'all'
    zmin = 0.01
    zmax = 4.0
    zpoints = 50
        
    # Luminosity Range
    Lmin = 40.0
    Lmax = 47.0
    Lpoints = 50
    
    z_unc = True
    L_unc=False
    if z_unc == True:
        unc = "withUnc"
    else:
        unc = 'noUnc'

    Likelihood_offset = 0.0
   
    #
    Nprocess = 5

    # Fields
    outname = "test"
    outpath = '/home/Sotiria/git/luminosity_function/output_files/'+outname + '/'
    if not os.path.exists(outpath): os.mkdir(outpath)
    inpath = '/home/Sotiria/git/luminosity_function/input_files/'
    fname = '_input.fits'
    acname = '_acurve.fits'
    
    fields = ['MAXI']
    #fields=['XXL_North_XLF_1_field','XXL_South_XLF_1_field'] # in folder: ../input_files
    photozFields = ['XXL_North_projected', 'XXL_South_projected','XXL_North_XLF_1_field','XXL_South_XLF_1_field']
                    
    #===========================================================================
    completeness =  {'XXL_North_XLF_1_projected':1.0,
                     'XXL_South_XLF_1_projected':1.0,
                     'XXL_North_XLF_1_field':1.0,
                     'XXL_South_XLF_1_field':1.0,
                     'MAXI':1.0, 
                     'HBSS': 1.0, 
                     'COSMOS': 1.0, 
                     'LH':1.0, 
                     'X_CDFS': 1.0, 
                     'AEGIS':1.0, 
                     'C_CDFS':1.0, 
                     'XXL_North':1.0, 
                     'XXL_South':1.0, 
                     'Chandra_COSMOS':1.0, 
                     'XMM_COSMOS':1.0, 
                     'Chandra_CDFS':1.0, 
                     'XMM_CDFS':1.0}
    
    scaling = {'XXL_North_XLF_1_projected':1.0,
               'XXL_South_XLF_1_projected':1.0,
               'XXL_North_XLF_1_field':1.0,
               'XXL_South_XLF_1_field':1.0,
               'MAXI':1.0, 
               'HBSS': 1.0, 
               'COSMOS': 1.0, 
               'LH':1.0, 
               'X_CDFS': 1.0, 
               'AEGIS': 1.0, 
               'C_CDFS':1.0, 
               'XXL_North':1.0, 
               'XXL_South':1.0, 
               'Chandra_COSMOS':1.0, 
               'XMM_COSMOS':1.0, 
               'Chandra_CDFS':1.0, 
               'XMM_CDFS':1.0}
    #===========================================================================
 
    #===========================================================================
    # TODO convert to JSON
    L0 = 43.0        
    g1 = -1.0
    g2 = 2.0
    p1 = 5.0        
    p2 = -5.0        
    zc = 1.5
    La = 44.0
    a = 0.3
    Norm = -4.0
 
    plot_Acurves = False
    save_Acurves = False
    plot_Lz = False
    save_Lz = False
    save_data = False

if __name__ == "__main__":
    LF_config()
    
    
    
