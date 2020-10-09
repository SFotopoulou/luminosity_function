from scipy import constants
import itertools
import random
""" 
Compute AGN Luminosity Function. 
Selects run conditions:
        0. Cosmological Parameters
        1. Fields
        2. Redshift Type (all, zspec, zphot)
        3. Luminosity Range
        4. X-ray spectrum
            a. power law
        5. Flux Uncertainties (True, False)
        6. Redshift Uncertainties (True, False)
        7. Optical Magnitudes
        8. Luminosity Function Model (PLE, PDE, ILDE, LADE, LDDE)
            

Computational method -- for the moment, separate files
    a. Vmax (Schmidt 1968)
    b. Page and Carrera 
    c. MLE
    d. MCMC
    e. Multinest   
                              
Very distant future TODO list:
    a. Poisson distribution from X-ray counts
    b. Use upper limits in X-ray flux
    c. Include outlier estimation for z-phot
"""
class LF_config:
    #0. Cosmology
    # WMAP Hinshaw+ 2009
    H0 = 70.0 # km/s/Mpc
    Om = 0.3
    Ol = 0.7
    Ok = 0.0#1.0 - Om - Ol
    c = 100.0*constants.c # in cm/s
    
    #1. Fields
    #all_fields = ['MAXI','HBSS', 'COSMOS', 'LH','CDFS','AEGIS']#, 'CDFN']
    #fields = ['MAXI', 'HBSS', 'XMM_COSMOS', 'Chandra_COSMOS','LH',  'AEGIS','XMM_CDFS',  'Chandra_CDFS']  
    #fields = ['XXL_North_field','XXL_South_field']
    #fields = ['XXL_North_projected','XXL_South_projected']
    fields = ['XXL_North','XXL_South']
    
    photozFields = ['XXL_North','XXL_South','COSMOS', 'LH', 'AEGIS','X_CDFS', 'XMM_COSMOS', 'Chandra_COSMOS', 'XMM_CDFS', 'Chandra_CDFS']
    expectation = ['AEGIS']
    depickle_data = False
    #2. Redshift type = ('measured', 'zspec', 'zphot', 'all')
    ztype = 'all'

    zmin = 0.01
    zmax = 4.0
    #zstep = 0.01
    zpoints = 100# 2**9+1#int((zmax - zmin) / zstep) 2^8+1 for romberg
    #===========================================================================
    completeness =  {'XXL_North_projected':0.96,'XXL_South_projected':0.89,'XXL_North_field':0.95,'XXL_South_field':0.89,'MAXI':1, 'HBSS': 1, 'COSMOS': 1, 'LH':1, 'X_CDFS': 1, 'AEGIS':1, 'C_CDFS':1, 'XXL_North':0.95, 'XXL_South':0.85, 'Chandra_COSMOS':1, 'XMM_COSMOS':1, 'Chandra_CDFS':1, 'XMM_CDFS':1}
    scaling = {'XXL_North_projected':1,'XXL_South_projected':1,'XXL_North_field':1,'XXL_South_field':1,'MAXI':1, 'HBSS': 1, 'COSMOS': 1, 'LH':1, 'X_CDFS': 1, 'AEGIS': 1., 'C_CDFS':1, 'XXL_North':1, 'XXL_South':1, 'Chandra_COSMOS':1, 'XMM_COSMOS':1, 'Chandra_CDFS':1, 'XMM_CDFS':1}
    #===========================================================================
    #completeness = {'MAXI':1.478, 'HBSS': 0.6751*0.97, 'COSMOS': 0.6751*0.975, 'LH':1.216*0.988, 'X_CDFS': 0.675*0.95, 'AEGIS':0.373, 'C_CDFS':1}
    #scaling = {'MAXI':1.478, 'HBSS': 0.6751, 'COSMOS': 0.6751, 'LH':1.216, 'X_CDFS': 0.675, 'AEGIS': 0.373, 'C_CDFS':1}

    #power_law  = {'MAXI':1.1, 'HBSS': 1, 'COSMOS': 0.7, 'LH':0.7, 'X_CDFS': 0.7, 'AEGIS': 0.9, 'C_CDFS':0.9}
    #power_law  = {'MAXI':1.1, 'HBSS': 1, 'COSMOS': 0.7, 'LH':1, 'X_CDFS': 0.7, 'AEGIS': 0.4, 'C_CDFS':0.9}
    #completeness = {'MAXI':1, 'HBSS': 0.968, 'COSMOS': 0.975, 'LH':0.988, 'X_CDFS': 0.949, 'AEGIS': 1, 'C_CDFS':1}
    #3. Luminosity Range
    Lmin = 42.0
    Lmax = 47.0
    #Lstep = 0.01
    Lpoints = 100#2**9+1#int((Lmax - Lmin) / Lstep) 2^7+1, for romberg
    
    outpath = '/home/Sotiria/workspace/Luminosity_Function/output_files/'
    inpath = '/home/Sotiria/workspace/Luminosity_Function/input_files/'
    fname = '_input.fits'
    acname = '_acurve.fits'
    #indata = inpath + "temp_above42.fits"
    #incurve = inpath + "test_Acurve.fits"
    #integral_grid_name = inpath+str(Lmin)+'_'+str(Lmax)+'_'+str(zmin)+'_'+str(zmax)+'_'+str(zstep)+'_'+str(Lstep)+'.dat'
    reset_grid = True
    #4. X-ray spectrum
    #    a. power law
    # pl_min = 0.7
    # pl_max = 1.1
    pl = 0.9
    #
    lstyles = itertools.cycle(['-','-','--','-.',':','-','--','-.'])
    mstyles = itertools.cycle(['o','s','v','o','v','s'])
    
    lcolors = itertools.cycle(['gray','black','gray','black','black','gray'])
    mcolors = itertools.cycle(['gray','black','gray','white','black','red'])
    medgecolors = itertools.cycle(['black','black','gray','black','black','red'])
    zorder = itertools.cycle([10, 8, 7, 6, 5, 4, 3, 2, 1])
#    lcolors_c = itertools.cycle(['red','blue','cyan','green','magenta','black'])
#    mcolors_c = itertools.cycle(['red','blue','cyan','green','magenta','black'])
#    medgecolors_c = itertools.cycle(['black','black','black','black','magenta','black'])


    #5. Optical Magnitudes
    Ootmag_min = -100.0
    Ootmag_max = 100.0
   
    model = 'LDDE'
#   6.-7. Redshift and flux uncertainties = (True, False)
    z_unc = False
    L_unc = False # (True needs attention)
    method = 'MLE'
    plot_Acurves = True
    save_Acurves = False
    plot_Lz = True
    save_Lz = False
    Nprocess = 4
##    parameters initial values
    if model == 'ILDE' and z_unc == False:
        parameters = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'Norm']
        p2  =  -0.106861961526
        p1  =  3.13095682996
        g2  =  2.25266047963
        g1  =  0.455177421003
        zc  =  1.54791589224
        L0  =  42.8442812159
        Norm  =  -4.48363348735
        
        L0_pr_min = 41.0
        L0_pr_max = 46.0
        g1_pr_min = -2.0
        g1_pr_max = 5.0
        g2_pr_min =-2.0
        g2_pr_max =  5.0
        p1_pr_min = 0.0
        p1_pr_max = 10.0
        p2_pr_min = -6.0
        p2_pr_max = 2.0
        zc_pr_min = 0.01
        zc_pr_max = 4.0
        Norm_pr_min = -11.0
        Norm_pr_max = -2.0
    
    
    elif model == 'PLE' and z_unc == True:
        parameters = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'Norm']
        p2  =  -0.156796418397
        p1  =  3.15344781082
        g2  =  2.31161845101
        g1  =  0.542347440278
        zc  =  1.54999986851
        L0  =  42.9117915192
        Norm  =  -4.58981774071
    if model == 'halted_PLE' and z_unc == False:
        parameters = ['L0', 'g1', 'g2', 'p1', 'zc', 'Norm']
        p1  =  3.11189430125
        g2  =  2.25442758947
        g1  =  0.453852705185
        zc  =  1.54000312017
        L0  =  42.8462877067
        Norm  =  -4.48183901001
    elif model == 'halted_PLE' and z_unc == True:
        parameters = ['L0', 'g1', 'g2', 'p1', 'zc', 'Norm']
        p1  =  3.12366751551
        g2  =  2.31536677515
        g1  =  0.542077192423
        zc  =  1.53996801304
        L0  =  42.9160531494
        Norm  =  -4.58876680517
                
    if model == 'PDE' and z_unc == False:
        parameters = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'Norm']
        p2  =  -0.172898375678
        p1  =  4.07889853044
        g2  =  2.96958349856
        g1  =  0.903452460019
        zc  =  1.54393076665
        L0  =  44.3551174239
        Norm  =  -6.57596218758
    elif model == 'PDE' and z_unc == True:
        parameters = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'Norm']
        p2  =  -0.188086904963
        p1  =  4.16052288901
        g2  =  2.97363285573
        g1  =  0.908879046521
        zc  =  1.54937811999
        L0  =  44.359669419
        Norm  =  -6.61148524317
    
    if model == 'halted_PDE' and z_unc == False:
        parameters = ['L0', 'g1', 'g2', 'p1', 'zc', 'Norm']
        p1  =  4.07889853044
        g2  =  2.96958349856
        g1  =  0.903452460019
        zc  =  1.54393076665
        L0  =  44.3551174239
        Norm  =  -6.57596218758
    elif model == 'halted_PDE' and z_unc == True:
        parameters = ['L0', 'g1', 'g2', 'p1', 'zc', 'Norm']
        p1  =  4.13422056288
        g2  =  2.97507167301
        g1  =  0.911971300286
        zc  =  1.54001054392
        L0  =  44.3601307454
        Norm  =  -6.61042182309
    
    if model == 'ILDE' and z_unc == False:
        parameters = ['L0', 'g1', 'g2', 'p1', 'p2', 'Norm']
        p2  =  -0.248679721227
        p1  =  2.231440504
        g2  =  2.34414112161
        g1  =  0.481464239531
        L0  =  43.0637067356
        Norm  =  -4.48028580308
    elif model == 'ILDE' and z_unc == True:
        parameters = ['L0', 'g1', 'g2', 'p1', 'p2', 'Norm']
        p2  =  0.296531668038
        p1  =  1.84169859445
        g2  =  2.53150509301
        g1  =  0.647088002807
        L0  =  43.3512321602
        Norm  =  -4.89617988667

    if model == 'halted_ILDE' and z_unc == False:
        parameters = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'Norm']
        p2  =  0.264119979772
        p1  =  2.98993083508
        g2  =  2.28356606468
        g1  =  0.47856240834
        zc  =  1.49699862277
        L0  =  42.9199809385
        Norm  =  -4.5980182939
    elif model == 'halted_ILDE' and z_unc == True:
        parameters = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'Norm']
        p2  =  0.628716676012
        p1  =  1.52500091741
        g2  =  2.67644674933
        g1  =  0.757481390813
        zc = 1.54780531416
        L0  =  43.594878261
        Norm  =  -5.21383823283
                    
    if model == 'LDDE' and z_unc == False:
        parameters = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'La', 'a', 'Norm']
        # no Uncertainties
        p2  =  -1.83567492149
        a  =  0.279999995462
        p1  =  5.66140306003
        g2  =  2.54385277221
        g1  =  1.03781437753
        La  =  44.3242690509+ 0.3098 
        zc  =  2.04146658694
        L0  =  43.9083100727+ 0.3098 
        Norm  =  -6.26751335035
        # with Uncertainties
    elif model == 'LDDE' and z_unc == True:
        parameters = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'La', 'a', 'Norm']
        p2  =  -1.34853227641
        a  =  0.260307788878
        p1  =  5.8361720265
        g2  =  2.55460679566
        g1  =  1.03913647028
        La  =  44.3258637967
        zc  =  1.87730303823
        L0  =  43.9402810729
        Norm  =  -6.34232103471
        
        Prior = {}
        
        Prior['L0'] = (41.0,46.0)
        Prior['g1'] = (-2.0, 5.0)  # g1
        Prior['g2'] = (-2.0, 5.0)  # g2
        Prior['p1'] = (0.0, 10.0) # p1
        Prior['p2'] = (-10.0, 3.0) # p2        
        Prior['zc'] = (0.01, 4.0)  # zc
        Prior['La'] = (41.0, 46.0) # La
        Prior['a'] = (0.0, 1.0) # a
        Prior['Norm'] = (-10.0, -2.0) # Norm
    elif model == 'Fotopoulou2' and z_unc == True:
        parameters = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'La', 'a', 'Norm']
        p2  =  -1.34853227641
        a  =  0.260307788878
        p1  =  5.8361720265
        g2  =  2.55460679566
        g1  =  1.03913647028
        La  =  44.3258637967
        zc  =  1.87730303823
        L0  =  43.9402810729
        Norm  =  -6.34232103471
        
        Prior = {}
        
        Prior['L0'] = (41.0,46.0)
        Prior['g1'] = (-2.0, 5.0)  # g1
        Prior['g2'] = (-2.0, 5.0)  # g2
        Prior['p1'] = (0.0, 10.0) # p1
        Prior['p2'] = (-10.0, 3.0) # p2        
        Prior['zc'] = (0.01, 4.0)  # zc
        Prior['La'] = (41.0, 46.0) # La
        Prior['a'] = (0.0, 1.0) # a
        Prior['Norm'] = (-10.0, -2.0) # Norm
    elif model == 'Fotopoulou3' and z_unc == True:
        parameters = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'La', 'a', 'Norm']
        p2  =  -1.34853227641
        a  =  0.260307788878
        p1  =  5.8361720265
        g2  =  2.55460679566
        g1  =  1.03913647028
        zc  =  1.87730303823
        L0  =  43.9402810729
        Norm  =  -6.34232103471
        
        Prior = {}
        
        Prior['L0'] = (41.0,46.0)
        Prior['g1'] = (-2.0, 5.0)  # g1
        Prior['g2'] = (-2.0, 5.0)  # g2
        Prior['p1'] = (0.0, 10.0) # p1
        Prior['p2'] = (-10.0, 3.0) # p2        
        Prior['zc'] = (0.01, 4.0)  # zc
        Prior['a'] = (0.0, 1.0) # a
        Prior['Norm'] = (-10.0, -2.0) # Norm

    if model == 'halted_LDDE' and z_unc == False:
        parameters = ['L0', 'g1', 'g2', 'p1', 'zc', 'La', 'a', 'Norm']
        # no Uncertainties
        a  =  0.285685957017
        p1  =  5.08790150657
        g2  =  2.55714268647
        g1  =  0.987040375521
        La  =  44.352459698
        zc  =  1.87313180381
        L0  =  43.966304848
        Norm  =  -6.28614016964
        # with Uncertainties
    elif model == 'halted_LDDE' and z_unc == True:
        parameters = ['L0', 'g1', 'g2', 'p1', 'zc', 'La', 'a', 'Norm']
        a  =  0.262278516747
        p1  =  4.86141022474
        g2  =  2.5459487885
        g1  =  0.972308463184
        La  =  44.3398635821
        zc  =  1.79380773276
        L0  =  44.0345169829
        Norm  =  -6.31840821444       
# LADE
    if model == 'LADE' and z_unc == False:
        parameters = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'd', 'Norm']
        p2  =  -1.28427206664
        p1  =  3.66515631729
        g2  =  2.23628006748
        g1  =  0.433037430966
        zc  =  1.79845895102
        L0  =  44.3980128314
        Norm  =  -4.38577159917
        d  =  -0.0340159786876
    
    elif model == 'LADE' and z_unc == True:   
        parameters = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'd', 'Norm']
        p2  =  -1.5822084126
        p1  =  3.54277980655
        g2  =  2.34371019522
        g1  =  0.565474051055
        zc  =  1.74207441536
        L0  =  44.4876482919
        Norm  =  -4.70898171224
        d  =  0.0328997090029     
        
    elif model == 'Ueda14' and z_unc == True:
        parameters = ['L0', 'g1', 'g2', 'p1', 'beta', 'Lp', 'p2', 'p3', 'zc1', 'zc2', 'La1', 'La2', 'a1', 'a2', 'Norm']
        Norm  =  -6.34232103471
        L0  =  43.9402810729
        g2  =  2.55460679566
        g1  =  1.03913647028
        
        p3  =  -6.0
        p2  =  -1.34853227641
        p1  =  5.8361720265
        
        La1  =  44.3258637967
        zc1  =  1.87730303823
        a1  =  0.260307788878

        La2  =  44.3258637967
        zc2  =  1.87730303823
        a2  =  0.260307788878
        
        beta = 0.5
        Lp = 44.0
        
        Prior = {}
        
        Prior['L0'] = (41.0,46.0)  # L0  
        Prior['g1'] = (-2.0, 5.0)  # g1 
        Prior['g2'] = (-2.0, 5.0)  # g2  
        Prior['p1'] = (0.0, 10.0) # p1 
        Prior['p2'] = (-10.0, 3.0) # p2 
        Prior['p3'] = (-10.0, 3.0) # p3                 
        Prior['zc1'] = (0.01, 4.0)  # zc1 
        Prior['La1'] = (41.0, 46.0) # La1 
        Prior['a1'] = (-0.7, 0.7) # a1  
        Prior['zc2'] = (0.01, 4.0)  # zc2 
        Prior['La2'] = (41.0, 46.0) # La2  
        Prior['a2'] = (-0.7, 0.7) # a2 
        Prior['beta'] = (-5.0, 5.0) # beta 
        Prior['Lp'] = (41.0, 46.0) # Lp 
        Prior['Norm'] = (-10.0, -2.0) # Norm         
    elif model == 'Ueda14' and z_unc == False:
        parameters = ['L0', 'g1', 'g2', 'p1', 'beta', 'Lp', 'p2', 'p3', 'zc1', 'zc2', 'La1', 'La2', 'a1', 'a2', 'Norm']
        Norm  =  -6.34232103471
        L0  =  43.9402810729
        g2  =  2.55460679566
        g1  =  1.03913647028
        
        p3  =  -6.0
        p2  =  -1.34853227641
        p1  =  5.8361720265
        
        La1  =  44.3258637967
        zc1  =  1.87730303823
        a1  =  0.260307788878

        La2  =  44.3258637967
        zc2  =  1.87730303823
        a2  =  0.260307788878
        
        beta = 0.5
        Lp = 44.0
        
        Prior = {}
        
        Prior['L0'] = (41.0,46.0)  # L0  
        Prior['g1'] = (-2.0, 5.0)  # g1 
        Prior['g2'] = (-2.0, 5.0)  # g2  
        Prior['p1'] = (0.0, 10.0) # p1 
        Prior['p2'] = (-10.0, 3.0) # p2 
        Prior['p3'] = (-10.0, 3.0) # p3                 
        Prior['zc1'] = (0.01, 4.0)  # zc1 
        Prior['La1'] = (41.0, 46.0) # La1 
        Prior['a1'] = (-0.7, 0.7) # a1  
        Prior['zc2'] = (0.01, 4.0)  # zc2 
        Prior['La2'] = (41.0, 46.0) # La2  
        Prior['a2'] = (-0.7, 0.7) # a2 
        Prior['beta'] = (-5.0, 5.0) # beta 
        Prior['Lp'] = (41.0, 46.0) # Lp 
        Prior['Norm'] = (-10.0, -2.0) # Norm             
    Likelihood_offset = 0.0#921964
    if z_unc == True:
        unc = "withUnc"
    else:
        unc = 'noUnc'
    outname = model+"_"+unc+"_"+''.join( [x[0] for x in fields] )+"_ztype_"+ztype+"_"+str(int(round(zmin,1)))+str(int(round(zmax,1)))+"_"+str(int(Lmin))+str(int(Lmax))+".out"

    data_out_name = outpath+outname+'.fits'
