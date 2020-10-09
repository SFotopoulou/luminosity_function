import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/models')
import iminuit as minuit
import astropy
import numpy as np
from AGN_LF_config import LF_config
from cosmology import *
from Survey import *
from Source import *
from LFunctions import *
import Likelihood as lk
#from SetUp_grid import set_up_grid
#from SetUp_data import set_up_data 
from scipy.integrate import simps
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
import time

#print "1.4 -> 1.9:", 1./ ((0.1/0.6) * ( ( 10**0.6 - 5**0.6 )/( 10**0.1 - 5**0.1 ) ))
#print "1.7 -> 1.9:", 1./ ((0.1/0.3) * ( ( 10**0.3 - 5**0.3 )/( 10**0.1 - 5**0.1 ) ))
#print "2.0 -> 1.9:", (10**0.1 - 5**0.1)/(0.1*np.log(10/5) )
#print "2.1 -> 1.9:", 1./ ((0.1/(-0.1)) * ( ( 10**(-0.1) - 5**(-0.1) )/( 10**0.1 - 5**0.1 ) ))

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    start_time = time.time()
    ############### Observations ###############
    # Prepare individual grid from each datum ##
    ############################################
    if LF_config.model == 'PLE':
        m = minuit.Minuit(lk.PLE_Likelihood)
        m.values["L0"] = LF_config.L0
        m.values["g1"] = LF_config.g1
        m.values["g2"] = LF_config.g2
        m.values["p1"] = LF_config.p1
        m.values["p2"] = LF_config.p2
        m.values["zc"] = LF_config.zc
        m.values["Norm"] = LF_config.Norm
        keys = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'Norm']
    elif LF_config.model == 'halted_PLE':
        m = minuit.Minuit(lk.halted_PLE_Likelihood)
        m.values["L0"] = LF_config.L0
        m.values["g1"] = LF_config.g1
        m.values["g2"] = LF_config.g2
        m.values["p1"] = LF_config.p1
        m.values["zc"] = LF_config.zc
        m.values["Norm"] = LF_config.Norm
        keys = ['L0', 'g1', 'g2', 'p1', 'zc', 'Norm']    
    elif LF_config.model == 'PDE':
        m = minuit.Minuit(lk.PDE_Likelihood)
        m.values["L0"] = LF_config.L0
        m.values["g1"] = LF_config.g1
        m.values["g2"] = LF_config.g2
        m.values["p1"] = LF_config.p1
        m.values["p2"] = LF_config.p2
        m.values["zc"] = LF_config.zc
        m.values["Norm"] = LF_config.Norm
        keys = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'Norm']
    elif LF_config.model == 'halted_PDE':
        m = minuit.Minuit(lk.halted_PDE_Likelihood)
        m.values["L0"] = LF_config.L0
        m.values["g1"] = LF_config.g1
        m.values["g2"] = LF_config.g2
        m.values["p1"] = LF_config.p1
        m.values["zc"] = LF_config.zc
        m.values["Norm"] = LF_config.Norm
        keys = ['L0', 'g1', 'g2', 'p1', 'zc', 'Norm']
    elif LF_config.model == 'ILDE':
        m = minuit.Minuit(lk.ILDE_Likelihood)
        m.values["L0"] = LF_config.L0
        m.values["g1"] = LF_config.g1
        m.values["g2"] = LF_config.g2
        m.values["p1"] = LF_config.p1
        m.values["p2"] = LF_config.p2
        m.values["Norm"] = LF_config.Norm
        keys = ['L0', 'g1', 'g2', 'p1', 'p2', 'Norm']
    elif LF_config.model == 'halted_ILDE':
        m = minuit.Minuit(lk.halted_ILDE_Likelihood)
        m.values["L0"] = LF_config.L0
        m.values["g1"] = LF_config.g1
        m.values["g2"] = LF_config.g2
        m.values["p1"] = LF_config.p1
        m.values["p2"] = LF_config.p2
        m.values["zc"] = LF_config.zc
        m.values["Norm"] = LF_config.Norm
        keys = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'Norm']
    elif LF_config.model == 'Hasinger':
        m = minuit.Minuit(lk.Hasinger_Likelihood)
        m.values["L0"] = LF_config.L0
        m.values["g1"] = LF_config.g1
        m.values["g2"] = LF_config.g2
        m.values["p1"] = LF_config.p1
        m.values["p2"] = LF_config.p2
        m.values["zc"] = LF_config.zc
        m.values["La"] = LF_config.La
        m.values["a"] = LF_config.a
        m.values["b1"] = LF_config.b1
        m.values["b2"] = LF_config.b2
        m.values["Norm"] = LF_config.Norm
        keys = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'La', 'a', 'b1', 'b2', 'Norm']
    elif LF_config.model == 'Ueda':
        m = minuit.Minuit(lk.Ueda_Likelihood)
        m.values["L0"] = LF_config.L0
        m.values["g1"] = LF_config.g1
        m.values["g2"] = LF_config.g2
        m.values["p1"] = LF_config.p1
        m.values["p2"] = LF_config.p2
        m.values["zc"] = LF_config.zc
        m.values["La"] = LF_config.La
        m.values["a"] = LF_config.a
        m.values["Norm"] = LF_config.Norm
        keys = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'La', 'a', 'Norm']
    elif LF_config.model == 'LADE':
        m = minuit.Minuit(lk.LADE_Likelihood)
        m.values["L0"] = LF_config.L0
        m.values["g1"] = LF_config.g1
        m.values["g2"] = LF_config.g2
        m.values["p1"] = LF_config.p1
        m.values["p2"] = LF_config.p2
        m.values["zc"] = LF_config.zc
        m.values["d"] = LF_config.d
        m.values["Norm"] = LF_config.Norm
        keys = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'd', 'Norm']
    elif LF_config.model == 'halted_LADE':
        m = minuit.Minuit(lk.halted_LADE_Likelihood)
        m.values["L0"] = LF_config.L0
        m.values["g1"] = LF_config.g1
        m.values["g2"] = LF_config.g2
        m.values["p1"] = LF_config.p1
        m.values["p2"] = LF_config.p2
        m.values["zc"] = LF_config.zc
        m.values["d"] = LF_config.d
        m.values["Norm"] = LF_config.Norm
        keys = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'd', 'Norm']
    elif LF_config.model == 'Miyaji':
        m = minuit.Minuit(lk.Miyaji_Likelihood)
        m.values["L0"] = LF_config.L0
        m.values["g1"] = LF_config.g1
        m.values["g2"] = LF_config.g2
        m.values["p1"] = LF_config.p1
        m.values["p2"] = LF_config.p2
        m.values["zc"] = LF_config.zc
        m.values["La"] = LF_config.La
        m.values["a"] = LF_config.a
        m.values["pmin"] = LF_config.pmin
        m.values["Norm"] = LF_config.Norm
        keys = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'La', 'a', 'pmin', 'Norm']
    elif LF_config.model == 'LDDE':
        print 'LDDE'
        m = minuit.Minuit(lk.Fotopoulou_Likelihood,        
                            limit_L0 = (41.0,46.0),
                            limit_g1 = (-2.0, 5.0), 
                            limit_g2 = (-2.0, 5.0),
                            limit_p1 = (0.0, 10.0), 
                            limit_p2 = (-10.0, 3.0),        
                            limit_zc = (0.01, 4.0), 
                            limit_La = (41.0, 46.0),
                            limit_a = (0.0, 1.0),
                            limit_Norm = (-10.0, -2.0))
        
        m.values["L0"] = LF_config.L0
        m.values["g1"] = LF_config.g1
        m.values["g2"] = LF_config.g2
        m.values["p1"] = LF_config.p1
        m.values["p2"] = LF_config.p2
        m.values["zc"] = LF_config.zc
        m.values["La"] = LF_config.La
        m.values["a"] = LF_config.a
        m.values["Norm"] = LF_config.Norm
        keys = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'La', 'a', 'Norm']    
    elif LF_config.model == 'Fotopoulou2':
        print 'Fotopoulou2'
        m = minuit.Minuit(lk.Fotopoulou2_Likelihood)
        m.values["L0"] = LF_config.L0
        m.values["g1"] = LF_config.g1
        m.values["g2"] = LF_config.g2
        m.values["p1"] = LF_config.p1
        m.values["p2"] = LF_config.p2
        m.values["zc"] = LF_config.zc
        m.values["La"] = LF_config.La
        m.values["a"] = LF_config.a
        m.values["Norm"] = LF_config.Norm
        keys = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'La', 'a', 'Norm']    
    elif LF_config.model == 'Fotopoulou3':
        print 'Fotopoulou3'
        m = minuit.Minuit(lk.Fotopoulou3_Likelihood)
        m.values["L0"] = LF_config.L0
        m.values["g1"] = LF_config.g1
        m.values["g2"] = LF_config.g2
        m.values["p1"] = LF_config.p1
        m.values["p2"] = LF_config.p2
        m.values["zc"] = LF_config.zc
        m.values["a"] = LF_config.a
        m.values["Norm"] = LF_config.Norm
        keys = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'a', 'Norm']    

    elif LF_config.model == 'halted_LDDE':
        m = minuit.Minuit(lk.halted_Fotopoulou_Likelihood)
        m.values["L0"] = LF_config.L0
        m.values["g1"] = LF_config.g1
        m.values["g2"] = LF_config.g2
        m.values["p1"] = LF_config.p1
        m.values["zc"] = LF_config.zc
        m.values["La"] = LF_config.La
        m.values["a"] = LF_config.a
        m.values["Norm"] = LF_config.Norm
        keys = ['L0', 'g1', 'g2', 'p1', 'zc', 'La', 'a', 'Norm']
    elif LF_config.model == 'Ueda14':
        print 'Ueda14'
        m = minuit.Minuit(lk.Ueda14_Likelihood)
        m.values["L0"] = LF_config.L0 
        m.values["g1"] = LF_config.g1 
        m.values["g2"] = LF_config.g2 
        m.values["p1"] = LF_config.p1 
        m.values["beta"] = LF_config.beta 
        #m.values["Lp"] = 44.0 #LF_config.Lp
        #m.values["p2"] = -1.5 #LF_config.p2
        #m.values["p3"] = -6.2 #LF_config.p3
        m.values["zc1"] = LF_config.zc1
        m.values["La1"] = LF_config.La1
        m.values["a1"] = LF_config.a1 
        #m.values["zc2"] = 3.0 #LF_config.zc2
        #m.values["La2"] = 44.0 #LF_config.La2
        #m.values["a2"] = LF_config.a2
        m.values["Norm"] = LF_config.Norm
        #keys = ['L0', 'g1', 'g2', 'p1', 'beta', 'Lp', 'p2', 'p3', 'zc1', 'La1', 'a1', 'zc2', 'La2', 'a2', 'Norm']   
        keys = ['L0', 'g1', 'g2', 'p1', 'beta', 'zc1', 'La1', 'a1', 'a2', 'Norm']    
    print LF_config.outname
    #    Accuracy
    
    
    m.print_level = 1
    m.strategy = 1.0
    #m.up = 1.0
    #m.tol = 100
    #m.maxcalls = 1000
    Nparameters = len(m.values)
    Ndata = lk.Ndata()
    
    field_names = ', '.join( [x for x in LF_config.fields])
    
    print "MLE, using "+LF_config.model+" model and fields: "+field_names
    print "Nparams:", Nparameters
    print "Ndata:", Ndata
    print str( LF_config.zmin ) + "<= z <=" + str( LF_config.zmax )
    print str( LF_config.Lmin ) + "<= L <=" + str( LF_config.Lmax )
    if LF_config.z_unc == True:
        print "including uncertainties"
    else:
        print "not including uncertainties"
    print "------------------------------"
    m.migrad(resume=False)
    print "MLE converged"
    print "Time lapsed = ",round(time.time()-start_time,2),"sec"
    print "------------------------------"
    print "Number of function calls: ",m.ncalls
    print "Vertical distance to minimum: ", m.edm
    print "Best fit paramerers:\n"
    for key, value in m.values.items():
        print key," = ",value
    
    
    output = open(LF_config.outpath+LF_config.outname,'w')
    
    output.write("MLE, using "+LF_config.model+" model and fields: "+field_names+"\n")
    output.write( "Nparams: "+str(Nparameters)+"\n")
    output.write( "Ndata: "+str(Ndata)+"\n")
    output.write( str( LF_config.zmin ) + "<=z<=" + str( LF_config.zmax )+"\n")
    output.write( str( LF_config.Lmin ) + "<=L<=" + str( LF_config.Lmax )+"\n")
    if LF_config.z_unc == True:
        output.write( "including uncertainties\n")
    else:
        output.write( "not including uncertainties\n")
    output.write( "------------------------------------------------------------------\n")
    output.write( "Least likelihood value = "+str(m.fval-LF_config.Likelihood_offset)+"\n")
    output.write( "with parameters :\n")
    for key, value in m.values.items():
        output.write(str(key)+" = "+str(value)+"\n")
    output.write( "------------------------------------------------------------------\n")
    output.write( "\n")
    output.write( "Minimum distance between function calls: "+str(m.edm)+"\n")
    output.write( "Vertical distance to minimum : "+str(m.edm)+"\n")
    output.write( "\n")
    output.write( "AIC="+str(2.*Nparameters+m.fval-LF_config.Likelihood_offset)+"\n")
    output.write( "AICc="+str(2.*Nparameters+m.fval-LF_config.Likelihood_offset+(2.*Nparameters*(Nparameters+1.))/(Ndata-Nparameters-1.))+"\n")
    output.write( "BIC="+str(m.fval-LF_config.Likelihood_offset+Nparameters*np.log10(Ndata))+"\n")
    
    #    Calculate errors and covariance matrix
    print "calling Hesse"
    m.hesse()
    output.write( "------------------------------------------------------------------\n")
    output.write( "\n\nLinear Errors\n" )
    for key, value in m.errors.items():
        output.write(str(key)+" = "+str(value)+"\n")
    output.write("\n")
    output.write( str(m.errors) )
    output.write( "\n\nCovariance Matrix\n" )
    output.write( str(m.covariance) )
    
    print np.array(m.matrix())
    output.write( "\n\nCorrelation Matrix\n" )
    output.write( str(m.matrix(correlation=True)) )
    print "Done in ", round(time.time()-start_time, 2), "sec"
    
    output.write( "\n\nCovariance Matrix\n" )
    output.write(np.array(m.matrix()))
    
    
    
    # Add the module path to the sys.path list
    sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
    sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/models')
    #
    import time, itertools
    from numpy import arange,savetxt, linspace,logspace,sqrt,log10,array,ones,tile, median, asarray, genfromtxt, power
    from Source import *
    import scipy.integrate
    from LFunctions import *
    from scipy.integrate import simps
    from cosmology import *
    
    zmin = LF_config.zmin
    zmax = LF_config.zmax
    Lmin = LF_config.Lmin
    Lmax = LF_config.Lmax
    
    parameters = np.array([m.values[key] for key in keys])
    
    matrix = np.array( m.matrix() )
    
    #    3-sigma area
    sigma = 3.0
    draws = np.random.multivariate_normal(parameters,matrix, 200) # draw random values using the covariance matrix
    
    # Observations
    from SetUp_data import Set_up_data
    from LFunctions import *
    
    setup_data = Set_up_data()
    data = setup_data.get_data()[0]
       
    Lbin_cycle = itertools.cycle([[41.17, 42.0, 42.5, 43.0, 43.25, 43.50, 44.0, 44.65],
                                  linspace(42.0, 44.60, 7),
                                  linspace(42.3, 45.0, 6),
                                  linspace(42.65, 45.00, 6),
                                  linspace(42.75, 45.20, 6),
                                  linspace(42.8, 45.2,6),
                                  linspace(43.0, 44.9,6),
                                  linspace(43.0, 45.0, 5),
                                  linspace(43.0, 45.0, 5)])        
    zbin = [0.01, 0.2, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 3.0, 4.0]
                   
    linecolor = (0., 0., 0.)
    fillcolor = (0.75, 0.75, 0.75)
    pointcolor = (0., 0., 0.)
    zlabel = []
    save_data = []
    ll = linspace(41.0,46.0)
    # LF at z=0
    class Params():
        pass
    
    params = Params()

    if LF_config.model == 'PLE':
        
        params.L0 = m.values["L0"]
        params.g1 = m.values["g1"]
        params.g2 = m.values["g2"]
        params.p1 = m.values["p1"]
        params.p2 = m.values["p2"]
        params.zc = m.values["zc"]
        params.Norm = m.values["Norm"]
   
    if LF_config.model == 'LDDE':
    
        params.L0 = m.values['L0']
        params.g1 = m.values['g1']
        params.g2 = m.values['g2']
        params.p1 = m.values['p1']
        params.p2 = m.values['p2']
        params.zc = m.values['zc']
        params.La = m.values['La'] 
        params.a = m.values['a']
        params.Norm = m.values['Norm']

    if LF_config.model == 'Fotopoulou2':
    
        params.L0 = m.values['L0']
        params.g1 = m.values['g1']
        params.g2 = m.values['g2']
        params.p1 = m.values['p1']
        params.p2 = m.values['p2']
        params.zc = m.values['zc']
        params.La = m.values['La'] 
        params.a = m.values['a']
        params.Norm = m.values['Norm']

    if LF_config.model == 'Fotopoulou3':
    
        params.L0 = m.values['L0']
        params.g1 = m.values['g1']
        params.g2 = m.values['g2']
        params.p1 = m.values['p1']
        params.p2 = m.values['p2']
        params.zc = m.values['zc']
        params.a = m.values['a']
        params.Norm = m.values['Norm']
              
            
    if LF_config.model == 'LADE':
        
        params.L0 = m.values["L0"]
        params.g1 = m.values["g1"]
        params.g2 = m.values["g2"]
        params.p1 = m.values["p1"]
        params.p2 = m.values["p2"]
        params.zc = m.values["zc"]
        params.d = m.values["d"]
        params.Norm = m.values["Norm"]        
        
            
    LF0 = []
    if LF_config.model == 'LDDE':

        for lx in ll:
            PF = Fotopoulou(lx, 0.0, params)
            LF0.append(log10(PF))
    if LF_config.model == 'Fotopoulou2':

        for lx in ll:
            PF = Fotopoulou2(lx, 0.0, params)
            LF0.append(log10(PF))

    if LF_config.model == 'Fotopoulou3':

        for lx in ll:
            PF = Fotopoulou3(lx, 0.0, params)
            LF0.append(log10(PF))

    if LF_config.model == 'PLE':

        for lx in ll:
            PF = PLE(lx, 0.0, params)
            LF0.append(log10(PF))
    if LF_config.model == 'LADE':

        for lx in ll:
            PF = LADE(lx, 0.0, params)
            LF0.append(log10(PF))
        
    redshift_bin = itertools.cycle([1.040e-01,  3.449e-01, 6.278e-01, 8.455e-01, 1.161e+00, 1.465e+00, 1.799e+00, 2.421e+00, 3.376e+00])
    fig_size = [15, 15]
    fig = plt.figure(figsize=fig_size)
    fig.subplots_adjust(left=0.10,  right=0.95, bottom=0.10, top=0.99, wspace=0.0, hspace=0.0)
    
    for i in arange(0,len(zbin)-1):
        LF = []
        print i
        Lbin = Lbin_cycle.next()            
    #    Find median redshift
        Zz = []
        for source in data:
            z = source.z
            if zbin[i] <= z < zbin[i+1]:
                Zz.append(z)    
    
        dz = (zbin[i+1]-zbin[i])/0.01
        zspace = linspace(zbin[i], zbin[i+1], dz)
        
        for j in arange(0,len(Lbin)-1):        
    #    Calculate denominator- 2D integral
            dL = (Lbin[j+1] - Lbin[j]) / 0.01
            Lspace = linspace(Lbin[j], Lbin[j+1], dL)   
    
            integral = []
            for z in zspace:
                dV = []
                for l in Lspace:
                    dV.append( dV_dz(l, z) )
                integral.append(scipy.integrate.simps(dV, Lspace))
    
            integ = scipy.integrate.simps(integral, zspace)
       
    #    Count sources per bin
            Ll = []
            count = 0
            for source in data:
                z = source.z
                Lx = source.l
                
                if zbin[i] <= z < zbin[i+1] and Lbin[j] <= Lx < Lbin[j+1]:
                    count = count + 1
                    Ll.append(Lx)        
            temp_Phi = count/integ
            temp_err = sqrt(count)/integ
           
            datum = [median(Zz), median(Ll), count, log10(temp_Phi), 0.434*temp_err/temp_Phi, median(Ll)-Lbin[j], Lbin[j+1]-median(Ll)]
            LF.append(datum)
            save_data.append(datum)
    
        LFU_model = []
        LFF_model = []
        LFF_low = []
        LFF_upper = []
        if LF_config.model == 'LDDE':
            for lx in ll:
                PF = Fotopoulou(lx, median(Zz), params)
                LFF_model.append(log10(PF))
        
        if LF_config.model == 'Fotopoulou2':
            for lx in ll:
                PF = Fotopoulou2(lx, median(Zz), params)
                LFF_model.append(log10(PF))
        
        if LF_config.model == 'Fotopoulou3':
            for lx in ll:
                PF = Fotopoulou3(lx, median(Zz), params)
                LFF_model.append(log10(PF))
            
        if LF_config.model == 'PLE':
            for lx in ll:
                PF = PLE(lx, median(Zz), params)
                LFF_model.append(log10(PF))
                
        if LF_config.model == 'LADE':
            for lx in ll:
                PF = LADE(lx, median(Zz), params)
                LFF_model.append(log10(PF))
        
               
        redshift = asarray(LF)[:,0]
        luminosity = asarray(LF)[:,1]
        number = asarray(LF)[:,2]
        Phi = asarray(LF)[:,3]
        err_Phi = asarray(LF)[:,4]
        lbin_l = asarray(LF)[:,5]
        lbin_h = asarray(LF)[:,6]
    
    #===============================================================================
        name = str("%.2f") % median(Zz)
    
        Phi_low = []
        Phi_high = []
        for lll in ll:
            
            LF = []
            for j in range( 0, len(draws) ):
                if LF_config.model == 'LDDE':
                    params.L0, params.g1, params.g2, params.p1, params.p2, params.zc, params.La, params.a, params.Norm = draws[j, :]
                    LF.append( log10(Fotopoulou(lll, median(Zz), params ))) 
                if LF_config.model == 'Fotopoulou2':
                    params.L0, params.g1, params.g2, params.p1, params.p2, params.zc, params.La, params.a, params.Norm = draws[j, :]
                    LF.append( log10(Fotopoulou2(lll, median(Zz), params ))) 
                if LF_config.model == 'Fotopoulou3':
                    params.L0, params.g1, params.g2, params.p1, params.p2, params.zc, params.a, params.Norm = draws[j, :]
                    LF.append( log10(Fotopoulou3(lll, median(Zz), params ))) 

                if LF_config.model == 'LADE':
                    params.L0, params.g1, params.g2, params.p1, params.p2, params.zc, params.d, params.Norm = draws[j, :]
                    LF.append( log10(LADE(lll, median(Zz), params ))) 
                if LF_config.model == 'PLE':
                    params.L0, params.g1, params.g2, params.p1, params.p2,params.zc, params.Norm = draws[j, :]
                    LF.append( log10(PLE(lll, median(Zz), params )))  
            Phi_low.append(np.mean(LF)-sigma*np.std(LF))
            Phi_high.append(np.mean(LF)+sigma*np.std(LF))
      
        ax = fig.add_subplot(3,3,i+1)
    
        plt.fill_between(ll, Phi_low, Phi_high, color=fillcolor)
        plt.plot(ll, LF0, ls='--', color=linecolor)
        plt.plot(ll, LFF_model, ls='-', color=linecolor)
        plt.errorbar(luminosity, Phi, ls = ' ',yerr=err_Phi, xerr=[lbin_l, lbin_h], label=name, markersize = 7, color=pointcolor, marker='o',markeredgecolor='black')
    #    print number of source in bin
        for j in range(0,len(number)):
            ax.annotate(str( int(number[j]) ), (luminosity[j], Phi[j]-0.1*Phi[j]), xycoords='data', fontstyle='normal', fontsize='xx-small', )
    #    print redshift bin width
        ax.annotate(str(zbin[i])+"$< $"+"z"+"$ < $"+str(zbin[i+1]), (0.1, 0.1) , xycoords='axes fraction', fontstyle='oblique', fontsize='small', )
        #plt.legend(loc=3)
        plt.xlim([40.5, 46.5])
        plt.ylim([-9.5, -2.5])
        plt.xticks([42, 43, 44, 45, 46])
        plt.yticks([-8, -6, -4])        
        
        if i+1 in [1,2,3,4,5,6]:
            ax.set_xticklabels([])
    
        if i+1 in [2,3,5,6,8,9]:
            ax.set_yticklabels([])
        
        if i+1 == 4 :
            plt.ylabel(r"$Log[d\Phi/dLogLx/(Mpc^{-3})]$")
            # ax.yaxis.set_label_coords(-0.20, -0.15)
        
        if i+1 == 8 :
            plt.xlabel(r"$Log[Lx/(erg/sec)]$")
            #ax.xaxis.set_label_coords(0.5, -0.12)    
        
        plt.draw()
        #plt.savefig("/home/Sotiria/workspace/Luminosity_Function/src/including_uncertainties/Bayesian/APEMoST/LDDE_Fotop/analysis_results/bw_dPhi%s.pdf" % i)
        #plt.savefig("/home/Sotiria/workspace/Luminosity_Function/src/including_uncertainties/Bayesian/APEMoST/LDDE_Fotop/analysis_results/bw_dPhi%s.jpg" % i)
    #plt.savefig("/home/Sotiria/workspace/Luminosity_Function/output_files/plots/Vmax_MLE_unc.pdf")
    plt.show()
    #savetxt('/home/Sotiria/workspace/Luminosity_Function/src/including_uncertainties/Bayesian/APEMoST/LDDEb_MAXI/analysis_results/Ueda_Vmax_test.dat', save_data)
    #plt.clf()