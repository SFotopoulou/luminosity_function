#!/home/Sotiria/Software/anaconda3/bin/python

def run_MLE():
    import os
    import iminuit as minuit
    import numpy as np
    #from AGN_LF_config import LF_config
    #from cosmology import *
    #from Survey import *
    #from Source import *
    #from LFunctions import *
    import Likelihood as lk
    #from scipy.integrate import simps
    #import matplotlib.pyplot as plt
    #from multiprocessing import Process, Queue
    #import warnings
    
    #with warnings.catch_warnings():
    #    warnings.simplefilter('ignore')
    #    ############### Observations ###############
    #    # Prepare individual grid from each datum ##
    #    ############################################
    path = LF_config.outpath + "MLE/"
    if not os.path.exists(path): os.mkdir(path)
    print('results in: ', path)
    
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
        print('LDDE')
        m = minuit.Minuit(lk.Fotopoulou_Likelihood,        
                            limit_L0 = (40.0,47.0),
                            limit_g1 = (-2.0, 1.5), 
                            limit_g2 = (1.5, 5.0),
                            limit_p1 = (0.0, 10.0), 
                            limit_p2 = (-20.0, 5.0),        
                            limit_zc = (0.0, 6.0), 
                            limit_La = (40.0, 47.0),
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
        print( 'Fotopoulou2')
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
        print( 'Fotopoulou3')
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
        print( 'Ueda14')
        m = minuit.Minuit(lk.Ueda14_Likelihood)
        m.values["L0"] = LF_config.L0 
        m.values["g1"] = LF_config.g1 
        m.values["g2"] = LF_config.g2 
        m.values["p1"] = LF_config.p1 
        m.values["beta"] = LF_config.beta 
        
        m.values["zc1"] = LF_config.zc1
        m.values["La1"] = LF_config.La1
        m.values["a1"] = LF_config.a1 
       
        m.values["Norm"] = LF_config.Norm
        keys = ['L0', 'g1', 'g2', 'p1', 'beta', 'zc1', 'La1', 'a1', 'a2', 'Norm']    
    print( LF_config.outname)
    
    m.print_level = 2
    m.strategy = 2.0
    m.set_errordef(0.5)
    Nparameters = len(m.values)
    Ndata = lk.Ndata()
    
    field_names = ', '.join( [x for x in LF_config.fields])
    
    print( "MLE, using "+LF_config.model+" model and fields: "+field_names)
    print( "Nparams:", Nparameters)
    print( "Ndata:", Ndata)
    print( str( LF_config.zmin ) + "<= z <=" + str( LF_config.zmax ))
    print( str( LF_config.Lmin ) + "<= L <=" + str( LF_config.Lmax ))
    if LF_config.z_unc == True:
        print( "including uncertainties")
    else:
        print( "not including uncertainties")
    print( "\n------------------------------")
    
    
    m.migrad(ncall=100000, resume=False, nsplit=1)
    
    
    
    print( "MLE finished")
    print( "\n------------------------------")
    print( "Number of function calls: ",m.ncalls)
    print( "Vertical distance to minimum: ", m.edm)
    print( "Best fit paramerers:\n")
    for key, value in m.values.items():
        print( key," = ",value)
    
    
    output = open(path+"MLE_"+LF_config.outname,'w')
    
    output.write("MLE, using "+LF_config.model+" model and fields: "+field_names+"\n")
    output.write( "Nparams: "+str(Nparameters)+"\n")
    output.write( "Ndata: "+str(Ndata)+"\n")
    output.write( str( LF_config.zmin ) + "<=z<=" + str( LF_config.zmax )+"\n")
    output.write( str( LF_config.Lmin ) + "<=L<=" + str( LF_config.Lmax )+"\n")
    if LF_config.z_unc == True:
        output.write( "including uncertainties\n")
    else:
        output.write( "not including uncertainties\n")
    output.write( "\n------------------------------------------------------------------\n")
    output.write( "Least likelihood value = "+str(m.fval-LF_config.Likelihood_offset)+"\n")
    output.write( "with parameters :\n")
    for key, value in m.values.items():
        output.write(str(key)+" = "+str(value)+"\n")
    output.write( "\n------------------------------------------------------------------\n")
    output.write( "\n\nSymmetric Errors\n" )
    for key, value in m.errors.items():
        output.write(str(key)+" = "+str(value)+"\n")
    output.write("\n")
    output.write( str(m.errors) )
    output.write( "\n------------------------------------------------------------------\n")
    output.write( "\n")
    output.write( "Minimum distance between function calls: "+str(m.edm)+"\n")
    output.write( "Vertical distance to minimum : "+str(m.edm)+"\n")
    output.write( "\n")
    output.write( "AIC="+str(2.*Nparameters+m.fval-LF_config.Likelihood_offset)+"\n")
    output.write( "AICc="+str(2.*Nparameters+m.fval-LF_config.Likelihood_offset+(2.*Nparameters*(Nparameters+1.))/(Ndata-Nparameters-1.))+"\n")
    output.write( "BIC="+str(m.fval-LF_config.Likelihood_offset+Nparameters*np.log10(Ndata))+"\n")
    try:
        #    Calculate asymmetric errors 
        print( "calling Minos")
        m.minos()
        output.write( "\n------------------------------------------------------------------\n")
        output.write( "\n\nAsymmetric Errors\n" )
        merrors = m.get_merrors()
        for key, value in merrors.items():
            output.write(str(key)+" = "+str(value)+"\n")
    except:
        print("minos failed")
    #    Calculate errors and covariance matrix
    print( "calling Hesse")
    m.hesse()
    output.write( "\n------------------------------------------------------------------\n")
    output.write( "\n\nCovariance Matrix\n" )
    output.write( str(m.covariance) )
    print( np.array(m.matrix()))
    output.write( "\n------------------------------------------------------------------\n")
    output.write( "\n\nCorrelation Matrix\n" )
    output.write( str(m.matrix(correlation=True)) )
    output.write( "\n------------------------------------------------------------------\n")
    output.write( "\n\nCovariance Matrix\n" )
    output.write(np.array(m.matrix()))
    output.write( "\n------------------------------------------------------------------\n")

if __name__ == "__main__":
    import AGN_LF_config
    LF_config = AGN_LF_config.LF_config()
    run_MLE(LF_config)
