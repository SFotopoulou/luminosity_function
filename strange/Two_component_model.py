import sys,os
# Add the module path to the sys.path list
#sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/no_uncertainties_included/MLE/MLE_modules')
#
sys.path.append( os.path.join(os.environ['HOME']+'/workspace/Luminosity_Function/src/LF_modules/') )
import time
import math   
import numpy as np 
from Source import Source
from parameters import Parameters
from LFunctions import Models
from scipy.integrate import simps,dblquad
import minuit
from class_source import AGN
from make_PDFz import Spectrum
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import numpy.ma as ma
params = Parameters()
Lmin, Lmax = Parameters.L(params)
zmin, zmax = Parameters.z(params)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
        
    model = Models()
    
    Vmax_z, Vmax_L, Vmax_count, Vmax_Phi, Vmax_ePhi, Vmax_lowLbin, Vmax_highLbin = np.genfromtxt('/home/sotiria/workspace/Luminosity_Function/src/including_uncertainties/Bayesian/APEMoST/LDDE_Fotop/analysis_results/Fotopoulou_Vmax.dat', unpack=True)
    #Vmax_z, Vmax_L, Vmax_count, Vmax_Phi, Vmax_ePhi, Vmax_lowLbin, Vmax_highLbin = np.genfromtxt('/home/sotiria/workspace/Luminosity_Function/src/no_uncertainties_included/Vmax/PageCarrera/Vmax.dat', unpack=True)
    redshift = [1.040000021457672119e-01, 3.449999988079071045e-01, 6.277999877929687500e-01, 8.455500006675720215e-01, 1.161000013351440430e+00, 1.465499997138977051e+00, 1.799999952316284180e+00, 2.415999889373779297e+00, 3.376500010490417480e+00]
    
    from scipy.stats import norm
    #   Two Component Model:
    #   Low Luminosity AGN, power law
    #   High Luminosity - quasars - lognormal -> rate of mergers higher at higher z?
    
    def Schechter_low(Lx,z,L01,a1,zc1,p11,p21,phi1):
    
    #    Lx = np.power(10.,Lx)
    #    La = np.power(10.,La)
    #    La = La*(1+z)**k
    #    #a = a/(1+z)**k
    #    x = Lx/La
    #    return norm*np.power(x,a)*np.exp(-x)
        Lx = np.power(10.,Lx)
        L01 = np.power(10.,L01)    
        phi1 = np.power(10.0, phi1)
        scale1 = np.where(z <= zc1, np.power((1.+z), p11), (np.power((1.+zc1), p11))*(np.power((1.+z)/(1.0+zc1), p21)))
        L01 = L01*scale1
        x1 = Lx/L01
        return phi1*np.power(x1,a1)*np.exp(-x1)
    
    def Schechter_high(Lx,z,L02,a2,zc2,p12,p22,phi2):
        
    #    Lx = np.power(10.,Lx)
    #    La = np.power(10.,La)
    #    La = La*(1+z)**k
    #    #a = a/(1+z)**k
    #    
    #    x = Lx/La
    #    return norm*np.power(x,a)*np.exp(-x)
        Lx = np.power(10.,Lx)
        L02 = np.power(10.,L02)  
        phi2 = np.power(10.0, phi2)  
        scale2 = np.where(z <= zc2, np.power((1.+z), p12), (np.power((1.+zc2), p12))*(np.power((1.+z)/(1.0+zc2), p22)))
        L02 = L02*scale2
        x2 = Lx/L02
        return phi2*np.power(x2,a2)*np.exp(-x2)
    
    x_plots = 3
    y_plots = 3 
    Lx = np.linspace(40,46)
    fig = plt.figure(figsize=(15,15))
    fig.subplots_adjust(left=0.13, right=0.97,wspace=0.35, hspace=0.25)
    
    for z in redshift:
        ax = fig.add_subplot(y_plots, x_plots, redshift.index(z)+1)
    
        mask = np.where(Vmax_z==z,0,1)
        vz = ma.masked_array(Vmax_z, mask)
        LF_L = ma.masked_array(Vmax_L ,mask)    
        LF_low = ma.masked_array(np.array(Vmax_lowLbin), mask)
        LF_upper = ma.masked_array(np.array(Vmax_highLbin), mask)
        LF_Phi = ma.masked_array(np.array(Vmax_Phi), mask)
        LF_ePhi = ma.masked_array(np.array(Vmax_ePhi), mask)
        ax.annotate("z = "+str(round(z,2)), (0.1, 0.1) , xycoords='axes fraction', fontstyle='oblique', fontsize='medium', )
#        phi1,L01,a1,phi2,L02,a2,p11,p21,zc1,p12,p22,zc2
#        LF1 = Schechter_low(Lx,z,phi1=4e-5,L01=42.5, a1=-0.2, p11=2.0,p21=0.0,zc1=1.0)
#        plt.plot(Lx, np.log10(LF1),label='Low Lum')  
#                
#        LF2 = Schechter_high(Lx,z, phi2=2e-5,L02=43.5,a2=0.5,p12=2.0,p22=0.0,zc2=1.0)
#        plt.plot(Lx, np.log10(LF2),label='High Lum') 

        LF1 = Schechter_low(Lx,z,phi1=-6.1,L01=42.5, a1=-0.2, p11=2.0,p21=0.0,zc1=1.0)
        plt.plot(Lx, np.log10(LF1),label='Low Lum')  
                
        LF2 = Schechter_high(Lx,z, phi2=-4.69897,L02=43.5,a2=0.5,p12=2.0,p22=0.0,zc2=1.0)
        plt.plot(Lx, np.log10(LF2),label='High Lum') 
        
        plt.plot(Lx, np.log10(LF1+LF2),label='Schecter-Schecter') 
       
        plt.errorbar(LF_L, LF_Phi, xerr=[LF_low, LF_upper] , yerr=LF_ePhi, ls=' ', color='k',markersize=14, marker='o')
        
        i = redshift.index(z)
    
        if i == 8:
            i = i +2
            plt.xlabel('Luminosity', fontsize='x-large')
            ax.xaxis.set_label_coords(-0.75, -0.2)
        if i == 3:
            i = i +2
            plt.ylabel(r'd$\Phi$/dlogLx', fontsize='x-large')
            ax.yaxis.set_label_coords(-0.30, 0.5)
        if i ==1 :
            i = i+1
        
        plt.ylim([-11.5,-0.5])
        plt.xlim([40.5, 46.5 ])
        plt.xticks([41, 42, 43, 44, 45, 46])
        plt.yticks([-10, -8, -6, -4, -2])
        
    plt.show()
    
    """
    def Phi(Lx,z,L0,g1,g2,p1,p2,zc,La,a,Normal):    
     
    #    The luminosity function model 
    
        return model.TwoComp(Lx,z,L0,g1,g2,p1,p2,zc,La,a)*power(10.0, Normal)
        
    def Source_sublist(list_in,L0,g1,g2,p1,p2,zc,La,a,Normal,queue):
    #    Source detection probability phi*dVc*Source_prob
        result = []
        for source in list_in:
            Grid = source.return_MultipliedGrid()
            Luminosity, Redshift = source.return_grid()
            PHI_Lz = Phi(Luminosity, Redshift, L0,g1,g2,p1,p2,zc,La,a,Normal)
            Lpoint, zpoint = source.return_points()
            
            x = Grid*PHI_Lz
            
            y = []
            count_range = xrange(0, Lpoint)
            for count in count_range:
                startz = count*zpoint
                endz = startz + zpoint            
                integrand = x[startz:endz]
                redshift_i = Redshift[startz:endz]    
                int1 = simps(integrand,redshift_i,even='last')
                y.append(int1)
            
            luminosity_i = Luminosity[::zpoint] 
    
            int2 = simps(y,luminosity_i,even='last')
            result.append(int2)
        
        result = array(result)
        result_pos = where(result>0, result, 1.0)
        result_log = log(result_pos)
        res = sum(result_log) # sum for each sublist
    
        queue.put(res)    
     
    
    def Marshall_Likelihood(L0,g1,g2,p1,p2,zc,La,a,Normal):
    
    #    Likelihood for MLE L = -2.0*ln(L)
    
    #    Data probability
    
        processes = []
        for queue_in, list_in in zip(queue_list, list_list):
            p = Process(target=Source_sublist, args=(list_in,L0,g1,g2,p1,p2,zc,La,a,Normal,queue_in))
            p.start()
            processes.append(p)
            
        result = []
        for q in queue_list:
            result.append( q.get() )
                    
        for p in processes:
            p.join()
                
        source_d = sum(result) # sum of all sublists
    
    
    #       Survey detection probability Phi*dV*Area_curve
    
        PHI_Lz = Phi(L,Z,L0,g1,g2,p1,p2,zc,La,a,Normal)
        X = DVcA*PHI_Lz # integrand
        
        y = []
        count_r = xrange(0, Lpoints)
        for count in count_r:
            startz = count*zpoints
            endz = startz + zpoints
            x = X[startz:endz]
            
            int1 = simps(x,Redshift_int, even='last')
            y.append(int1)
       
        survey_d = simps(y,Luminosity_int, even='last')
    #    Marshall Likelihood including errors
        
        Like = - source_d + survey_d # Difference, no area curve in the data prob.
        return 2.0*Like+37758.8
    
    start_time = time.time()
    ############### Observations ###############
    # Prepare individual grid from each datum ##
    ############################################
    Lpoints = int((Lmax-Lmin)/0.01)
    zpoints = int((zmax-zmin)/0.01)
    
    try:
        L, Z, DVcA  = genfromtxt('input_integral_MAXI.dat', unpack=True)
    except IOError as e:
        print 'Oh dear. Generating data file 1, be patient...'   
        LL = array([ones( (zpoints), float )*item for item in linspace(Lmin,Lmax,Lpoints)])
        L = LL.ravel() #    make LL 1D
        Z = tile(linspace(zmin, zmax, zpoints), Lpoints) # repeat as many times as Lpoints
        # Set up grid for survey integral
        g = Source('grid')
        DVcA = g.Dz_area(L,Z)
        integr = zip(L, Z, DVcA)
        savetxt('input_integral_MAXI.dat', integr)    # save to file
    
    
    Redshift_int = Z[0:zpoints]
    Luminosity_int = linspace(Lmin,Lmax,Lpoints)
    
    #############################################################################################3
    #    Prepare data
    print "Preparing data"
    d = Source('data')
    ID, Fx, e_Fx, z, z_flag, field = d.get_data()
    ndat = len(ID)
    t1 = time.time()
    ################## for data creation #############################
    def source_creation(source_list_in, source_queue_in):
        source_list = []
        for ID, Fx, e_Fx, z, z_flag, field in source_list_in:
            source = AGN( ID, Fx, e_Fx, z, z_flag, field )
            source.PDFz()
            source.make_grid()
            source.PDFf()
            source_list.append(source)
        source_queue_in.put(source_list)
    
    ################## multiprocessing for data #############################
    Nprocess = 4
    step = int( len(ID)//Nprocess )
    source_queue_list = []
    source_list_list = []
    if Nprocess > 1:
        n_range = xrange(0, Nprocess)     
        for n in n_range:
            start_point = n*step
            end_point = (n+1)*step
            if n < Nprocess-1:
                temp_list = zip(ID[start_point:end_point], Fx[start_point:end_point], e_Fx[start_point:end_point], z[start_point:end_point], z_flag[start_point:end_point], field[start_point:end_point])
            else:
                temp_list = zip(ID[start_point:], Fx[start_point:], e_Fx[start_point:], z[start_point:], z_flag[start_point:], field[start_point:])
            source_list_list.append(temp_list)
            source_queue_list.append(Queue())
    else:
        source_list_list.append( zip(ID[:], Fx[:], e_Fx[:], z[:], z_flag[:], field[:]) )
        source_queue_list.append( Queue() )    
    
    ############################### create data ########################################################
    source_processes = []
    for source_queue_in, source_list_in in zip(source_queue_list, source_list_list):
        p = Process(target=source_creation, args=(source_list_in, source_queue_in))
        p.start()
        source_processes.append(p)
        
    source_list = []
    for q in source_queue_list:
        source_list.extend( q.get() )
                
    for p in source_processes:
        p.join()
            
    print "Source list creation:", time.time()-t1,"sec"
    print "-------------------------------------------"
    ##########################################################################################
    Nprocess = 4
    step = int( len(source_list)//Nprocess )
    queue_list = []
    list_list = []
    if Nprocess > 1:
        n_range = xrange(0, Nprocess)
        for n in n_range:
            queue_list.append(Queue())
            if n < Nprocess-1:
                temp_list = source_list[n*step:(n+1)*step]
            else:
                temp_list = source_list[n*step:]
            list_list.append(temp_list)
    else:
        list_list.append( source_list[:] )
        queue_list.append( Queue() )
    ##########################################################################################
    m = minuit.Minuit(Marshall_Likelihood)
    #    Accuracy
    m.printMode = 0
    m.up = 1.0
    Nparameters = 9.0
    Ndata = 499
    #    Careful! Do not use integers!
    #    Initial Values
    m.values["L0"] = 43.9
    m.values["g1"] = 1.190
    m.values["g2"] = 2.700
    m.values["p1"] = 4.833
    m.values["p2"] = -3.151
    m.values["zc"] = 2.4104
    m.values["La"] = 44.8
    m.values["a"] = 0.366
    m.values["Normal"] = -6.524
    
    print "------------------------------"
    print "Running MLE"
    print "Desired tolerance :", m.tol
    m.strategy = 1
    m.migrad()
    print
    print m.values
    
    Normal = m.values["Normal"]
    L0 = m.values["L0"]
    g1 = m.values["g1"]
    g2 = m.values["g2"]
    p1 = m.values["p1"]
    p2 = m.values["p2"]
    zc = m.values["zc"]
    La = m.values["La"]
    a = m.values["a"]
    
    print "MLE converged"
    print "Time lapsed =",round(time.time()-start_time,2),"sec"
    print "------------------------------"
    print "Parameter values are:"
    print "N=", Normal
    print "L0 =", L0
    print "g1 =", g1
    print "g2 =", g2
    print "p1 =", p1
    print "p2 =", p2
    print "zc =", zc
    print "La =", La
    print "a =", a
    print "Number of function calls:",m.ncalls
    print "Vertical distance to minimum", m.edm
    
    output = open('TwoComp.out','w')
    output.write("No. model parameters = "+str(Nparameters)+"\n")
    output.write( "Least likelihood value = "+str(m.fval-37758.8)+"\n")
    output.write( "with parameters : "+str(m.values)+"\n")
    output.write( "Vertical distance to minimum : "+str(m.edm)+"\n")
    output.write( "AIC="+str(2.*Nparameters+m.fval-37758.8)+"\n")
    #output.write( "AICc="+str(2.*Nparameters+m.fval+(2.*Nparameters*(Nparameters+1.))/(Ndata-Nparameters-1.))+"\n")
    #output.write( "BIC="+str(m.fval+Nparameters*log10(Ndata))+"\n")
    output.close()
    #    Calculate errors and covariance matrix
    print "calling Hesse"
    output = open('TwoComp.out','a')
    m.hesse()
    output.write( "\nLinear Errors\n" )
    output.write( str(m.errors) )
    output.write( "\n\nCovariance Matrix\n" )
    output.write( str(m.covariance) )
    #output.write( "\n\nMINOS Errors\n" )
    output.close()
    # calculate non linear errors
    print
    print m.values
    print
    print m.errors
    print
    print
    
    print "calling MINOS"
    for param in ["L0", "g1", "g2", "p1", "p2", "zc", "La",  "a", "Normal"]:
        try:
            print param, "-1"
            m.minos(param, -1)
            output = open('test_TwoComp.out','a')
            output.write( str(m.merrors)+"\n" )
            output.close()
        except:
            print "problem in param ", param, "-1\n"
            output = open('test_TwoComp.out','a')
            output.write( "problem in param "+param+" -1 "+"\n" )
            output.close()
            pass
            
    for param in ["L0", "g1", "g2", "p1", "p2", "zc", "La",  "a", "Normal"]:
        try:
            print param, "1"
            m.minos(param, 1)
            output = open('test_TwoComp.out','a')
            output.write( str(m.merrors)+"\n" )
            output.close()
        except:
            print "problem in param ", param, "1\n"
            output = open('test_TwoComp.out','a')
            output.write( "problem in param "+param+" 1 "+"\n" )
            output.close()
            pass
           
    print "================================"        
    print "Minimization finished with:"
    print "Function minimum: ", m.fval
    print "MLE parameters: ", m.values
    print "Linear Errors: ", m.errors
    #print "MINOS Errors: ", m.merrors
    print "================================"        
    
    redshift = [0.26, 0.73, 1.44, 2.42, 3.37]
    Luminosity = linspace(42.0, 45.5)
    import itertools
    colors = itertools.cycle(['blue', 'green', 'red', 'cyan', 'black'])
    
    for item, color in zip(redshift, colors):
        dPhi = log10( Phi(Luminosity, item,L0,g1,g2,p1,p2,zc,Normal) )
        plt.plot(Luminosity, dPhi, color = color)
    
    plt.ylim([-8,-3])
    plt.xlim([41.5, 45.5])
    #plt.yscale("log")
    plt.title("Pure Density Evolution")
    plt.draw()
    plt.savefig("TwoComp.eps")
    plt.savefig("TwoComp.pdf")
    plt.savefig("TwoComp.jpg")
    plt.show()
    """