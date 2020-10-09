import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/models')
import numpy as np
from AGN_LF_config import LF_config
from cosmology import *
from Survey import *
from Source import *
from LFunctions import *
from SetUp_grid import *
from SetUp_data import Set_up_data
from scipy.integrate import simps,romb,dblquad,quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
import time
import itertools
import pylab
#
#pylab.ion()
#pylab.hold(False)

setup_data = Set_up_data()
source_list = setup_data.get_data()[0]

L, Z, DVcA, temp_Fx, area, Redshift_int, Luminosity_int = set_up_grid()
#dz = Redshift_int[1] - Redshift_int[0]
#dL = Luminosity_int[1] - Luminosity_int[0]


def Ndata():
    return len(source_list) 

################ Multiprocessing for data ################
Nprocess = LF_config.Nprocess
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
################################################      
def Phi(Lx, z, params):    
    """ 
    The luminosity function model 
    """

    if LF_config.model == 'PLE':
        return PLE(Lx, z, params)
    
    if LF_config.model == 'halted_PLE':
        return halted_PLE(Lx, z, params)   
     
    if LF_config.model == 'PDE':
        return PDE(Lx, z, params)                          

    if LF_config.model == 'halted_PDE':
        return halted_PDE(Lx, z, params)                          
    
    if LF_config.model == 'ILDE':
        return ILDE(Lx, z, params)
    
    if LF_config.model == 'halted_ILDE':
        return halted_ILDE(Lx, z, params)    
    
    if LF_config.model == 'Hasinger':
        return Hasinger(Lx, z, params)
    
    if LF_config.model == 'Ueda':
        return Ueda(Lx, z, params)

    if LF_config.model == 'Ueda14':
        return Ueda14(Lx, z, params)

    
    if LF_config.model == 'LADE':
        return LADE(Lx, z, params)                          

    if LF_config.model == 'halted_LADE':
        return halted_LADE(Lx, z, params)                          
    
    if LF_config.model == 'Miyaji':
        return Miyaji(Lx, z, params)
    
    if LF_config.model == 'LDDE':
        return Fotopoulou(Lx, z, params)

    if LF_config.model == 'Fotopoulou2':
        return Fotopoulou2(Lx, z, params)
    
    
    if LF_config.model == 'Fotopoulou3':
        return Fotopoulou3(Lx, z, params)
    
    if LF_config.model == 'halted_LDDE':
        return halted_Fotopoulou(Lx, z, params)

    if LF_config.model == 'FDPL':
        return FDPL(Lx, z, params)
    if LF_config.model == 'Schechter':
        return Schechter(Lx, z, params)
 
    
def plot_fit(params,
             Luminosity = np.linspace(42.0, 45.5),
             redshift = [0.26, 0.73, 1.44, 2.42, 3.37],
             save_plot=False):    
    colors = itertools.cycle(['blue', 'green', 'red', 'cyan', 'black'])
    pylab.clf()
    pylab.ylim([-8,-3])
    pylab.xlim([41.5, 45.5])
    #plt.yscale("log")
    pylab.title(LF_config.model)

    for item, color in zip(redshift, colors):
        dPhi = np.log10( Phi(Luminosity, item, params) )
        pylab.plot(Luminosity, dPhi, color = color, hold=True)
    pylab.draw()    
    if save_plot == True:
        for ext in ["pdf", "svg", "eps", "png"]:
            pylab.savefig(LF_config.outpath+LF_config.method+"_"+LF_config.model+"."+ext)
    #plt.show()

def Source_sublist(list_in,params,queue):
    """Source detection probability phi*dVc*Source_prob"""
#    outfile = open('redshift.dat', 'w')
#    outfile2 = open('integrand1.dat', 'w')
#    outfile3 = open('luminosity.dat', 'w')
#    outfile4 = open('integrand2.dat', 'w')
    if LF_config.z_unc == True and LF_config.L_unc == True:
        result = []
        for source in list_in:
            Grid = source.return_MultipliedGrid_zL()
            Luminosity, Redshift = source.return_grid()
            PHI_Lz = Phi(Luminosity, Redshift, params)
            Lpoint, zpoint = source.return_points()
            
            x = Grid*PHI_Lz
            
            y = []
            count_range = xrange(0, Lpoint)
            for count in count_range:
                startz = count*zpoint
                endz = startz + zpoint            
                integrand = x[startz:endz]
                redshift_i = Redshift[startz:endz]    
#                outfile.write(str(redshift_i))
#                outfile2.write(str(integrand))
#                outfile.write("\n")
#                outfile2.write("\n")
                int1 = simps(integrand,redshift_i,even='avg')
                y.append(int1)
                #print int1
#                plt.plot(redshift_i, integrand, 'ko')
#                plt.show()
                
            luminosity_i = Luminosity[::zpoint] 

#            outfile3.write(str(luminosity_i))
#            outfile4.write(str(y))
#            outfile3.write("\n")
#            outfile4.write("\n")
            int2 = simps(y,luminosity_i,even='avg')
            result.append(int2)
#            print "+++++++++++++"
#            print int2
#            print "+++++++++++++"
        result = np.array(result)
        #result_pos = np.where(result>0, result, 1.0)
        result_log = np.log(result)#_pos)
        res = sum(result_log) # sum for each sublist
        queue.put(res)    
    elif LF_config.z_unc == True and LF_config.L_unc == False:
        result = []
        for source in list_in:
            #t1 = time.time()
            xpdf = source.return_PDFz()[0]
            #print "t1=", time.time()-t1
            #t2 = time.time()
            Grid = source.return_MultipliedGrid()
            #print "t2=", time.time()-t2
            #t3 = time.time()
            Luminosity, Redshift = source.return_lumis()
            
            PHI_Lz = Phi(Luminosity, Redshift, params)            
            #print "t3=", time.time()-t3
            #t4 = time.time()
            x = Grid*PHI_Lz
            #print "t4=", time.time()-t4            
            #t5 = time.time()
            int1 = simps(x, xpdf,even='avg')
            #print "t5=", time.time()-t5
          
            
            result.append(int1)
#        
#        result=  [ simps(source.return_MultipliedGrid() *\
#                             Phi(source.l, source.return_PDFz()[0], params),\
#                             source.return_PDFz()[0], even='avg') \
#                             for source in list_in ]
        result = np.array(result)
        #print result
        #result_pos = np.where(result>0, result, 1.0)
        result_log = np.log(result)#_pos)
        res = sum(result_log) # sum for each sublist
        queue.put(res) 
    elif LF_config.z_unc == False:
        result = 0.0
        for source in list_in:
            ID, field, counts, e_counts, flux, e_flux, mag, z, zflag = source.return_data()
            #print "source data:",ID, field, flux, e_flux, z, zflag
            Lum = source.l
            #print "luminosity:",Lum
            PHI_Lz = Phi(Lum, z, params)
            #print "dPhi:", PHI_Lz
            Vol = dif_Vc(z)
            #print "Volume:",Vol
            result = np.log(PHI_Lz*Vol) + result    
            #print "ln(dPhi*Vol):", np.log(PHI_Lz*Vol)
        queue.put(result)    
    
def Marshall_Likelihood(params):
    """
    Likelihood for MLE L = -2.0*ln(L)
    """
    """ 
    Data probability
    """
    #plot_fit(params)
    #t_dat = time.time()
    processes = []
    for queue_in, list_in in zip(queue_list, list_list):
        p = Process(target=Source_sublist, args=(list_in,params,queue_in))
        p.start()
        processes.append(p)
        
    result = []
    for q in queue_list:
        result.append( q.get() )
                
    for p in processes:
        p.join()
            
    source_d = sum(result) # sum of all sublists
    # print "data in:",time.time()-t_dat,"sec"
    """
       Survey detection probability Phi*dV*Area_curve
    """
#    t_vol = time.time()
#    PHI_Lz = Phi(L,Z,params)
#    
#    X = DVcA*PHI_Lz # integrand
#
#    y = []
#    count_r = xrange(0, LF_config.Lpoints)
#    for count in count_r:
#        
#        startz = count * LF_config.zpoints
#        endz = startz + LF_config.zpoints
#        x = X[startz:endz]
#
#        int1_new = romb(x, dz)
#        y.append(int1_new)
#    survey_d = romb(y, dL)
#    
##    """ Marshall Likelihood including errors"""
#    print "vol: ",survey_d," in ",time.time()-t_vol, "sec"
##----------------------------------------------------
#    t_vol = time.time()
#    
    PHI_Lz = Phi(L,Z,params)

    X = DVcA*PHI_Lz # integrand

    y = []
    count_r = xrange(0, LF_config.Lpoints)
    non_zero_int = []
    general_int = []
    simps_avg = []
    simps_first = []
    simps_last = []
    for count in count_r:

        startz = count * LF_config.zpoints
        endz = startz + LF_config.zpoints
        x = X[startz:endz]
        
        #plt.plot(Redshift_int, x, '-')

#        print "x: ",x
#        print
#        print "no zero x: ", np.nonzero(x)
#        print
#        print "redshift no zero_x: ",Redshift_int[np.nonzero(x)]
#        print
#        print "limits redshift", min(Redshift_int), max(Redshift_int), max(Redshift_int[np.nonzero(x)])
#        xintp = interp1d(Redshift_int, x, kind='linear')
#        print "interpolated function: ",xintp(Redshift_int)
#        print
##        xintplinear = interp1d(Redshift_int, x, kind='linear')
#        print "redshift points",Redshift_int
#        print
#        print "Luminosity:", Luminosity_int[count]
#        x_int = quad(xintp, min(Redshift_int), max(Redshift_int[np.nonzero(x)]) )[0]
#        print "non zero integral: ", x_int
#        non_zero_int.append(x_int)
#        print
#        x_int = quad(xintp, min(Redshift_int), max(Redshift_int) )[0]
#        print "general intergral: ", x_int
#        print
#        general_int.append(x_int)
        int1 = simps(x, Redshift_int, even='avg')
#        print "simps integral, avg: ", int1
#        print
#        simps_avg.append(int1)
#        int1 = simps(x, Redshift_int, even='first')
#        print "simps integral, first: ", int1
#        print
#        simps_first.append(int1)
#        int1 = simps(x, Redshift_int, even='last')
#        print "simps integral, last: ", int1
#        print                
        simps_last.append(int1)
        #raw_input()
        y.append(int1)
    #yintp = interp1d(Luminosity_int, y, kind='linear')
#    plt.plot(Luminosity_int, non_zero_int, label='non zero')
#    plt.plot(Luminosity_int, general_int, label='general')
#    plt.plot(Luminosity_int, simps_avg, label='simps avg')
#    plt.plot(Luminosity_int, simps_first, label='simps first')
#    plt.plot(Luminosity_int, simps_last, label='simps last')
#    plt.legend()
#    plt.title(str(LF_config.zpoints))
#    plt.show()  
#    plt.plot(Redshift_int, DVcA[startz:endz], 'o')
#    plt.plot(Redshift_int, PHI_Lz[startz:endz], ':')
#    plt.show()    
##     
#    plt.plot(Luminosity_int, y, '-')
#    #plt.plot(Luminosity_int, yintp(Luminosity_int), '-')
#    plt.show()
    

    #survey_d = quad(yintp, min(Luminosity_int), max(Luminosity_int))[0]
 
    survey_d = simps(y, Luminosity_int, even='avg')
#    print
#    print survey_d, y_int, survey_d-y_int
#    print
    
## Compare to interpolation
#    XX = interpolate.interp2d(L, Z, X)
#    survey = dblquad(XX, 42.0, 46.0, lambda x: 0.01, lambda x: 4.0)    
#    #print survey_d/survey[0]
#    survey_d = -survey[0]
    
#    """ Marshall Likelihood including errors"""
    #print "vol:",survey_d, " in ", time.time()-t_dat, " sec"
#    
    #raw_input()    
    Like = -source_d + survey_d # Difference, no area curve in the data prob.
#    print "data: ",source_d
#    print "survey: ", survey_d
#    
    return 2.0*Like + LF_config.Likelihood_offset
    
class Params: pass
params = Params()

def PLE_Likelihood(L0, g1, g2, p1, p2, zc, Norm):
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.zc = zc
    params.Norm = Norm
    return Marshall_Likelihood(params)

def halted_PLE_Likelihood(L0, g1, g2, p1, zc, Norm):
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.zc = zc
    params.Norm = Norm
    return Marshall_Likelihood(params)

def PDE_Likelihood(L0, g1, g2, p1, p2, zc, Norm):
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.zc = zc
    params.Norm = Norm
    return Marshall_Likelihood(params)

def halted_PDE_Likelihood(L0, g1, g2, p1, zc, Norm):
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.zc = zc
    params.Norm = Norm
    return Marshall_Likelihood(params)

def ILDE_Likelihood(L0, g1, g2, p1, p2, Norm):
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.Norm = Norm
    return Marshall_Likelihood(params)

def halted_ILDE_Likelihood(L0, g1, g2, p1, p2, zc, Norm):
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.zc = zc
    params.Norm = Norm
    return Marshall_Likelihood(params)

def Hasinger_Likelihood(L0, g1, g2, p1, p2, zc, La, a, b1, b2, Norm):
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.zc = zc
    params.La = La
    params.a = a
    params.b1 = b1
    params.b2 = b2
    params.Norm = Norm
    return Marshall_Likelihood(params)
    
def Ueda_Likelihood(L0, g1, g2, p1, p2, zc, La, a, Norm):
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.zc = zc
    params.La = La
    params.a = a
    params.Norm = Norm
    return Marshall_Likelihood(params)

#def Ueda14_Likelihood(L0, g1, g2, p1, beta, zc1, La1, a1, a2, Norm):
def Ueda14_Likelihood(L0, g1, g2, p1, beta, Lp, p2, p3, zc1, zc2, La1, La2, a1, a2, Norm):

    #print L0, g1, g2, p1, beta, Lp, p2, p3, zc1, zc2, La1, La2, a1, a2, Norm
    #raw_input()
    # fixed params from Ueda
    #m.values["zc2"] = 3.0 #LF_config.zc2
    #m.values["La2"] = 44.0 #LF_config.La2
    #m.values["Lp"] = 44.0 #LF_config.Lp
    #m.values["p2"] = -1.5 #LF_config.p2
    #m.values["p3"] = -6.2 #LF_config.p3
    #m.values["a2"] = -0.1 #LF_config.a2

    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.beta = beta
    params.Lp = Lp#44.0
    params.p2 = p2#-1.5
    params.p3 = p3#-6.2
    params.zc1 = zc1
    params.La1 = La1
    params.a1 = a1
    params.zc2 = zc2#3.0
    params.La2 = La2#44.0
    params.a2 = a2#-0.1
    params.Norm = Norm
    return Marshall_Likelihood(params)
    
def LADE_Likelihood(L0, g1, g2, p1, p2, zc, d, Norm):                          
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.zc = zc
    params.d = d
    params.Norm = Norm
    return Marshall_Likelihood(params)
    
def halted_LADE_Likelihood(L0, g1, g2, p1, p2, zc, d, Norm):                          
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.zc = zc
    params.d = d
    params.Norm = Norm
    return Marshall_Likelihood(params)

def Miyaji_Likelihood(L0, g1, g2, p1, p2, zc, La, a, pmin, Norm):
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.zc = zc
    params.La = La
    params.a = a
    params.pmin = pmin
    params.Norm = Norm
    return Marshall_Likelihood(params)
    
def Fotopoulou_Likelihood(L0, g1, g2, p1, p2, zc, La, a, Norm):
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.zc = zc
    params.La = La
    params.a = a
    params.Norm = Norm
    return Marshall_Likelihood(params)

def Fotopoulou2_Likelihood(L0, g1, g2, p1, p2, zc, La, a, Norm):
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.zc = zc
    params.La = La
    params.a = a
    params.Norm = Norm
    return Marshall_Likelihood(params)


def Fotopoulou3_Likelihood(L0, g1, g2, p1, p2, zc, a, Norm):
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.zc = zc
    params.a = a
    params.Norm = Norm
    return Marshall_Likelihood(params)


def halted_Fotopoulou_Likelihood(L0, g1, g2, p1, zc, La, a, Norm):
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.zc = zc
    params.La = La
    params.a = a
    params.Norm = Norm
    return Marshall_Likelihood(params)


def FDPL_Likelihood(K0,K1,L0,L1,L2,g1,g2):
    params.K0 = K0
    params.K1 = K1
    params.L0 = L0
    params.L1 = L1
    params.L2 = L2
    params.g1 = g1
    params.g2 = g2
    return Marshall_Likelihood(params)


def Schechter_Likelihood(A, Lx, a, b):
    params.A = A
    params.Lx = Lx
    params.a = a
    params.b = b
    return Marshall_Likelihood(params)
#print "Likelihood ok"
