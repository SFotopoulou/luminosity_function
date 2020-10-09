import sys
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/no_uncertainties_included/MLE/MLE_modules/')
from numpy import array,log, log10, sqrt,linspace,vectorize,ones,column_stack,tile,sum, savetxt, genfromtxt, random, arange, histogram, cumsum, power, where,zeros,mean,std
from Source import Source
from parameters import Parameters
from scipy.integrate import simps
import matplotlib.pyplot as plt
import itertools
from LFunctions import Models

params = Parameters()
Lmin, Lmax = Parameters.L(params)
zmin, zmax = Parameters.z(params)

model = Models()

def Phi(Lx,z,L0,g1,g2,p1,a,Normal):    
    """ 
    The luminosity function model 
    """
    p2 = -1.5
    zc = 1.9
    La = 44.6
    return model.Ueda(Lx,z,L0,g1,g2,p1,p2,zc,La,a)*power(10.0, Normal)

## Draw from the luminosity function    
import time
start_time = time.time()
draws = genfromtxt('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/covariance/Ebrero.prob')
#zbin = [0.26, 0.73, 1.22, 1.68, 2.42, 3.38]
#zbin = [0.10, 0.34, 0.63, 0.85, 1.16, 1.47, 1.80, 2.42, 3.38]
zbin = [1.040e-01,  3.449e-01, 6.278e-01, 8.455e-01, 1.161e+00, 1.465e+00, 1.799e+00, 2.421e+00, 3.376e+00]
L = linspace(41.0, 46.0, 10)

for z in zbin:
    filename = '/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/covariance/Ebrero_Phi_'+str(z)+"_prob.dat"
    outLF = linspace(1, len(draws), len(draws))
    for Lx in L:
        print z, Lx
        LF = []
        for i in range( 0, len(draws) ):
            params = draws[i, :]
            #print i, params
            LF.append( log10( Phi(Lx, z, *params) ) )
        LF = array(LF)
        outLF = column_stack((outLF, LF))
    savetxt(filename, outLF)    
    
    out_interval_name = "/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/covariance/Ebrero_"+str(z)+"_99_interval.dat"  
    out_interval = open(out_interval_name, 'w')
    pdf_all = outLF
    for i in range(1, len(pdf_all[0])):
        PDF = pdf_all[:,i]
        mean_val = mean(PDF)
        std_val = std(PDF)  
        out_interval.write(str(L[i-1])+"  "+str(mean_val)+"   "+str(std_val)+"   "+str(mean_val-std_val)+"   "+str(mean_val+std_val)+"   "+str(mean_val-3.0*std_val)+"   "+str(mean_val+3.0*std_val)+"   "+str(mean_val-5.0*std_val)+"   "+str(mean_val+5.0*std_val)+"   "+"\n")
    out_interval.close()        

print "time lapsed:", round(time.time()-start_time, 2), "sec"