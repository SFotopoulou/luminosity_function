import sys
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/no_uncertainties_included/MLE/MLE_modules/')
from numpy import array,log, log10, sqrt,linspace,vectorize,ones,column_stack,tile,sum, savetxt, genfromtxt, random, arange, histogram, cumsum, power, where,zeros
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

def Phi(Lx,z,L0,g1,g2,p1,p2,zc,La,a,Normal):    
    """ 
    The luminosity function model 
    """
    return model.Fotopoulou(Lx,z,L0,g1,g2,p1,p2,zc,La,a)*power(10.0, Normal)

## Draw from the luminosity function    
import time
start_time = time.time()
draws = genfromtxt('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/previous_fits/Fotopoulou/Fotopoulou.prob')
#zbin = [0.26, 0.73, 1.22, 1.68, 2.42, 3.38]
#zbin = [0.10, 0.34, 0.63, 0.85, 1.16, 1.47, 1.80, 2.42, 3.38]
zbin = [1.040e-01,  3.449e-01, 6.278e-01, 8.455e-01, 1.161e+00, 1.465e+00, 1.799e+00, 2.421e+00, 3.376e+00]
L = linspace(41.0, 46.0, 20)

for z in zbin:
    filename = '/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/previous_fits/Fotopoulou/Phi_'+str(z)+"_prob.dat"
    outLF = linspace(1, len(draws), len(draws))
    for Lx in L:
        print z, Lx
        LF = []
        for i in range( 0, len(draws) ):
            params = draws[i, :]
            #print i, params
            LF.append( Phi(Lx, z, *params) )
        LF = array(LF)
        outLF = column_stack((outLF, LF))
    savetxt(filename, outLF)    
    
    out_interval_name = "/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/previous_fits/Fotopoulou/"+str(z)+"_99_interval.dat"  
    out_interval = open(out_interval_name, 'w')
    pdf_all = outLF
    for i in range(1, len(pdf_all[0])):
        PDF = pdf_all[:,i]

        pdf, bins = histogram(PDF, bins = 100, normed=True)
        bins = bins[:-1]
        binwidth = bins[1]-bins[0]

#        gray = (.5,.5,.5)
#        orange = (1.0, 0.647, 0.0)
#        red = (1.0, 0.0, 0.0)

#        clrs = [gray for xx in bins]

        idxs = pdf.argsort()
        idxs = idxs[::-1]
#        reds = idxs[(cumsum(pdf[idxs])*binwidth < 0.99).nonzero()]
#        oranges = idxs[(cumsum(pdf[idxs])*binwidth < 0.6).nonzero()]
        
        credible_interval = idxs[(cumsum(pdf[idxs])*binwidth < 0.99).nonzero()]
        idxs = idxs[::-1] # reverse
        low = min( sorted(credible_interval) )
        high = max( sorted(credible_interval) )
        min_val = bins[low]
        max_val = bins[high]
        out_interval.write(str(L[i-1])+"  "+str(min_val)+"   "+str(max_val)+"\n")
    out_interval.close()        
#    for idx in oranges:
#        clrs[idx] = orange
    
#    for idx in reds:
#        clrs[idx] = red

#    plt.bar(left=bins,height=pdf,width=binwidth,color=clrs)
#    plt.show()
    
print "time lapsed:", round(time.time()-start_time, 2), "sec"
