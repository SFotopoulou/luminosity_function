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

zbin = [1.040e-01, 1.161e+00, 2.421e+00, 3.376e+00]
L = linspace(41.0, 46.0, 20)

for z in zbin:
    filename_in = '/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/previous_fits/combined/Phi_'+str(z)+"_prob.dat"
    LF = genfromtxt(filename_in)
    
    out_interval_name = "/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/previous_fits/combined/"+str(z)+"_99_interval.dat"  
    out_interval = open(out_interval_name, 'w')
    pdf_all = LF
    bins = LF[:,0]
    for i in range(1, len(pdf_all[0])):
        #print i
        pdf = pdf_all[:,i]
        #print pdf
        binwidth = log10( bins[1]-bins[0] )
        #print binwidth
        idxs = pdf.argsort()
        idxs = idxs[::-1] # reverse
        print cumsum(pdf[idxs])*binwidth
        credible_interval = idxs[(cumsum(pdf[idxs])*binwidth < 0.90).nonzero()]
        print credible_interval
        idxs = idxs[::-1] # reverse
        
        low = min( sorted(credible_interval) )
        high = max( sorted(credible_interval) )
        print low, high
        min_val = bins[low]
        max_val = bins[high]
        out_interval.write(str(L[i-1])+"  "+str(min_val)+"   "+str(max_val)+"\n")
    out_interval.close()        
