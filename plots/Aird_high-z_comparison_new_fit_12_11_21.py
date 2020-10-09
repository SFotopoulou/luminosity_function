import warnings
import sys
# Append the module path to the sys.path list
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/LF_modules/configuration/')
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/LF_modules/models/')
import time
import math   
from numpy import array,log,sqrt,power,linspace,vectorize,ones,tile,sum, log10, savetxt, genfromtxt,loadtxt,isnan,min,max
from Source import Source

from LFunctions import Models
from scipy.integrate import simps
import minuit
import matplotlib.pyplot as plt
import itertools
from AGN_LF_config import LF_config
import numpy as np
from matplotlib.transforms import Bbox
from matplotlib.path import Path
from matplotlib.patches import Rectangle

model = Models()
#Vmax_z, Vmax_L, Vmax_count, Vmax_Phi, Vmax_ePhi, Vmax_lowLbin, Vmax_highLbin = genfromtxt('/home/sotiria/workspace/Luminosity_Function/src//Vmax/PageCarrera/Vmax.dat', unpack=True)
#    Data from Dexter
#Aird_ll = itertools.cycle([[42.878,43.249,43.620,43.991,44.362,44.734],[42.887,43.237,43.657,44.008,44.334,44.756,45.103]])
#Aird_LF = itertools.cycle([[5.823e-5,3.334e-5,1.706e-5,1.157e-5,5.920e-6,5.325E-6],[1.623e-4,5.945e-5,2.306e-5,1.657e-5,7.181e-6,5.782e-6,6.888e-7]])    
#err_Aird_LF = itertools.cycle()
#
#AirdX_ll = itertools.cycle([[43.534,43.930,44.302,44.697,45.069],[43.230,43.624,43.975,44.348,44.746,45.066]])
#AirdX_LF = itertools.cycle([[2.919e-5,1.417e-5,1.437e-5,6.981e-6,8.684e-7],[3.209e-5,1.142e-5,1.058e-5,5.285e-6,4.619e-6,7.110e-7]])
#err_AirdX_LF = itertools.cycle()

#    Data from James
# 2<z<2.5, computed at 2.25, 2.5<z<3.5, computed at 3
# X-ray sample
Aird_l = itertools.cycle([[43.5625, 43.9375, 44.3125, 44.6875, 45.0625], [43.1875, 43.5625, 43.9375, 44.3125, 44.6875, 45.0625]] )
e_Aird_low = itertools.cycle([[43.3750,43.7500,44.1250,44.5000,44.8750],[43.0000,43.3750,43.7500,44.1250,44.5000, 44.8750]])
e_Aird_high = itertools.cycle([[43.7500,44.1250,44.5000,44.8750,45.2500],[43.3750,43.7500,44.1250,44.5000, 44.8750,45.2500]])

Aird_XLF = itertools.cycle([[2.65939e-05, 1.22097e-05, 1.23578e-05, 5.67365e-06, 6.95238e-07], [3.95531e-05, 1.26670e-05, 1.34310e-05, 6.48547e-06, 5.77475e-06, 7.91706e-07]])    
err_Aird_XLF_low = itertools.cycle([[9.07020e-06, 3.97898e-06, 3.33015e-06, 2.05293e-06, 5.67818e-07], [2.72160e-05, 6.71777e-06, 4.01875e-06, 1.94785e-06, 1.54134e-06, 4.77435e-07]])
err_Aird_XLF_high = itertools.cycle([[1.30624e-05, 5.63148e-06, 4.42298e-06, 3.02844e-06, 1.61451e-06], [6.17042e-05, 1.21852e-05, 5.51823e-06, 2.67804e-06, 2.04131e-06, 9.55148e-07]])
# Color preselected sample
Aird_cl = itertools.cycle([[42.8125, 43.1875, 43.5625, 43.9375, 44.3125, 44.6875], [42.8125, 43.1875, 43.5625, 43.9375, 44.3125, 44.6875, 45.0625]], )
e_Aird_cl_low = itertools.cycle([[42.6250,43.00,43.3750,43.7500,44.1250,44.5000],[42.6250,43.0000,43.3750,43.7500,44.1250,44.5000,44.8750]])
e_Aird_cl_high = itertools.cycle([[43.00,43.3750,43.7500,44.1250,44.5000,44.8750],[43.0000,43.3750,43.7500,44.1250,44.5000,44.8750,45.2500]])

Aird_cXLF = itertools.cycle([[ 6.41569e-05, 3.63649e-05, 1.79293e-05, 1.28719e-05, 6.39078e-06, 5.69329e-06], [0.000173090,6.31526e-05,2.41756e-05,1.80302e-05,7.27135e-06,5.57555e-06,6.12555e-07]])    
err_Aird_cXLF_low = itertools.cycle([[4.06788e-05, 1.25321e-05, 5.33435e-06, 3.89764e-06, 2.38806e-06, 1.65616e-06], [7.81118e-05,1.94726e-05,5.17967e-06, 3.24222e-06,1.85414e-06,1.33784e-06,3.16868e-07]])
err_Aird_cXLF_high = itertools.cycle([[8.50483e-05, 1.81222e-05, 7.31074e-06, 5.37364e-06, 3.57183e-06, 2.25292e-06], [0.000128217,2.70153e-05,6.47448e-06,3.90504e-06,2.42338e-06,1.71984e-06,5.65123e-07]])

########################################################
# Data
########################################################
Lx = linspace(42.0, 46.0)
redshift = [2.25, 3.0]
import matplotlib.pyplot as plt
x_plots = 2
y_plots = 1
fig = plt.figure(figsize=(15,8))
fig.subplots_adjust(left=0.10, right=0.97, top=0.86,wspace=0.34, hspace=0.15)


work = ["Fotopoulou"]
# MLE
L0 = 43.8537
g1 = 0.94556
g2 = 2.4729
p1 = 5.0857
p2 = -3.59788
zc = 2.408
La = 44.31137
a = 0.3097
Normal = -6.0655
params = [44.63,1.40,3.4,4.89,-3.8,2.03,43.85,0.55,-7.45]
matrix = [[ 0.017204289068638900,  0.007268401044233920,  0.046185986681261300,  0.008831151855127620,  0.008008659441831710,-0.005135914323722490,-0.003778033052427710, 0.003122590220226560,-0.036283372340088000],
          [ 0.007268401044233920,  0.005391987767748420,  0.017571084612313300,  0.010893110974085300,  0.011146954214315000,-0.003559319878384610,-0.001401343390654160, 0.002444412730958810,-0.018723345258687800],
          [ 0.046185986681261300,  0.017571084612313300,  0.200244321108370000,  0.012514222127205800,  0.012927883050572700,-0.010699134782714200,-0.010136807001380100, 0.007321112098157700,-0.094358019621848000],
          [ 0.008831151855127620,  0.010893110974085300,  0.012514222127205800,  0.077623449963435100,  0.073131067272393700,-0.029594608551928500,-0.000964745158397721, 0.001547531655523990,-0.038512775751049300],
          [ 0.008008659441831710,  0.011146954214315000,  0.012927883050572700,  0.073131067272393700,  0.366839901298749000,-0.051584742233182700,-0.003114831722374510, 0.014118126335858600,-0.030115259746022400],
          [-0.005135914323722490, -0.003559319878384610, -0.010699134782714200, -0.029594608551928500, -0.051584742233182700, 0.019373735214431500, 0.002673030793697170,-0.001433710618131040, 0.015528238728249800],
          [-0.003778033052427710, -0.001401343390654160, -0.010136807001380100, -0.000964745158397721, -0.003114831722374510, 0.002673030793697170, 0.003382802725850100,-0.002044858541149150, 0.007497880541595920],
          [ 0.003122590220226560,  0.002444412730958810,  0.007321112098157700,  0.001547531655523990,  0.014118126335858600,-0.001433710618131040,-0.002044858541149150, 0.003332825680025220,-0.007109608590661420],
          [-0.036283372340088000, -0.018723345258687800, -0.094358019621848000, -0.038512775751049300, -0.030115259746022400, 0.015528238728249800, 0.007497880541595920,-0.007109608590661420, 0.084079017987664700]]

#    3-sigma area
sigma = 3.0
values = np.random.multivariate_normal(params,matrix, 10000) # draw random values using the covariance matrix
#print values
for z in redshift:
    Phi_low = []
    Phi_high = []
    for L in Lx:
        #print z, L
        LF = []
        for i in range( 0, len(values) ):
            L0,g1,g2,p1,p2,zc,La,a,Normal = values[i, :]
            LF.append( log10(model.Fotopoulou(L,z,L0+0.3098,g1,g2,p1,p2,zc,La+0.3098,a)*power(10.0, Normal) )) 
        Phi_low.append(np.mean(LF)-sigma*np.std(LF))
        Phi_high.append(np.mean(LF)+sigma*np.std(LF))
        #print np.mean(LF),np.std(LF)
#for z in redshift :
    ax = fig.add_subplot(y_plots, x_plots, redshift.index(z)+1)
    plt.fill_between(Lx,Phi_low,Phi_high,color='gray',alpha=0.5)

    Phi = log10(model.Fotopoulou(Lx,z,L0+0.3098,g1,g2,p1,p2,zc,La+0.3098,a)*power(10.0, Normal))
    plt.plot(Lx,Phi,ls='-',color='k', label="this work")

# Convert to lorarithmic errorbars
    LFX_ll = np.asarray(Aird_l.next())
    e_L_low = LFX_ll-np.asarray(e_Aird_low.next())
    e_L_high = np.asarray(e_Aird_high.next())-LFX_ll
    xlf = np.asarray(Aird_XLF.next())
    LFX = log10(xlf)
    e_high = 0.434*np.asarray(err_Aird_XLF_high.next())/(xlf)
    e_low = 0.434*np.asarray(err_Aird_XLF_low.next())/(xlf)
    plt.errorbar(LFX_ll, LFX, xerr=[e_L_low,e_L_high], yerr=[e_high, e_low],marker='o', ls=" ",ecolor='k',markeredgecolor='k',markerfacecolor='white',markersize=14, label="X-ray detection")

    LFX_ll = np.asarray(Aird_cl.next())
    e_L_low = LFX_ll-np.asarray(e_Aird_cl_low.next())
    e_L_high = np.asarray(e_Aird_cl_high.next())-LFX_ll
    xlf = np.asarray(Aird_cXLF.next())
    LFX = log10(xlf)
    e_high = 0.434*np.asarray(err_Aird_cXLF_high.next())/(xlf)
    e_low = 0.434*np.asarray(err_Aird_cXLF_low.next())/(xlf)
    plt.errorbar(LFX_ll, LFX, xerr=[e_L_low,e_L_high], yerr=[e_high, e_low],marker='^', ls=" ",markeredgecolor='k',markerfacecolor='red',markersize=14,label="Color pre-selected")

    plt.xlabel('Luminosity', fontsize='x-large')
    plt.ylabel(r'd$\Phi$/dlogLx', fontsize='x-large')
   
    ax.annotate("z="+str(round(z,2)), (0.7, 0.85) , xycoords='axes fraction', fontstyle='oblique', fontsize='medium', )
    plt.xlim([42, 46])
    plt.ylim([-8.2,-2.8])
    plt.xticks([42, 43, 44, 45, 46])
    plt.yticks([ -8, -7, -6, -5, -4, -3])
    #plt.yscale('log') 
    ax.fill([42.0, 43.55,43.55,42.0],[-8.2,-8.2,-2.8,-2.8], fill=False, hatch='\\', color='gray')
 
    plt.legend(loc=3)#bbox_to_anchor=(-2.30, 2.4, 3., 0.1,), loc=2, ncol=3, mode="expand", borderaxespad=0.)
#plt.draw()
for ext in ['jpg','pdf','eps','png','svg']:
    plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/eAird_comparison_new_fit.'+ext)
plt.show()
