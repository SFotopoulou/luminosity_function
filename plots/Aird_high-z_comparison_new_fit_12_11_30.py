import warnings
import sys
# Append the module path to the sys.path list
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/LF_modules/configuration/')
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/LF_modules/models/')
import time
import math   
from numpy import array,log,sqrt,power,linspace,vectorize,ones,tile,sum, log10, savetxt, genfromtxt,loadtxt,isnan,min,max
from Source import *
from scipy.integrate import simps
import minuit
import matplotlib.pyplot as plt
import itertools
from AGN_LF_config import LF_config
import numpy as np
from matplotlib.transforms import Bbox
from matplotlib.path import Path
from matplotlib.patches import Rectangle
from LFunctions import *

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
p2 = -1.34853227641
a = 0.260307788878
p1 = 5.8361720265
g2 = 2.55460679566
g1 = 1.03913647028
La = 44.3258637967
zc = 1.87730303823
L0 = 43.9402810729
Norm = -6.34232103471

parameters = [43.9403, 1.0391, 2.5546, 5.83617, -1.3485, 1.8773, 44.32586, 0.26031, -6.342321]

matrix = [[ 0.027414483358256400, 0.009185811147251570,  0.024585585436562200, -0.025438119503221600, -0.027392420918882700,  0.007192235482021060,   0.002963160876719450, -0.002909163707161460,-0.041877891411498200],
          [0.009185811147251570,  0.006546785883021330,  0.008425511473072540,  0.003897095265996940,  0.002554832567327960, -0.001369163682634570,  -0.000723386505863384,  0.000095508653477705,-0.017369968324576700],
          [0.024585585436562200,  0.008425511473072540,  0.030818578707093500, -0.014678165197323700, -0.014910924910161700,  0.008228073619160770,   0.007683225069501030, -0.002233582892853550,-0.038059997041001100],
          [-0.025438119503221600, 0.003897095265996940, -0.014678165197323700,  0.155518507775386000,  0.151164555001805000, -0.057815279141344300,  -0.004301123991898550,  0.004649599528615900, 0.013981354317924200],
          [-0.027392420918882700, 0.002554832567327960, -0.014910924910161700,  0.151164555001805000,  0.324456001631511000, -0.081635399889806100,  -0.002211087030002840,  0.007139118187159160, 0.026655099680892600],
          [0.007192235482021060, -0.001369163682634570,  0.008228073619160770, -0.057815279141344300, -0.081635399889806100,  0.035496658032260300,   0.007746782469778650, -0.001329407316093770,-0.003593660623643860],
          [0.002963160876719450, -0.000723386505863384,  0.007683225069501030, -0.004301123991898550, -0.002211087030002840,  0.007746782469778650,   0.010339733433261000, -0.001343902504603640,-0.003176008379971750],
          [-0.002909163707161460, 0.000095508653477705, -0.002233582892853550,  0.004649599528615900,  0.007139118187159160, -0.001329407316093770,  -0.001343902504603640,  0.001019416244804420, 0.003972578046047490],
          [-0.041877891411498200,-0.017369968324576700, -0.038059997041001100,  0.013981354317924200,  0.026655099680892600, -0.003593660623643860,  -0.003176008379971750,  0.003972578046047490, 0.070355261229252000]]

class Params():
    pass

params = Params()
#    3-sigma area
sigma = 3.0
values = np.random.multivariate_normal(parameters,matrix, 10000) # draw random values using the covariance matrix
#print values
for z in redshift:
    Phi_low = []
    Phi_high = []
    for L in Lx:
        #print z, L
        LF = []
        for i in range( 0, len(values) ):
            params.L0, params.g1, params.g2, params.p1, params.p2, params.zc, params.La, params.a, params.Norm = values[i, :]
            params.L0 = params.L0 + 0.3098
            params.La = params.La + 0.3098 
            LF.append( log10(Fotopoulou(L, z, params ))) 
        Phi_low.append(np.mean(LF)-sigma*np.std(LF))
        Phi_high.append(np.mean(LF)+sigma*np.std(LF))
        #print np.mean(LF),np.std(LF)
#for z in redshift :
    ax = fig.add_subplot(y_plots, x_plots, redshift.index(z)+1)
    plt.fill_between(Lx,Phi_low,Phi_high,color='gray',alpha=0.5)
    params.L0 = L0 + 0.3098
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.zc = zc
    params.La = La + 0.3098
    params.a = a
    params.Norm = Norm
    Phi = log10(Fotopoulou(Lx, z, params))
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
    plt.savefig('/home/sotiria/workspace/Luminosity_Function/output_files/plots/Aird_comparison_LDDE_fit.'+ext)
plt.show()
