import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import itertools
import sys
import copy
# Add the ptdraft folder path to the sys.path list
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/models')
from AGN_LF_config import LF_config
#import astroML as ML

from LFunctions import Ueda14, LADE, Miyaji15, highz, Fotopoulou
########################################################
class Params: pass

Ueda_params = Params()

Ueda_params.L0 = 43.97
Ueda_params.g1 = 0.96
Ueda_params.g2 = 2.71
Ueda_params.p1 = 4.78
Ueda_params.beta = 0.84
Ueda_params.Lp = 44.0
Ueda_params.p2 = -1.5
Ueda_params.p3 = -6.2
Ueda_params.zc1 = 1.86
Ueda_params.La1 = 44.61
Ueda_params.a1 = 0.29
Ueda_params.zc2 = 3.0
Ueda_params.La2 = 44.0
Ueda_params.a2 = -0.1
Ueda_params.Norm = np.log10(2.91e-6)

Ueda_parameters = [Ueda_params.L0, Ueda_params.g1, Ueda_params.g2, Ueda_params.p1, Ueda_params.p2, Ueda_params.zc1, Ueda_params.La1, Ueda_params.a1, Ueda_params.Norm]
Ueda_min = [0.06, 0.04, 0.09, 0.16, 0.0, 0.07, 0.07, 0.02, 0.07]
Ueda_max = [0.06, 0.04, 0.09, 0.16, 0.0, 0.07, 0.07, 0.02, 0.07]

##############################
XMM_params = Params()

XMM_params.L0 = 43.97+0.3098
XMM_params.g2 = 2.53
XMM_params.g1 = 0.97
XMM_params.p1 = 5.72
XMM_params.p2 = -2.72
XMM_params.zc = 2.19
XMM_params.La = 44.55+0.3098
XMM_params.a = 0.26
XMM_params.Norm = -6.26

XMM_parameters = [XMM_params.L0, XMM_params.g1, XMM_params.g2, XMM_params.p1, XMM_params.p2, XMM_params.zc, XMM_params.La, XMM_params.a, XMM_params.Norm]
XMM_min = [0.19, 0.07, 0.22, 0.51, 1.14, 0.35, 0.34, 0.05, 0.28]
XMM_max = [0.19, 0.07, 0.22, 0.51, 1.14, 0.35, 0.34, 0.05, 0.28]

Chandra_params = Params()

Chandra_params.L0 = 43.62+0.3098
Chandra_params.g2 = 2.46
Chandra_params.g1 = 0.92
Chandra_params.p1 = 7.02
Chandra_params.p2 = -1.81
Chandra_params.zc = 1.99
Chandra_params.La = 44.48+0.3098
Chandra_params.a = 0.26
Chandra_params.Norm = -6.04

Chandra_parameters = [Chandra_params.L0, Chandra_params.g1, Chandra_params.g2, Chandra_params.p1, Chandra_params.p2, Chandra_params.zc, Chandra_params.La, Chandra_params.a, Chandra_params.Norm]
Chandra_min = [0.14, 0.09, 0.17, 0.93, 0.56, 0.21, 0.10, 0.03, 0.24]
Chandra_max = [0.14, 0.09, 0.17, 0.93, 0.56, 0.21, 0.10, 0.03, 0.24]

MAXI_XMM_params = Params()

MAXI_XMM_params.L0 = 44.04+0.3098
MAXI_XMM_params.g2 = 2.65
MAXI_XMM_params.g1 = 0.99
MAXI_XMM_params.p1 = 5.72
MAXI_XMM_params.p2 = -2.72
MAXI_XMM_params.zc = 2.28
MAXI_XMM_params.La = 44.68+0.3098
MAXI_XMM_params.a = 0.24
MAXI_XMM_params.Norm = -6.38

MAXI_XMM_parameters = [MAXI_XMM_params.L0, MAXI_XMM_params.g1, MAXI_XMM_params.g2, MAXI_XMM_params.p1, MAXI_XMM_params.p2, MAXI_XMM_params.zc, MAXI_XMM_params.La, MAXI_XMM_params.a, MAXI_XMM_params.Norm]
MAXI_XMM_min = [0.15, 0.07, 0.22, 0.36, 1.07, 0.36, 0.34, 0.04, 0.24]
MAXI_XMM_max = [0.15, 0.07, 0.22, 0.36, 1.07, 0.36, 0.34, 0.04, 0.24]

MAXI_Chandra_params = Params()

MAXI_Chandra_params.L0 = 43.77+0.3098
MAXI_Chandra_params.g2 = 2.43
MAXI_Chandra_params.g1 = 0.86
MAXI_Chandra_params.p1 = 5.92
MAXI_Chandra_params.p2 = -2.19
MAXI_Chandra_params.zc = 2.15
MAXI_Chandra_params.La = 44.53+0.3098
MAXI_Chandra_params.a = 0.24
MAXI_Chandra_params.Norm = -5.98

MAXI_Chandra_parameters = [MAXI_Chandra_params.L0, MAXI_Chandra_params.g1, MAXI_Chandra_params.g2, MAXI_Chandra_params.p1, MAXI_Chandra_params.p2, MAXI_Chandra_params.zc, MAXI_Chandra_params.La, MAXI_Chandra_params.a, MAXI_Chandra_params.Norm]
MAXI_Chandra_min = [0.13, 0.06, 0.16, 0.38, 0.56, 0.25, 0.17, 0.02, 0.20]
MAXI_Chandra_max = [0.13, 0.06, 0.16, 0.38, 0.56, 0.25, 0.17, 0.02, 0.20]

XMM_Chandra_params = Params()

XMM_Chandra_params.L0 = 43.72+0.3098
XMM_Chandra_params.g2 = 2.37
XMM_Chandra_params.g1 = 0.86
XMM_Chandra_params.p1 = 6.03
XMM_Chandra_params.p2 = -2.19
XMM_Chandra_params.zc = 2.08
XMM_Chandra_params.La = 44.49+0.3098
XMM_Chandra_params.a = 0.25
XMM_Chandra_params.Norm = -5.92

XMM_Chandra_parameters = [XMM_Chandra_params.L0, XMM_Chandra_params.g1, XMM_Chandra_params.g2, XMM_Chandra_params.p1, XMM_Chandra_params.p2, XMM_Chandra_params.zc, XMM_Chandra_params.La, XMM_Chandra_params.a, XMM_Chandra_params.Norm]
XMM_Chandra_min = [0.12, 0.06, 0.11, 0.45, 0.54, 0.17, 0.11, 0.02, 0.18]
XMM_Chandra_max = [0.12, 0.06, 0.11, 0.45, 0.54, 0.17, 0.11, 0.02, 0.18]

MAXI_XMM_Chandra_params = Params()

MAXI_XMM_Chandra_params.L0 = 43.77+0.3098
MAXI_XMM_Chandra_params.g2 = 2.40
MAXI_XMM_Chandra_params.g1 = 0.87
MAXI_XMM_Chandra_params.p1 = 5.89
MAXI_XMM_Chandra_params.p2 = -2.30
MAXI_XMM_Chandra_params.zc = 2.12
MAXI_XMM_Chandra_params.La = 44.51+0.3098
MAXI_XMM_Chandra_params.a = 0.24
MAXI_XMM_Chandra_params.Norm = -5.97

MAXI_XMM_Chandra_parameters = [MAXI_XMM_Chandra_params.L0, MAXI_XMM_Chandra_params.g1, MAXI_XMM_Chandra_params.g2, MAXI_XMM_Chandra_params.p1, MAXI_XMM_Chandra_params.p2, MAXI_XMM_Chandra_params.zc, MAXI_XMM_Chandra_params.La, MAXI_XMM_Chandra_params.a, MAXI_XMM_Chandra_params.Norm]
MAXI_XMM_Chandra_min = [0.11, 0.06, 0.11, 0.31, 0.50, 0.16, 0.11, 0.02, 0.17]
MAXI_XMM_Chandra_max = [0.11, 0.06, 0.11, 0.31, 0.50, 0.16, 0.11, 0.02, 0.17]

####################################

Aird_params = Params()

Aird_params.L0 = 44.09
Aird_params.g1 = 0.73
Aird_params.g2 = 2.22
Aird_params.p1 = 4.34
Aird_params.beta = -0.19
Aird_params.Lp = 44.48
Aird_params.p2 = -0.30
Aird_params.p3 = -7.33
Aird_params.zc1 = 1.85
Aird_params.La1 = 44.78
Aird_params.a1 = 0.23
Aird_params.zc2 = 3.16
Aird_params.La2 = 44.46
Aird_params.a2 = 0.13
Aird_params.Norm = -5.72

Aird_parameters = [Aird_params.L0, Aird_params.g1, Aird_params.g2, Aird_params.p1, Aird_params.p2, Aird_params.zc1, Aird_params.La1, Aird_params.a1, Aird_params.Norm]
Aird_min = [0.05, 0.02, 0.06, 0.18, 0.13, 0.08, 0.07, 0.01, 0.07]
Aird_max = [0.05, 0.02, 0.06, 0.18, 0.13, 0.08, 0.07, 0.01, 0.07]

Miyaji_params = Params()

Miyaji_params.L0 = 44.04
Miyaji_params.g1 = 1.17
Miyaji_params.g2 = 2.80
Miyaji_params.p1 = 5.29
Miyaji_params.p2 = -0.35
Miyaji_params.p3 = -5.6
Miyaji_params.zb0 = 1.1
Miyaji_params.zb2 = 2.7
Miyaji_params.Lb = 44.5
Miyaji_params.a = 0.18
Miyaji_params.b1 = 1.2
Miyaji_params.b2 = 1.5
Miyaji_params.Norm = np.log10(1.56e-6)

Miyaji_parameters = [Miyaji_params.L0, Miyaji_params.g1, Miyaji_params.g2, Miyaji_params.p1, Miyaji_params.p2, Miyaji_params.zb0, Miyaji_params.Lb, Miyaji_params.a, Miyaji_params.Norm]
Miyaji_min = [0.11, 0.04, 0.08, 0.11, 0.13, 0.0, 0.0, 0.02, 0.0]
Miyaji_max = [0.11, 0.04, 0.08, 0.11, 0.13, 0.0, 0.0, 0.01, 0.0]

Vito_params = Params()

Vito_params.L0 = np.log10(4.92e44)
Vito_params.g1 = 0.66
Vito_params.g2 = 3.71
Vito_params.q = -6.65
Vito_params.b = 2.40
Vito_params.Norm = np.log10(1.19e-5)

Vito_parameters = [Vito_params.L0, Vito_params.g1, Vito_params.g2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]


Georgakakis_params = Params()

Georgakakis_params.L0 = 44.31
Georgakakis_params.g1 = 0.21
Georgakakis_params.g2 = 2.15
Georgakakis_params.q = -7.46
Georgakakis_params.b = 2.30
Georgakakis_params.Norm = -4.79

Georgakakis_parameters = [Georgakakis_params.L0, Georgakakis_params.g1, Georgakakis_params.g2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
#

chains_path = '/home/Sotiria/Dropbox/transfer/XLF_output_files/combination_fields/All_Coherent/'

#parameters = np.loadtxt( chains_path + '/1-.txt')
#params_in = parameters[:, 2:]

parameters = np.loadtxt( chains_path + '/1-.txt')
print parameters[:,2:]
factor = np.zeros_like(parameters[:,2:])
factor[:,0] = factor[:,0] + 0.3098
factor[:,6] = factor[:,6] + 0.3098

params_in = parameters[:, 2:] + factor 


n_params = len(params_in[0,:])
labels = ["$L_0$", "$\gamma_1$", "$\gamma_2$", "$p_1$", "$p_2$", "$z_c$", "$L_a$", "$a$", "$Norm$"]
min_range = [43, 0,   1.5, 2.0, -6, 0, 44, 0,   -8.2]
max_range = [46, 1.5, 3.5, 8.5,  2, 4, 45.5, 0.5, -4 ]
fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(left=0.06, top=0.96, bottom=0.06, right=0.92, wspace=0.00, hspace=0.00)   

for i in range(0, n_params+1):
    for j in range(i+1, n_params):
        
        ax = fig.add_subplot(n_params-1, n_params-1, (n_params-1)*i+j)
        plt.plot(params_in[::3,j],params_in[::3,i],'.',color='black',markersize=0.35)
        
        plt.errorbar(Ueda_parameters[j],Ueda_parameters[i],xerr=Ueda_min[j], yerr=Ueda_max[i], marker='s',color='blue',markersize=9)
        plt.errorbar(Aird_parameters[j],Aird_parameters[i],xerr=Aird_min[j], yerr=Aird_max[i], marker='s',color='red',markersize=9)
        plt.errorbar(Miyaji_parameters[j],Miyaji_parameters[i],xerr=Miyaji_min[j], yerr=Miyaji_max[i], marker='s',color='green',markersize=9)
        
        
        #plt.errorbar(XMM_parameters[j],XMM_parameters[i],xerr=XMM_min[j], yerr=XMM_max[i], marker='o',color='blue',markersize=5)
        #plt.errorbar(Chandra_parameters[j],Chandra_parameters[i],xerr=Chandra_min[j], yerr=Chandra_max[i], marker='o',color='red',markersize=5)
        
        #plt.errorbar(MAXI_XMM_parameters[j],MAXI_XMM_parameters[i],xerr=MAXI_XMM_min[j], yerr=MAXI_XMM_max[i], marker='o',color='blue',markersize=7)
        #plt.errorbar(MAXI_Chandra_parameters[j],MAXI_Chandra_parameters[i],xerr=MAXI_Chandra_min[j], yerr=MAXI_Chandra_max[i], marker='o',color='red',markersize=7)
        
        #plt.errorbar(XMM_Chandra_parameters[j],XMM_Chandra_parameters[i],xerr=XMM_Chandra_min[j], yerr=XMM_Chandra_max[i], marker='o',color='blue',markersize=9)
        #plt.errorbar(MAXI_XMM_Chandra_parameters[j],MAXI_XMM_Chandra_parameters[i],xerr=MAXI_XMM_Chandra_min[j], yerr=MAXI_XMM_Chandra_max[i], marker='o',color='red',markersize=9)
        
        
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        plt.xlim([min_range[j], max_range[j]])
        plt.ylim([min_range[i], max_range[i]])  
        if j == n_params-1 : 
            yvis = True
        else:
            yvis=False
        ax.yaxis.tick_right()    
        ax.yaxis.set_ticks_position('both')
        plt.yticks(np.round(np.linspace(min_range[i], max_range[i],5),2)[1:-1:2],fontsize=12.5, visible=yvis)

        if i == 0 and j == i+1:
            plt.ylabel(labels[i],rotation=0,labelpad=25)
        if j == i+1:    
            
            plt.xlabel(labels[j],labelpad=20)
            
        if i == 0 : 
            xvis = True
        else:
            xvis = False
        
        ax.xaxis.tick_top()    
        ax.xaxis.set_ticks_position("both")
        
        plt.xticks(np.round(np.linspace(min_range[j], max_range[j],5),2)[1:-1:2],fontsize=12.5, visible=xvis)
                
  

#        if j==n_params:
#            ax.yaxis.tick_right()
#
#        else:
    
#plt.xlabel(labels[j])
plt.savefig('chains.pdf')
plt.show()

    
    
