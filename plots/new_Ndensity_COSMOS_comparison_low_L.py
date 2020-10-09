import sys
# Add the module path to the sys.path list
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/LF_modules/models')
#
import numpy as np
import time, itertools
from numpy import hstack,vstack,arange,savetxt, linspace,logspace,sqrt,log10,array,ones,tile, median, asarray, genfromtxt, power, column_stack
from Source import *
import matplotlib.pyplot as plt
import scipy.integrate
from LFunctions import *
from scipy.integrate import simps
from AGN_LF_config import LF_config


linecolor = (0., 0., 0.)
fillcolor = (0.75, 0.75, 0.75)
pointcolor = (0., 0., 0.)
linestyles = itertools.cycle( ['-', '--',':','-.','steps'] )
colors = itertools.cycle( ['k', 'k','k','k','gray'] )

zlabel = []
save_data = []

fig_size = [10, 10]
fig = plt.figure(figsize=fig_size)
fig.subplots_adjust(left=0.175,  right=0.95, bottom=0.10, top=0.95, wspace=0.0, hspace=0.0)
Lbin = [43.56, 44.15]
zspace = linspace(3, 5 ,20)


z_COSMOS = [3.09, 3.29, 3.45]
z_COSMOS_err = [0.09, 0.11, 0.05]
N_COSMOS = [log10(5.63e-06), log10(3.56e-06), log10(1.04e-05)]
N_COSMOS_err = [0.434*1.86e-06/(5.63e-06), 0.434*1.28e-06/(3.56e-06), 0.434*4.86e-6/(1.04e-05)]

z_ikeda = [4]
N_ikeda = [ log10(1.3e-6)]
N_ikeda_err = [0.434*0.6e-06/(1.3e-06)]

z_glikman = [4]
N_glikman = [log10(4.6e-6)]
N_glikman_err = [0.434*2.0e-06/(4.6e-06)]

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

#    3-sigma area
sigma = 3.0
draws = np.random.multivariate_normal(parameters,matrix, 10000) # draw random values using the covariance matrix
   
class Params():
    pass

params = Params()
   
for i in xrange(0, len(Lbin)-1):
    name = '/home/sotiria/workspace/Luminosity_Function/output_files/plots/Ndensity_lowL'+str(Lbin[i])+'.dat'    
    values = hstack(( zspace ))
    #print Lbin[i]
    ll = linspace(Lbin[i], Lbin[i+1], 10)
    #print ll
    min_Nmodel = []
    max_Nmodel = []
    for z in zspace:
        Nmodel = []
        for k in arange(0, len(draws[:,0])):
            params.L0, params.g1, params.g2, params.p1, params.p2, params.zc, params.La, params.a, params.Norm = draws[k, :]
            params.L0 = params.L0 + 0.3098
            params.La = params.La + 0.3098 
            LF_model = []
            for lx in ll:
                PF = Fotopoulou(lx, z, params)
                LF_model.append(PF)
            integral = log10( simps(LF_model,ll) )
            Nmodel.append(integral)
        #print Nmodel
        min_Nmodel.append( np.mean( asarray(Nmodel) ) - sigma*np.std( asarray(Nmodel) ) )
        max_Nmodel.append( np.mean( asarray(Nmodel) ) + sigma*np.std( asarray(Nmodel) ) )

    values = column_stack((values, min_Nmodel , max_Nmodel))
    savetxt(name, values)
    plt.fill_between(zspace, min_Nmodel, max_Nmodel, color=fillcolor, label=str(Lbin[i]), alpha = 0.2)

    params.L0 = L0 + 0.3098
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.zc = zc
    params.La = La + 0.3098 
    params.a = a
    params.Norm = Norm
    
    mle_Nmodel = []
    for z in zspace:
        LF_model = []
        for lx in ll:
            PF = Fotopoulou(lx, z, params)
            LF_model.append(PF)
        integral = log10(simps(LF_model, ll))
        mle_Nmodel.append(integral)
    plt.plot(zspace, mle_Nmodel, color=colors.next(), ls=linestyles.next(), lw=3, label=str(Lbin[i])+'$<logL_x<$'+str(Lbin[i+1]) )       
#plt.yscale('log')
plt.errorbar(z_COSMOS, N_COSMOS, xerr= z_COSMOS_err, yerr = N_COSMOS_err, marker='o', color='r', ls = ' ', markersize=14)
plt.errorbar(z_ikeda, N_ikeda,  yerr = N_ikeda_err, marker='s', markerfacecolor='w', markeredgecolor='black',ecolor='k',ls=' ',markersize=14)
plt.errorbar(z_glikman, N_glikman, yerr = N_glikman_err, marker='s', color='black', ls=' ',markersize=14)


plt.legend(loc=4)
plt.xlim([3,4.5])
plt.ylim([-7.0,-4.0])
plt.xlabel('$Redshift$', fontsize='x-large')
plt.ylabel('$Number\,Density\,(Mpc^{-3}$)', fontsize='x-large')
plt.draw()
for extension in ['.eps', '.jpg', '.pdf', '.png']:
    plt.savefig('/home/sotiria/workspace/Luminosity_Function/output_files/plots/Ndensity_COSMOS_comparison_43.56_44.15'+extension)
plt.show()