import sys
# Add the module path to the sys.path list
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/LF_modules/models')
#
import time, itertools
from numpy import hstack,vstack,arange,savetxt, linspace,logspace,sqrt,log10,array,ones,tile, median, asarray, genfromtxt, power, column_stack
from Source import *
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.integrate import simps
from SetUp_data import Set_up_data
from LFunctions import *

setup_data = Set_up_data()
data_in = setup_data.get_data()[0]
   

zbin = [0.01, 0.2, 0.5, 0.7, 1.2, 1.7, 2.0, 3.0, 4.0]
#Lbin = [41.25, 42.0, 42.5, 43.0, 43.25, 43.50, 44.0, 44.65]               
Lbin = [42.0, 43.0, 44.0, 45.0, 46.0]
linecolor = (0., 0., 0.)
fillcolor = (0.75, 0.75, 0.75)
pointcolor = (0., 0., 0.)
linestyles = itertools.cycle( ['-',':','-.','steps'] )
colors = itertools.cycle( ['k', 'k','k','gray'] )
markers = itertools.cycle( ['o', 's', '^' , 'v'] )
markerface = itertools.cycle( ['k', 'w', 'k','gray'] )
ecolors = itertools.cycle( ['k', 'k','k','gray'] )

zlabel = []
save_data = []
fig_size = [10, 10]
fig = plt.figure(figsize=fig_size)
fig.subplots_adjust(left=0.175,  right=0.95, bottom=0.10, top=0.95, wspace=0.0, hspace=0.0)
zspace = linspace(0,4.1,20)


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
import numpy as np
draws = np.random.multivariate_normal(parameters,matrix, 10000) # draw random values using the covariance matrix

class Params():
    pass
params = Params()
from Source import *
for i in xrange(0, len(Lbin)-1):
    name = '/home/sotiria/workspace/Luminosity_Function/output_files/Ndensity_'+str(Lbin[i])+'.dat'    
    values = hstack(( zspace ))
    print Lbin[i]
    ll = linspace(Lbin[i], Lbin[i+1], 6)
    min_Nmodel = []
    max_Nmodel = []
    
    for z in zspace:
        Nmodel = []
        for k in arange(0, len(draws[:,0])):
            params.L0, params.g1, params.g2, params.p1, params.p2, params.zc, params.La, params.a, params.Norm = draws[k,:]
            LF_model = []
            for lx in ll:
                PF = Fotopoulou(lx, z, params)
                LF_model.append(PF)
            integral = log10( simps(LF_model, ll) )
            Nmodel.append(integral)
        min_Nmodel.append( np.mean( asarray(Nmodel) ) - sigma*np.std( asarray(Nmodel) ) )
        max_Nmodel.append( np.mean( asarray(Nmodel) ) + sigma*np.std( asarray(Nmodel) ) )
    
    values = column_stack((values, min_Nmodel , max_Nmodel))
    savetxt(name, values)
    plt.fill_between(zspace, min_Nmodel, max_Nmodel, color=fillcolor, label=str(Lbin[i]), alpha = 0.2)
    
    mle_Nmodel = []
    
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.zc = zc
    params.La = La
    params.a = a
    params.Norm = Norm
    
    for z in zspace:
        LF_model = []
        for lx in ll:
            PF = Fotopoulou(lx, z, params)
            LF_model.append(PF)
        integral = log10( simps(LF_model, ll) )
        mle_Nmodel.append(integral)
    plt.plot(zspace, mle_Nmodel, color=colors.next(), ls=linestyles.next(), lw=3, label=str(Lbin[i])+'$<logL_x<$'+str(Lbin[i+1]) )       

    data = []
    for j in arange(0,len(zbin)-1):
        count = 0
        dN_dV = 0.0
        err_dN_dV = 0.0
        Ll = []
        Zz = []
        for source in data_in:
            z = source.z
            Lx = source.l
            if zbin[j] <= z < zbin[j+1] and Lbin[i] <= Lx < Lbin[i+1]:
                count = count + 1
                Vmax =  get_V(Lx, zbin[j], zbin[j+1], 0.01) 
                dN_dV = dN_dV + 1.0/Vmax
                err_dN_dV = err_dN_dV + power( (1.0/Vmax), 2.0)
                
                Ll.append(Lx)        
                Zz.append(z)
        err_dN_dV = sqrt( err_dN_dV )
        if count>0:
            print count
            #print median(Zz), median(Ll), count, log10(dN_dV), 0.434*err_dN_dV/dN_dV, median(Ll)-Lbin[i], Lbin[i+1]-median(Ll)
            datum = [median(Zz), median(Ll), count, log10(dN_dV), 0.434*err_dN_dV/dN_dV, median(Ll)-Lbin[i], Lbin[i+1]-median(Ll), median(Zz)-zbin[j], zbin[j+1]-median(Zz)]
            data.append(datum)
    data = array(data)        
    
    redshift = data[:,0]
    e_red_l = data[:,7]
    e_red_h = data[:,8]
    
    Number_density = data[:,3]
    err_Number_density = data[:,4]
    face = markerface.next() 
    plt.errorbar(redshift, Number_density, yerr = err_Number_density, xerr=[e_red_l, e_red_h], ls= ' ', marker=markers.next(), markersize=14, markerfacecolor=face, ecolor=ecolors.next())
#plt.yscale('log')
plt.legend(loc=4)
plt.xlim([0,4.1])
plt.ylim([-11.0,-3.0])
plt.xlabel('$Redshift$', fontsize='x-large')
plt.ylabel('$Number\,Density\,(Mpc^{-3}$)', fontsize='x-large')
plt.draw()
for extension in ['.eps', '.jpg', '.pdf', '.png']:
    plt.savefig('/home/sotiria/workspace/Luminosity_Function/output_files/plots/Ndensity'+extension)
plt.show()