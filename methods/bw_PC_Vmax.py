import sys
# Add the module path to the sys.path list
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/LF_modules/models')
#
import time, itertools
from numpy import arange,savetxt, linspace,logspace,sqrt,log10,array,ones,tile, median, asarray, genfromtxt, power
import numpy as np
from Source import *
import matplotlib.pyplot as plt
import scipy.integrate
from LFunctions import *
from scipy.integrate import simps
from AGN_LF_config import LF_config

zmin = LF_config.zmin
zmax = LF_config.zmax
Lmin = LF_config.Lmin
Lmax = LF_config.Lmax



#p2 = -1.34853227641
#a = 0.260307788878
#p1 = 5.8361720265
#g2 = 2.55460679566
#g1 = 1.03913647028
#La = 44.3258637967
#zc = 1.87730303823
#L0 = 43.9402810729
#Norm = -6.34232103471
#
#parameters = [43.9403, 1.0391, 2.5546, 5.83617, -1.3485, 1.8773, 44.32586, 0.26031, -6.342321]
#
#matrix = [[ 0.027414483358256400, 0.009185811147251570,  0.024585585436562200, -0.025438119503221600, -0.027392420918882700,  0.007192235482021060,   0.002963160876719450, -0.002909163707161460,-0.041877891411498200],
#          [0.009185811147251570,  0.006546785883021330,  0.008425511473072540,  0.003897095265996940,  0.002554832567327960, -0.001369163682634570,  -0.000723386505863384,  0.000095508653477705,-0.017369968324576700],
#          [0.024585585436562200,  0.008425511473072540,  0.030818578707093500, -0.014678165197323700, -0.014910924910161700,  0.008228073619160770,   0.007683225069501030, -0.002233582892853550,-0.038059997041001100],
#          [-0.025438119503221600, 0.003897095265996940, -0.014678165197323700,  0.155518507775386000,  0.151164555001805000, -0.057815279141344300,  -0.004301123991898550,  0.004649599528615900, 0.013981354317924200],
#          [-0.027392420918882700, 0.002554832567327960, -0.014910924910161700,  0.151164555001805000,  0.324456001631511000, -0.081635399889806100,  -0.002211087030002840,  0.007139118187159160, 0.026655099680892600],
#          [0.007192235482021060, -0.001369163682634570,  0.008228073619160770, -0.057815279141344300, -0.081635399889806100,  0.035496658032260300,   0.007746782469778650, -0.001329407316093770,-0.003593660623643860],
#          [0.002963160876719450, -0.000723386505863384,  0.007683225069501030, -0.004301123991898550, -0.002211087030002840,  0.007746782469778650,   0.010339733433261000, -0.001343902504603640,-0.003176008379971750],
#          [-0.002909163707161460, 0.000095508653477705, -0.002233582892853550,  0.004649599528615900,  0.007139118187159160, -0.001329407316093770,  -0.001343902504603640,  0.001019416244804420, 0.003972578046047490],
#          [-0.041877891411498200,-0.017369968324576700, -0.038059997041001100,  0.013981354317924200,  0.026655099680892600, -0.003593660623643860,  -0.003176008379971750,  0.003972578046047490, 0.070355261229252000]]
p2  =  -2.51732289173
a  =  0.28500252112
p1  =  5.48106536333
g2  =  2.56364894324
g1  =  1.04344797942
La  =  44.2828751738
zc  =  1.99790382378
L0  =  43.991000164
Norm  =  -6.32945832569

parameters = [L0, g1, g2, p1, p2, zc, La, a, Norm]

matrix = [[1.0, 0.55659109238987581, 0.72573173859707685, -0.30131609850344476, -0.12175673713116987, 0.089566339760646554, 0.028806414654154462, -0.32735046735247531, -0.90565078892147832], 
          [0.5565910923898757, 1.0, 0.42341365813758686, 0.28764313814023451, 0.13781753686159229, -0.14831133410763014, -0.11472389651512767, 0.21049408090621677, -0.7539417999196194], 
          [0.72573173859707685, 0.42341365813758686, 1.0, -0.08631616362355625, -0.0015232145470696955, 0.087037685030060097, 0.19199046050273197, -0.059295688186677499, -0.67437589111786544],
          [-0.30131609850344476, 0.28764313814023446, -0.086316163623556263, 1.0, 0.51724909948850784, -0.73300176240639836, 0.013054415749808502, 0.15527575424259266, -0.079680533117392927], 
          [-0.12175673713116987, 0.13781753686159226, -0.0015232145470696955, 0.51724909948850784, 1.0, -0.73422490478650237, 0.064070394811608325, 0.090061744647336184, 0.0066505123236200395], 
          [0.089566339760646541, -0.14831133410763017, 0.087037685030060097, -0.73300176240639847, -0.73422490478650226, 1.0, 0.12054255923324053, 0.14994810784937895, 0.10856129503430628], 
          [0.028806414654154459, -0.11472389651512767, 0.19199046050273197, 0.013054415749808503, 0.064070394811608325, 0.12054255923324055, 1.0, -0.32190186505613949, -0.0011144605505522909], 
          [-0.32735046735247531, 0.21049408090621677, -0.059295688186677492, 0.15527575424259266, 0.090061744647336184, 0.14994810784937895, -0.32190186505613949, 1.0, 0.25870628977513221], 
          [-0.90565078892147832, -0.75394179991961929, -0.67437589111786544, -0.07968053311739294, 0.0066505123236200404, 0.10856129503430628, -0.0011144605505522909, 0.25870628977513221, 1.0]]
#    3-sigma area
sigma = 3.0
draws = np.random.multivariate_normal(parameters,matrix, 5000) # draw random values using the covariance matrix
   
# Set up grid for data
# L sample
Lpoints = 100 # no need to go too thin
# z sample
zpoints = 400 # governed by the photoz step

LL = array([ones( (zpoints), float )*item for item in linspace(Lmin,Lmax,Lpoints)])
L = LL.ravel() #    make LL 1D
Z = tile(linspace(zmin, zmax, zpoints), Lpoints) # repeat as many times as Lpoints
## Set up grid for survey integral
#g = Source('grid')
#int_grid = g.Dz_area(L,Z)
#DVcA = int_grid
#redshift_int = Z[0:zpoints]
#Luminosity_int = linspace(Lmin,Lmax,Lpoints)

# Observations
from SetUp_data import Set_up_data
from LFunctions import *

setup_data = Set_up_data()
data = setup_data.get_data()[1]

#Lbin_cycle = itertools.cycle([[41.25, 42.0, 42.35, 42.90, 43.25, 43.650, 44.65],
#                              [42.0, 42.5, 42.8, 43.0, 43.1, 43.2, 43.6, 43.8, 44.60],
#                              [42.3, 42.6, 42.9, 43.25, 43.4, 43.6, 44.15, 45.0],
#                              [42.65, 43.0, 43.25, 43.45, 43.55, 43.65, 43.75, 43.85, 44.3, 45.00],
#                              [42.75, 43.1, 43.3, 43.5, 43.7, 43.85, 43.95, 44.1, 44.5, 45.20],
#                              [42.8, 43.5, 43.7, 43.9, 44.0, 44.1, 44.2, 44.4, 45.2],
#                              [43.0, 43.6, 43.9, 44.15, 44.25, 44.35, 44.9],
#                              [43.0, 43.8, 44.0, 44.25, 44.5, 44.6, 44.8, 45.0],
#                              [43.0, 43.8, 44.35, 45.0]])     
Lbin_cycle = itertools.cycle([[41.17, 42.0, 42.5, 43.0, 43.25, 43.50, 44.0, 44.65],
                              linspace(42.0, 44.60, 7),
                              linspace(42.3, 45.0, 6),
                              linspace(42.65, 45.00, 6),
                              linspace(42.75, 45.20, 6),
                              linspace(42.8, 45.2,6),
                              linspace(43.0, 44.9,6),
                              linspace(43.0, 45.0, 5),
                              linspace(43.0, 45.0, 4)])        
zbin = [0.01, 0.2, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 3.0, 4.0]
               
linecolor = (0., 0., 0.)
fillcolor = (0.75, 0.75, 0.75)
pointcolor = (0., 0., 0.)
zlabel = []
save_data = []
ll = linspace(41.0,46.0)
# LF at z=0
class Params():
    pass

params = Params()
params.L0 = L0
params.g1 = g1
params.g2 = g2
params.p1 = p1
params.p2 = p2
params.zc = zc
params.La = La 
params.a = a
params.Norm = Norm


LF0 = []
for lx in ll:
    PF = Fotopoulou(lx, 0.0, params)
    LF0.append(log10(PF))

redshift_bin = itertools.cycle([1.040e-01,  3.449e-01, 6.278e-01, 8.455e-01, 1.161e+00, 1.465e+00, 1.799e+00, 2.421e+00, 3.376e+00])
fig_size = [15, 15]
fig = plt.figure(figsize=fig_size)
fig.subplots_adjust(left=0.10,  right=0.95, bottom=0.10, top=0.99, wspace=0.0, hspace=0.0)

for i in arange(0,len(zbin)-1):
    LF = []
    print i
    Lbin = Lbin_cycle.next()            
#    Find median redshift
    Zz = []
    for Lx, z, f in zip(Lx_s, z_s, F_s):
        if zbin[i] <= z < zbin[i+1]:
            Zz.append(z)    

    dz = (zbin[i+1]-zbin[i])/0.005
    zspace = linspace(zbin[i], zbin[i+1], dz)
    
    for j in arange(0,len(Lbin)-1):        
#    Calculate denominator- 2D integral
        dL = (Lbin[j+1] - Lbin[j]) / 0.005
        Lspace = linspace(Lbin[j], Lbin[j+1], dL)   

        integral = []
        for z in zspace:
            dV = []
            for l in Lspace:
                dV.append( s.dV_dz(l, z) )
            integral.append(scipy.integrate.simps(dV, Lspace))
        integ = scipy.integrate.simps(integral, zspace)

#    Count sources per bin
        Ll = []
        count = 0
        for Lx, z, f in zip(Lx_s, z_s, F_s):
            if zbin[i] <= z < zbin[i+1] and Lbin[j] <= Lx < Lbin[j+1]:
                count = count + 1
                Ll.append(Lx)        
        temp_Phi = count/integ
        temp_err = sqrt(count)/integ

        datum = [median(Zz), median(Ll), count, log10(temp_Phi), 0.434*temp_err/temp_Phi, median(Ll)-Lbin[j], Lbin[j+1]-median(Ll)]
        LF.append(datum)
        save_data.append(datum)
    LFU_model = []
    LFF_model = []
    LFF_low = []
    LFF_upper = []

    for lx in ll:
        PF = modelF_Phi(lx, median(Zz), L0_F, g1_F, g2_F, p1_F, p2_F, zc_F, La_F, a_F)*norm_F
        LFF_model.append(log10(PF))
    
    redshift = asarray(LF)[:,0]
    luminosity = asarray(LF)[:,1]
    number = asarray(LF)[:,2]
    Phi = asarray(LF)[:,3]
    err_Phi = asarray(LF)[:,4]
    lbin_l = asarray(LF)[:,5]
    lbin_h = asarray(LF)[:,6]

#===============================================================================
    name = str("%.2f") % median(Zz)
# #    print min(redshift), max(redshift)
# #    print min(luminosity), max(luminosity)
#    interval_name = "/home/sotiria/workspace/Luminosity_Function/src/including_uncertainties/Bayesian/APEMoST/LDDEb_MAXI/analysis_results/"+str( redshift_bin.next()  )+"_99_interval.dat"
#    interval = genfromtxt(interval_name)
#    LF_ll = interval[:, 0]
#    LF_low = log10( interval[:, 1] )
#    LF_upper = log10( interval[:, 2] )
#    
    ax = fig.add_subplot(3,3,i+1)
#    plt.fill_between(LF_ll, LF_low, LF_upper, color=fillcolor)
#    plt.plot(ll, LF0, ls='--', color=linecolor)
    plt.plot(ll, LFF_model, ls='-', color=linecolor)
#===============================================================================
    plt.errorbar(luminosity, Phi, ls = ' ',yerr=err_Phi, xerr=[lbin_l, lbin_h], label=name, markersize = 7, color=pointcolor, marker='o',markeredgecolor='black')
#    print number of source in bin
    for j in range(0,len(number)):
        ax.annotate(str( int(number[j]) ), (luminosity[j], Phi[j]-0.1*Phi[j]), xycoords='data', fontstyle='normal', fontsize='xx-small', )
#    print redshift bin width
    ax.annotate(str(zbin[i])+"$< $"+"z"+"$ < $"+str(zbin[i+1]), (0.1, 0.1) , xycoords='axes fraction', fontstyle='oblique', fontsize='small', )
    #plt.legend(loc=3)
    plt.xlim([40.5, 46.5])
    plt.ylim([-9.5, -2.5])
    plt.xticks([42, 43, 44, 45, 46])
    plt.yticks([-8, -6, -4])        
    
    if i+1 in [1,2,3,4,5,6]:
        ax.set_xticklabels([])

    if i+1 in [2,3,5,6,8,9]:
        ax.set_yticklabels([])
    
    if i+1 == 4 :
        plt.ylabel("$Log[d\Phi/dLogLx/(Mpc^{-3})]$")
        # ax.yaxis.set_label_coords(-0.20, -0.15)
    
    if i+1 == 8 :
        plt.xlabel("$Log[Lx/(erg/sec)]$")
        #ax.xaxis.set_label_coords(0.5, -0.12)    
    
    plt.draw()
    #plt.savefig("/home/sotiria/workspace/Luminosity_Function/src/including_uncertainties/Bayesian/APEMoST/LDDE_Fotop/analysis_results/bw_dPhi%s.pdf" % i)
    #plt.savefig("/home/sotiria/workspace/Luminosity_Function/src/including_uncertainties/Bayesian/APEMoST/LDDE_Fotop/analysis_results/bw_dPhi%s.jpg" % i)
#plt.savefig("/home/sotiria/workspace/Luminosity_Function/src/including_uncertainties/Bayesian/APEMoST/LDDEb_MAXI/analysis_results/Ueda_dPhi_test.eps")
plt.show()
#savetxt('/home/sotiria/workspace/Luminosity_Function/src/including_uncertainties/Bayesian/APEMoST/LDDEb_MAXI/analysis_results/Ueda_Vmax_test.dat', save_data)
#plt.clf()