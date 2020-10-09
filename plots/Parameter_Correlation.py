import warnings
import sys
# Append the module path to the sys.path list
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/no_uncertainties_included/MLE/MLE_modules/')
import time
import math   
from numpy import array,log,sqrt,power,linspace,vectorize,ones,tile,sum, log10, savetxt, genfromtxt,loadtxt,isnan
from Source import Source
from parameters import Parameters
from LFunctions import Models
from scipy.integrate import simps
import minuit
import matplotlib.pyplot as plt

params = Parameters()
Lmin, Lmax = Parameters.L(params)
zmin, zmax = Parameters.z(params)

model = Models()
Vmax_z, Vmax_L, Vmax_count, Vmax_Phi, Vmax_ePhi, Vmax_lowLbin, Vmax_highLbin = genfromtxt('/home/sotiria/workspace/Luminosity_Function/src/no_uncertainties_included/Vmax/PageCarrera/Vmax.dat', unpack=True)
    
########################################################
# Data
########################################################
F_params = [44.63, 1.40, 3.4, 4.89, -3.8, 2.03, 43.85, 0.55, -7.45]
F_e_params = [0.13, 0.07, 0.4, 0.28, 0.6, 0.14, 0.06, 0.06, 0.18]

L0_d = F_params[0]
g1_d =F_params[1]
g2_d =F_params[2]
p1_d =F_params[3]
p2_d = F_params[4]
zc_d = F_params[5]
La_d =F_params[6]
a_d =F_params[7]
Normal_d = F_params[8]
 
L0_mean=44.2128571429
g1_mean=0.910285714286
g2_mean=2.37571428571
p1_mean=4.26714285714
p2_mean=-2.11285714286
zc_mean=1.93142857143
La_mean=44.9385714286
alpha_mean=0.225857142857
A_mean=-5.8326223138

 
redshift = [1.040e-01, 1.161e+00, 2.421e+00, 3.376e+00]
zbin = [0.01, 0.2, 1.0, 1.3, 2.0, 3.0, 4.0]
metaLF_file = '/home/sotiria/workspace/Luminosity_Function/src/meta-luminosity_function/second_approach/results/add/4/output/collated.txt'
dtype = ['L','z','median','q01','q99','+1sigma','-1sigma','+3sigma','-3sigma','q10','q90']
d = loadtxt(metaLF_file, dtype=[(d,'f') for d in dtype])
Lx = linspace(41.0, 46.0, 40)
#z = [1.040e-01, 1.161e+00, 2.421e+00, 3.376e+00]
zii = [0.11101266, 1.1716455, 2.4343038, 3.393924]
import matplotlib.pyplot as plt
x_plots = 2
y_plots = 2 
fig = plt.figure(figsize=(10,10))
#fig.subplots_adjust(left=0.1, right=0.8,wspace=0.15, hspace=0.15)
fig.subplots_adjust(left=0.15, right=0.95,wspace=0.28, hspace=0.15)
gray = 'gray'#(0.85, 0.85, 0.85)

for z, zi in zip(redshift,zii) :
    ax = fig.add_subplot(y_plots, x_plots, redshift.index(z)+1)

    Phi = log10( model.Fotopoulou(Lx,z,L0_d,g1_d,g2_d,p1_d,p2_d,zc_d,La_d,a_d)*power(10.0, Normal_d) )
    plt.plot(Lx, Phi,ls='-.',lw=2, color ='r',label='Fotopoulou')       
    
    Phi = log10( model.Ueda(Lx,z,L0_mean-0.3098,g1_mean,g2_mean,p1_mean,p2_mean,zc_mean,La_mean-0.3098,alpha_mean)*power(10.0, A_mean) )
    plt.plot(Lx, Phi,ls='-',lw=3, color ='b',label='Fotopoulou')       

#        else:
#            Phi = log10( model.Ueda(Lx,z,L0_d,g1_d,g2_d,p1_d,p2_d,zc_d,La_d,a_d)*power(10.0, Normal_d) )
#            plt.plot(Lx, Phi,label=label,ls='-',lw=4)

#    e = d[d['z'] == zi]
#    e = e[-isnan(e['median'])]
#    #print e
#    #plt.errorbar(x=e['L'], y=e['median'], yerr=[-e['-1sigma']+e['median'],e['+1sigma']-e['median']] )
#    plt.fill_between(x=e['L'], y1 =e['-3sigma'], y2=e['+3sigma'], color=(0.85, 0.85, 0.85))
#    plt.fill_between(x=e['L'], y1 =e['-1sigma'], y2=e['+1sigma'], color='gray' )
#    plt.plot(e['L'], e['median'], color='black', ls='-', lw=4)
    #plt.show()
#
#        interval_name = "/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/previous_fits/Ebrero/"+str( z  )+"_99_interval.dat"
#        interval = genfromtxt(interval_name)
#        LF_ll = interval[:, 0]
#        LF_low = log10( interval[:, 1] )
#        LF_upper = log10( interval[:, 2] )
#        plt.fill_between(LF_ll, LF_low, LF_upper, color=gray, alpha=0.03)
#        plt.plot(LF_ll, LF_low, color='gray', ls='-', lw = 1)
#        plt.plot(LF_ll, LF_upper, color='gray', ls='-', lw = 1)
#
#        interval_name = "/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/previous_fits/LaFranca/"+str( z  )+"_99_interval.dat"
#        interval = genfromtxt(interval_name)
#        LF_ll = interval[:, 0]
#        LF_low = log10( interval[:, 1] )
#        LF_upper = log10( interval[:, 2] )
#        plt.fill_between(LF_ll, LF_low, LF_upper, color=gray, alpha=0.03)
#        plt.plot(LF_ll, LF_low, color='gray', ls='-', lw = 1)
#        plt.plot(LF_ll, LF_upper, color='gray', ls='-', lw = 1)
#
#        interval_name = "/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/previous_fits/Silverman/"+str( z  )+"_99_interval.dat"
#        interval = genfromtxt(interval_name)
#        LF_ll = interval[:, 0]
#        LF_low = log10( interval[:, 1] )
#        LF_upper = log10( interval[:, 2] )
#        plt.fill_between(LF_ll, LF_low, LF_upper, color=gray, alpha=0.03)
#        plt.plot(LF_ll, LF_low, color='gray', ls='-', lw = 1)
#        plt.plot(LF_ll, LF_upper, color='gray', ls='-', lw = 1)
#
    interval_name = "/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/covariance/Fotopoulou_"+str( z  )+"_99_interval.dat"
    interval = genfromtxt(interval_name)
    
    LF_ll = interval[:, 0]
    LF_low = interval[:, 4] 
    LF_upper = interval[:, 5]
    plt.fill_between(LF_ll, interval[:, 7], interval[:, 8], color='black', alpha=0.25)
    plt.fill_between(LF_ll, interval[:, 5], interval[:, 6], color='black', alpha=0.5)
    plt.fill_between(LF_ll, interval[:, 3], interval[:, 4], color='black', alpha=0.75)
    
#    plt.plot(LF_ll, LF_low, color='blue', ls='-', lw = 1)
#    plt.plot(LF_ll, LF_upper, color='blue', ls='-', lw = 1)

#    interval_name = "/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/previous_fits/Fotopoulou/"+str( z  )+"_99_interval.dat"
#    interval = genfromtxt(interval_name)
#    LF_ll = interval[:, 0]
#    LF_low = log10( interval[:, 1] )
#    LF_upper = log10( interval[:, 2] )
#    plt.fill_between(LF_ll, LF_low, LF_upper, color=gray, alpha=0.1)
#    plt.plot(LF_ll, LF_low, color='gray', ls='-', lw = 1)
#    plt.plot(LF_ll, LF_upper, color='gray', ls='-', lw = 1)

    interval_name = "/home/sotiria/workspace/Luminosity_Function/src/including_uncertainties/Bayesian/APEMoST/LDDE_Fotop/analysis_results/"+str( z  )+"_99_interval.dat"
    interval = genfromtxt(interval_name)
    LF_ll = interval[:, 0]
    LF_low = log10( interval[:, 1] )
    LF_upper = log10( interval[:, 2] )
    plt.plot(LF_ll, LF_low,  color='red', ls = '-',lw=2)
    plt.plot(LF_ll, LF_upper, color='red', ls='-',lw=2)

    i = redshift.index(z)

    if i == 3:
        i = i +2
        plt.xlabel('Luminosity', fontsize='x-large')
        ax.xaxis.set_label_coords(-0.075, -0.15)
    if i == 2:
        i = i +2
        plt.ylabel(r'd$\Phi$/dlogLx', fontsize='x-large')
        ax.yaxis.set_label_coords(-0.25, 1.0)
    if i ==1 :
        i = i+1

    ax.annotate(str(zbin[i])+"$< $"+"z"+"$ < $"+str(zbin[i+1]), (0.1, 0.1) , xycoords='axes fraction', fontstyle='oblique', fontsize='medium', )

    plt.ylim([-11.5,-0.5])
    plt.xlim([42, 46.0 ])
    plt.xticks([42, 43, 44, 45, ])#46])
    plt.yticks([-10, -8, -6, -4, -2])
plt.draw()
#plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/covariance/Fotopoulou_MLE_error.pdf')
plt.show()