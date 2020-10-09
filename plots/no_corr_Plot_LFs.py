import warnings
import sys
# Append the module path to the sys.path list
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/no_uncertainties_included/MLE/MLE_modules/')
import time
import math   
from numpy import array,log,sqrt,power,linspace,vectorize,ones,tile,sum, log10, savetxt, genfromtxt
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
#work = ["LaFranca", "La Franca", "Silverman 1", "Silverman 2", "Ebrero", "Yencho All", "Yencho BLAGN", "Fotopoulou", "Aird1", "Aird2", "Fotopoulou uncert."]
#L0 = [43.94, 44.25, 44.33, 44.33, 43.91, 43.99, 43.64, 43.99, 44.24, 44.42, 44.21]
#g1 = [0.86, 1.01, 1.10, 1.10, 0.96,  1.004, 0.45, 0.91, 0.80, 0.77, 1.09]
#g2 = [2.23, 2.38, 2.15, 2.15, 2.35, 2.24, 2.15, 2.7, 2.36, 2.80, 2.88]
#p1 = [4.23, 4.62, 4.0, 4.22, 4.07,  5.58, 4.93, 4.8, 4.48, 4.64, 4.30]
#p2 = [-1.5, -1.15, -1.5, -3.27, -1.5,  -1.34, -2.16, -2.85, -2.85, -1.69, -3.16]
#zc = [1.9, 2.49, 1.9, 1.89, 1.9,  1.69, 2.23, 2.41, 1.89, 1.27, 2.39]
#La = [44.6, 45.74, 44.6, 44.6, 44.6, 44.68, 44.566, 44.59, 45.24, 44.70, 44.29]
#a = [0.335, 0.2, 0.317, 0.333, 0.245, 0.303, 0.553, 0.306, 0.15, 0.11, 0.385]
#Normal = [-5.297, -5.92, -6.077, -6.163, -5.32, -6.140,-5.51, -6.24, -5.91, -6.08, -6.5]
work = ["Ueda", "La Franca", "Silverman", "Ebrero", "Yencho","Aird", "Fotopoulou"]
L0 = [43.94, 44.25,  44.33, 43.91, 43.99,  44.24, 44.63]
g1 = [0.86, 1.01,  1.10, 0.96,  1.004, 0.80,  1.40]
g2 = [2.23, 2.38,  2.15, 2.35, 2.24, 2.36,  3.4]
p1 = [4.23, 4.62, 4.22, 4.07,  5.58, 4.48, 4.89]
p2 = [-1.5, -1.15, -3.27, -1.5,  -1.34,  -2.85,  -3.8]
zc = [1.9, 2.49,  1.89, 1.9,  1.69,  1.89, 2.03]
La = [44.6, 45.74,  44.6, 44.6, 44.68,  45.24,  43.85]
a = [0.335, 0.2,  0.333, 0.245, 0.303, 0.15,  0.55]
Normal = [-5.297, -5.92, -6.163, -5.32, -6.140,-5.91, -7.45]
pmin = 3.3
d = -0.3
vMiyaji = vectorize(model.Miyaji)
Lx = linspace(40.0, 47.0)
#redshift = [0.26, 0.73, 1.0, 1.44, 2.42, 3.37]
redshift = [1.040e-01, 1.161e+00, 2.421e+00, 3.376e+00]
zbin = [0.01, 0.2, 1.0, 1.3, 2.0, 3.0, 4.0]

x_plots = 2
y_plots = 2 
fig = plt.figure(figsize=(10,10))
#fig.subplots_adjust(left=0.1, right=0.8,wspace=0.15, hspace=0.15)
fig.subplots_adjust(left=0.15, right=0.95,wspace=0.28, hspace=0.15)
gray = 'gray'#(0.95, 0.95, 0.95)

for L0_d, g1_d, g2_d, p1_d, p2_d, zc_d, La_d, a_d, Normal_d,label in zip(L0,g1,g2,p1,p2,zc,La,a,Normal,work):
    for z in redshift :
        ax = fig.add_subplot(y_plots, x_plots, redshift.index(z)+1)
        if label == 'Fotopoulou':
            Phi = log10( model.Fotopoulou(Lx,z,L0_d+0.3098,g1_d,g2_d,p1_d,p2_d,zc_d,La_d+0.3098,a_d)*power(10.0, Normal_d) )
            plt.plot(Lx, Phi,ls='-.',lw=3, color ='k',label='Fotopoulou')       
#        else:
#            Phi = log10( model.Ueda(Lx,z,L0_d,g1_d,g2_d,p1_d,p2_d,zc_d,La_d,a_d)*power(10.0, Normal_d) )
#            plt.plot(Lx, Phi,label=label,ls='-',lw=4)

        #interval_name = "/home/sotiria/workspace/Luminosity_Function/src/including_uncertainties/Bayesian/APEMoST/LDDEb_MAXI/analysis_results/"+str( z  )+"_99_interval.dat"
        interval_name = "/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/previous_fits/Aird/"+str( z  )+"_99_interval.dat"
        interval = genfromtxt(interval_name)
        LF_ll = interval[:, 0]
        LF_low = log10( interval[:, 1] )
        LF_upper = log10( interval[:, 2] )
        plt.fill_between(LF_ll, LF_low, LF_upper, color=gray,alpha=0.03)
        plt.plot(LF_ll, LF_low, color='gray', ls='-', lw = 1)
        plt.plot(LF_ll, LF_upper, color='gray', ls='-', lw = 1)

        interval_name = "/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/previous_fits/Yencho/"+str( z  )+"_99_interval.dat"
        interval = genfromtxt(interval_name)
        LF_ll = interval[:, 0]
        LF_low = log10( interval[:, 1] )
        LF_upper = log10( interval[:, 2] )
        plt.fill_between(LF_ll, LF_low, LF_upper, color=gray,alpha=0.03)
        plt.plot(LF_ll, LF_low, color='gray', ls='-', lw = 1)
        plt.plot(LF_ll, LF_upper, color='gray', ls='-', lw = 1)

        interval_name = "/home/sotiria/workspace/Luminosity_Function/src/including_uncertainties/Bayesian/APEMoST/LDDE_Fotop/analysis_results/"+str( z  )+"_99_interval.dat"
        interval = genfromtxt(interval_name)
        LF_ll = interval[:, 0]
        LF_low = log10( interval[:, 1] )
        LF_upper = log10( interval[:, 2] )
        plt.plot(LF_ll+0.3098, LF_low,  color='black', ls = '-')
        plt.plot(LF_ll+0.3098, LF_upper, color='black', ls='-')

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

            
#plt.errorbar(Vmax_L[:6], Vmax_Phi[:6], yerr=Vmax_ePhi[:6], xerr=[Vmax_lowLbin[:6], Vmax_highLbin[:6]], marker='o',ls = ' ', color='blue', markersize=10)
#plt.errorbar(Vmax_L[6:13], Vmax_Phi[6:13], yerr=Vmax_ePhi[6:13], xerr=[Vmax_lowLbin[6:13], Vmax_highLbin[6:13]], marker='o',ls = ' ', color='green', markersize=10)
#plt.errorbar(Vmax_L[13:21], Vmax_Phi[13:21], yerr=Vmax_ePhi[13:21], xerr=[Vmax_lowLbin[13:21], Vmax_highLbin[13:21]], marker='o',ls = ' ', color=gray, markersize=10)
#plt.errorbar(Vmax_L[21:24], Vmax_Phi[21:24], yerr=Vmax_ePhi[21:24], xerr=[Vmax_lowLbin[21:24], Vmax_highLbin[21:24]], marker='o',ls = ' ', color='cyan', markersize=10)
#plt.errorbar(Vmax_L[24:26], Vmax_Phi[24:26], yerr=Vmax_ePhi[24:26], xerr=[Vmax_lowLbin[24:26], Vmax_highLbin[24:26]], marker='o',ls = ' ', color='black', markersize=10)
    
        plt.ylim([-11.5,-0.5])
        plt.xlim([42, 46 ])
        plt.xticks([42, 43, 44, 45, 46])
        plt.yticks([-10, -8, -6, -4, -2])
#plt.legend(bbox_to_anchor=(1.05, 1.5), loc=2, borderaxespad=0.)
#plt.yscale("log")
#plt.title("Luminosity Dependent Density Evolution (0)")
#plt.draw()
#plt.savefig("LDDE0.eps")
#plt.savefig("LDDE0.pdf")
#plt.savefig("LDDE0.jpg")
plt.draw()
#plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/no_correction_comparison_w_errors.jpg')
#plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/no_correction_comparison_w_errors.eps')
plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/no_correction_comparison_w_errors.pdf')
#plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/no_correction_comparison_w_errors.png')

plt.show()