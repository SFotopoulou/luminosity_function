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
#work = ["LaFranca", "La Franca", "Silverman 1", "Silverman 2", "Ebrero", "Ebrero All", "Ebrero BLAGN", "Fotopoulou", "Aird1", "Aird2", "Fotopoulou uncert."]
#L0 = [43.94, 44.25, 44.33, 44.33, 43.91, 43.99, 43.64, 43.99, 44.24, 44.42, 44.21]
#g1 = [0.86, 1.01, 1.10, 1.10, 0.96,  1.004, 0.45, 0.91, 0.80, 0.77, 1.09]
#g2 = [2.23, 2.38, 2.15, 2.15, 2.35, 2.24, 2.15, 2.7, 2.36, 2.80, 2.88]
#p1 = [4.23, 4.62, 4.0, 4.22, 4.07,  5.58, 4.93, 4.8, 4.48, 4.64, 4.30]
#p2 = [-1.5, -1.15, -1.5, -3.27, -1.5,  -1.34, -2.16, -2.85, -2.85, -1.69, -3.16]
#zc = [1.9, 2.49, 1.9, 1.89, 1.9,  1.69, 2.23, 2.41, 1.89, 1.27, 2.39]
#La = [44.6, 45.74, 44.6, 44.6, 44.6, 44.68, 44.566, 44.59, 45.24, 44.70, 44.29]
#a = [0.335, 0.2, 0.317, 0.333, 0.245, 0.303, 0.553, 0.306, 0.15, 0.11, 0.385]
#Normal = [-5.297, -5.92, -6.077, -6.163, -5.32, -6.140,-5.51, -6.24, -5.91, -6.08, -6.5]
work = ["Ueda", "LaFranca", "Silverman", "Ebrero", "Yencho","Aird", "Fotopoulou"]
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
Lx = linspace(40.0, 47.0)
redshift = [1.040e-01,  1.161e+00, 3.376e+00]
import matplotlib.pyplot as plt
x_plots = 3
y_plots = 2 
fig = plt.figure(figsize=(15,10))
fig.subplots_adjust(left=0.10, right=0.97, top=0.86,wspace=0.34, hspace=0.15)
gray = 'gray'

for L0_d, g1_d, g2_d, p1_d, p2_d, zc_d, La_d, a_d, Normal_d,label_d in zip(L0,g1,g2,p1,p2,zc,La,a,Normal,work):
    for z in redshift :
        ax = fig.add_subplot(y_plots, x_plots, redshift.index(z)+1)
        if label_d == 'Fotopoulou':
            Phi = log10( model.Fotopoulou(Lx,z,L0_d+0.3098,g1_d,g2_d,p1_d,p2_d,zc_d,La_d+0.3098,a_d)*power(10.0, Normal_d) )
#            plt.plot(Lx, Phi,ls='-.',lw=3, color ='k',label=label_d)       
        else:
            Phi = log10( model.Ueda(Lx,z,L0_d,g1_d,g2_d,p1_d,p2_d,zc_d,La_d,a_d)*power(10.0, Normal_d) )
            plt.plot(Lx, Phi, ls='-',lw=4, label=label_d)

        ax.annotate("z="+str(round(z,1)), (0.1, 0.1) , xycoords='axes fraction', fontstyle='oblique', fontsize='medium', )
        plt.ylim([-11.5,-0.5])
        plt.xlim([42, 46 ])
        plt.xticks([42, 43, 44, 45, 46])
        plt.yticks([-10, -8, -6, -4, -2])
        
        
    for z in redshift :
        ax = fig.add_subplot(y_plots, x_plots, redshift.index(z)+4)
#   99% probability
        if label_d!='Ebrero':
            pass
#            interval_name = "/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/previous_fits/"+label_d+"/"+str( z  )+"_99_interval.dat"
#            interval = genfromtxt(interval_name)
#            LF_ll = interval[:, 0]
#            LF_low = log10( interval[:, 1] )
#            LF_upper = log10( interval[:, 2] )
            
        else:
            interval_name = "/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/covariance/"+label_d+"_"+str( z  )+"_99_interval.dat"
            interval = genfromtxt(interval_name)
            LF_ll = interval[:, 0]
            LF_low = ( interval[:, 3] )
            LF_upper = ( interval[:, 4] )
            clr='gray'
            plt.fill_between(LF_ll, LF_low, LF_upper, color=clr, alpha=0.1)
            plt.plot(LF_ll, LF_low, color=clr, ls='-', lw = 1)
            plt.plot(LF_ll, LF_upper, color=clr, ls='-', lw = 1)
            Phi = log10( model.Ueda(Lx,z,L0_d,g1_d,g2_d,p1_d,p2_d,zc_d,La_d,a_d)*power(10.0, Normal_d) )
            plt.plot(Lx, Phi, ls='-',lw=4, label=label_d)
#        if label_d=='Ebrero':
#            clr = 'red'
#        else: clr = 'w'
#        plt.fill_between(LF_ll, LF_low, LF_upper, color=clr, alpha=0.1)
#        plt.plot(LF_ll, LF_low, color=clr, ls='-', lw = 1)
#        plt.plot(LF_ll, LF_upper, color=clr, ls='-', lw = 1)
#   'best' fit value
#        if label_d == 'Fotopoulou':
#            Phi = log10( model.Fotopoulou(Lx,z,L0_d+0.3098,g1_d,g2_d,p1_d,p2_d,zc_d,La_d+0.3098,a_d)*power(10.0, Normal_d) )
##            plt.plot(Lx, Phi,ls='-.',lw=3, color ='k',label=label_d)       
#        else:
#            Phi = log10( model.Ueda(Lx,z,L0_d,g1_d,g2_d,p1_d,p2_d,zc_d,La_d,a_d)*power(10.0, Normal_d) )
#            plt.plot(Lx, Phi, ls='-',lw=4, label=label_d)
#            
        i = redshift.index(z)
        if i == 2:
            i = i +2
            plt.xlabel('Luminosity', fontsize='x-large')
            ax.xaxis.set_label_coords(-0.82, -0.15)
        if i == 1:
            i = i + 2
            plt.ylabel(r'd$\Phi$/dlogLx', fontsize='x-large')
            ax.yaxis.set_label_coords(-1.6, 1.0)

        ax.annotate("z="+str(round(z,1)), (0.1, 0.1) , xycoords='axes fraction', fontstyle='oblique', fontsize='medium', )
        plt.ylim([-11.5,-0.5])
        plt.xlim([42, 46 ])
        plt.xticks([42, 43, 44, 45, 46])
        plt.yticks([-10, -8, -6, -4, -2])
#   bbox_to_anchor = (x, y, width, height)        
plt.legend(bbox_to_anchor=(-2.30, 2.4, 3., 0.1,), loc=2, ncol=3, mode="expand", borderaxespad=0.)
plt.draw()
#for ext in ['jpg','pdf','eps','png']:
#    plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/covariance/Ebrero_best_fit_LFs.'+ext)
plt.show()
