import warnings
import sys
# Append the module path to the sys.path list
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/no_uncertainties_included/MLE/MLE_modules/')
import time
import math   
from numpy import array,log,sqrt,power,linspace,vectorize,ones,tile,sum, log10, savetxt, genfromtxt, asarray
import numpy as np
from Source import Source
from parameters import Parameters
from LFunctions import Models
from scipy.integrate import simps
import minuit
import matplotlib.pyplot as plt
import matplotlib.mlab as mlt

params = Parameters()
Lmin, Lmax = Parameters.L(params)
zmin, zmax = Parameters.z(params)

model = Models()

Ueda_params = [43.94, 0.86, 2.23, 4.23, 0.335]
Ueda_e_params = [0.26, 0.15, 0.13, 0.39, 0.070]
Ueda_norm = [log10(5.04e-6)]
Ueda_norm_err = [0.434*0.33e-6/5.04e-6] # 1-sigma
Ueda_params_fixed=[-1.5, 1.9, 44.6]
Ueda_e_params_fixed=[0.1, 0.1, 0.1]

F_params = [44.211425778419596, 1.1057008744120482,2.9004884976698597,4.345370918226949,0.38971594822991923]
F_e_params = [0.09273032649686454, 0.06459855361259513, 0.1836900609087202, 0.1747323475793662, 0.017598938922500647]
F_params_fixed = [-3.124, 2.39, 44.288]
F_e_params_fixed = [0.1, 0.1, 0.1]

covariance_matrix = [[ 0.008598913452215100, 0.004110576716423850, 0.012924796234118700, -0.002574572366199900, -0.000119768800941508],
                     [ 0.004110576716423850, 0.004172973128839330, 0.005610381609773650,  0.004033560085748800,  0.000390628720549451],
                     [ 0.012924796234118700, 0.005610381609773650, 0.033742038476649300, -0.001341020279010050, -0.000032012420374967],
                     [-0.002574572366199900, 0.004033560085748800,-0.001341020279010050,  0.030531393290596400,  0.001147395887834340],
                     [-0.000119768800941508, 0.000390628720549451,-0.000032012420374967,  0.001147395887834340,  0.000309722651197908]]

import matplotlib.mlab as mlab
from numpy.linalg import *

sigma = 1.0/array( F_e_params )
stds = np.outer(sigma, sigma)
scaled = stds*covariance_matrix
#print scaled
#print
Ueda_sigma = array(Ueda_e_params)
stds = np.outer(Ueda_sigma, Ueda_sigma)
Ueda_matrix = stds*scaled
#print Fotopoulou_matrix
label = 'Ueda'
values = np.random.multivariate_normal(Ueda_params, Ueda_matrix, 100000)
name = '/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/covariance/'+label+'.prob'
savetxt(name, values)
fig = plt.figure(figsize=(10,10))
fig.subplots_adjust(left=0.05, right=1.1, top=0.99, bottom=-0.05,wspace=0.0, hspace=0.0)
#labels = ["$L_0$", "$\gamma_1$", "$\gamma_2$", "$p_1$", "$p_2$", "$z_c$", "$L_a$", "$a$", "$Norm$"]
labels = ["$L_0$", "$\gamma_1$", "$\gamma_2$", "$p_1$", "$a$"]

#maximum = power(len(F_params),2)/2.0
#pbar = ProgressBar(widgets=[Percentage(), Bar()],maxval=maximum).start()
for i in range(0,len(Ueda_sigma)+1):
    for j in range(i+1,len(Ueda_sigma)):
        y = linspace(-5.0*Ueda_e_params[i]+Ueda_params[i], 5.0*Ueda_e_params[i]+Ueda_params[i])
        x = linspace(-5.0*Ueda_e_params[j]+Ueda_params[j], 5.0*Ueda_e_params[j]+Ueda_params[j])
        X, Y = np.meshgrid( x , y )
        Z = mlab.bivariate_normal(X, Y, sqrt(Ueda_matrix[j][j]), sqrt(Ueda_matrix[i][i]), Ueda_params[j], Ueda_params[i], Ueda_matrix[j][i])
        fig.add_subplot(len(Ueda_sigma),len(Ueda_sigma),len(Ueda_sigma)*i+j)
        plt.plot(values[:,j],values[:,i],'.',color='gray',markersize=0.25)
        CS = plt.contour(X,Y,Z,3, colors='red',zorder=10)
        if j == i+1:
            plt.xlabel(labels[j])
            plt.ylabel(labels[i],rotation=0)
        plt.xticks([])
        plt.yticks([])     
plt.draw()
plt.savefig('Ueda_contours.pdf')
plt.show()