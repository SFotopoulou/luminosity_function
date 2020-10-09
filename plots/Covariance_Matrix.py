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



Ebrero_params = [43.91, 0.96, 2.35, 4.07, 0.245, -5.32057]
Ebrero_e_params = [0.01, 0.02, 0.07, 0.07, 0.003, 0.021]
Ebrero_params_fixed = [ -1.5, 1.9, 44.6 ]
Ebrero_e_params_fixed = [ 0.1, 0.1, 0.1]

F_params = [44.21134274667063, 1.1056826371560786, 2.900270595387168,4.345379683172033,0.3897130660294979,-6.517108894915298]
F_e_params = [0.09343581300935494, 0.060992041593273205, 0.18525916390985256, 0.16225113080201664, 0.0019001894392412877, 0.17484232214388587]

F_params_fixed = [-3.124, 2.39, 44.288]
F_e_params_fixed = [0.1, 0.1, 0.1]

covariance_matrix = [[ 0.008730251152719140, 0.004338414626059660, 0.013235552177667900, -0.002183459880194990, -0.000001467295717298, -0.015247936089820900],
                     [ 0.004338414626059660, 0.003720029137715570, 0.005786360286882470,  0.002574387977674640,  0.000004539630782054, -0.009490477872657170],
                     [ 0.013235552177667900, 0.005786360286882470, 0.034320957812577600, -0.001316198482750900, -0.000000495083837659, -0.022794047354544700],
                     [-0.002183459880194990, 0.002574387977674640,-0.001316198482750900,  0.026325429446533100,  0.000013477047674266, -0.005254904980031620],
                     [-0.000001467295717298, 0.000004539630782054,-0.000000495083837659,  0.000013477047674266,  0.000003610719905004, -0.000001602696112224],
                     [-0.015247936089820900,-0.009490477872657170,-0.022794047354544700, -0.005254904980031620, -0.000001602696112224,  0.030569837612666400]]

import matplotlib.mlab as mlab
from numpy.linalg import *

sigma = 1.0/array( F_e_params )
stds = np.outer(sigma, sigma)
scaled = stds*covariance_matrix
#print scaled
#print
Ebrero_sigma = array(Ebrero_e_params)
stds = np.outer(Ebrero_sigma, Ebrero_sigma)
Ebrero_matrix = stds*scaled
#print Fotopoulou_matrix
label = 'Ebrero'
values = np.random.multivariate_normal(Ebrero_params, Ebrero_matrix, 100000)
name = '/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/covariance/'+label+'.prob'
savetxt(name, values)
fig = plt.figure(figsize=(10,10))
fig.subplots_adjust(left=0.05, right=1.1, top=0.99, bottom=-0.05,wspace=0.0, hspace=0.0)
#labels = ["$L_0$", "$\gamma_1$", "$\gamma_2$", "$p_1$", "$p_2$", "$z_c$", "$L_a$", "$a$", "$Norm$"]
labels = ["$L_0$", "$\gamma_1$", "$\gamma_2$", "$p_1$", "$a$", "$Norm$"]

#maximum = power(len(F_params),2)/2.0
#pbar = ProgressBar(widgets=[Percentage(), Bar()],maxval=maximum).start()
for i in range(0,len(Ebrero_sigma)+1):
    for j in range(i+1,len(Ebrero_sigma)):
        y = linspace(-5.0*Ebrero_e_params[i]+Ebrero_params[i], 5.0*Ebrero_e_params[i]+Ebrero_params[i])
        x = linspace(-5.0*Ebrero_e_params[j]+Ebrero_params[j], 5.0*Ebrero_e_params[j]+Ebrero_params[j])
        X, Y = np.meshgrid( x , y )
        Z = mlab.bivariate_normal(X, Y, sqrt(Ebrero_matrix[j][j]), sqrt(Ebrero_matrix[i][i]), Ebrero_params[j], Ebrero_params[i], Ebrero_matrix[j][i])
        fig.add_subplot(len(Ebrero_sigma),len(Ebrero_sigma),len(Ebrero_sigma)*i+j)
        #plt.plot(values[:,j],values[:,i],'.',color='gray',markersize=0.25)
        CS = plt.contour(X,Y,Z,3, colors='red',zorder=10)
        if j == i+1:
            plt.xlabel(labels[j])
            plt.ylabel(labels[i],rotation=0)
        plt.xticks([])
        plt.yticks([])     
plt.draw()
plt.savefig('Ebrero_contours.pdf')
plt.show()

