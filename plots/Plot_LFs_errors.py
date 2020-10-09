import warnings
import sys
# Append the module path to the sys.path list
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/no_uncertainties_included/MLE/MLE_modules/')
import time
import math   
from numpy import array,log,sqrt,power,linspace,vectorize,ones,tile,sum, log10, savetxt, genfromtxt, asarray
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
########################################################
# Data
#work = ["Ueda", "La Franca", "Silverman", "Ebrero", "Yencho","Aird", "Fotopoulou"]
#L0 = [43.94, 44.25,  44.33, 43.91, 43.99,  44.24, 44.63]
#g1 = [0.86, 1.01,  1.10, 0.96,  1.004, 0.80,  1.40]
#g2 = [2.23, 2.38,  2.15, 2.35, 2.24, 2.36,  3.4]
#p1 = [4.23, 4.62, 4.22, 4.07,  5.58, 4.48, 4.89]
#p2 = [-1.5, -1.15, -3.27, -1.5,  -1.34,  -2.85,  -3.8]
#zc = [1.9, 2.49,  1.89, 1.9,  1.69,  1.89, 2.03]
#La = [44.6, 45.74,  44.6, 44.6, 44.68,  45.24,  43.85]
#a = [0.335, 0.2,  0.333, 0.245, 0.303, 0.15,  0.55]
#Normal = [-5.297, -5.92, -6.163, -5.32, -6.140,-5.91, -7.45]

work =  ["Ueda", "LaFranca","Silverman", "Ebrero",         "Yencho",       "Aird",     "Fotopoulou"]
L0 =     [43.94,    44.25,   44.33,         43.91,           44.40,         44.24,      44.63]
g1 =     [0.86,     1.01,    1.10,          0.96,            0.872,         0.80,       1.40]
g2 =     [2.23,     2.38,    2.15,          2.35,            2.36,          2.36,       3.4]
p1 =     [4.23,     4.62,    4.22,          4.07,            3.61,          4.48,       4.89]
p2 =     [-1.5,     -1.15,   -3.27,         -1.5,           -2.83,          -2.85,      -3.8]
zc =     [1.9,      2.49,    1.89,          1.9,             2.18,          1.89,       2.03]
La =     [44.6,     45.74,   44.6,          44.6,            45.09,         45.24,      43.85]
a =      [0.335,    0.2,     0.333,         0.245,          0.208,          0.15,       0.55]
Normal = [log10(5.04e-6),  log10(1.21e-6), -6.163, log10(4.78e-6), -6.140, -5.91, -7.45]

e_L0 =  [0.26,      0.18,    0.10,          0.01,            0.14,           0.11,      0.13]
e_g1 =  [0.15,      0.10,    0.42,          0.02,            0.060,          0.03,       0.07]
e_g2 =  [0.13,      0.13,    0.13,          0.07,            0.24,           0.15,       0.4]
e_p1 =  [0.39,      0.26,    0.27,          0.07,            0.49,           0.30,       0.28]
e_p2 =  [0.0001,    0.72,    0.34,          0.0001,          0.24,           0.24,      0.6]
e_zc =  [0.0001,    0.82,    0.14,          0.0001,          0.55,           0.14,       0.14]
e_La =  [0.0001,    0.63,    0.0001,        0.0001,          0.49,           0.19,      0.06]
e_a =   [0.070,     0.04,    0.013,         0.003,           0.019,          0.01,       0.06]
e_Normal = [0.434294482*0.33e-6/5.04e-6, 0.434294482*0.0605e-6/1.21e-6, 0.015, 0.434294482*0.23e-6/4.78e-6, 0.038,0.19, 0.18]

import random

for L0_d, g1_d, g2_d, p1_d, p2_d, zc_d, La_d, a_d, Normal_d,label, e_L0, e_g1, e_g2, e_p1, e_p2, e_zc, e_La, e_a, e_Normal in zip(L0,g1,g2,p1,p2,zc,La,a,Normal,work, e_L0, e_g1, e_g2, e_p1, e_p2, e_zc, e_La, e_a, e_Normal):
    values = []
    for i in xrange(1,100000):
        L0 = random.gauss(L0_d, e_L0)
        g1 = random.gauss(g1_d, e_g1)
        g2 = random.gauss(g2_d, e_g2)
        p1 = random.gauss(p1_d, e_p1)
        p2 = random.gauss(p2_d, e_p2)
        zc = random.gauss(zc_d, e_zc)
        La = random.gauss(La_d, e_La)
        a = random.gauss(a_d, e_a)
        Normal = random.gauss(Normal_d, e_Normal)
                
        values.append( [L0, g1, g2, p1, p2, zc, La, a, Normal] )
    #print asarray(values)[:,0]
    #plt.hist(asarray(values)[:,0], bins=20)
    #plt.show()
    print label+'.prob'
    name = '/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/previous_fits/'+label+'/'+label+'.prob'
    savetxt(name, values)