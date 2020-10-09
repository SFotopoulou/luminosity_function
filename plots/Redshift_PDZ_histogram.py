import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/models')
sys.path.append('/home/Sotiria/Documents/Science/Talks/configuration/XKCDify')

import numpy as np
from AGN_LF_config import LF_config
from cosmology import *
from Survey import *
from Source import *
from LFunctions import *
from SetUp_grid import *
from SetUp_data import Set_up_data
from scipy.integrate import simps,romb
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
import time
import itertools
import numpy.ma as ma

setup_data = Set_up_data()
source_list = setup_data.get_data()[0]
redshift=[ source.z for source in source_list]

fig = plt.figure()
ax = fig.add_subplot(111)

bins = 10
zbins = np.linspace(0.0, 4.0, bins+1)
print min(redshift), max(redshift)
dz = zbins[1]-zbins[0]

N=[]
for j in range(0, len(zbins)-1):
    n=0
    for source in source_list:
        if source.zflag == 1 or source.field in ['MAXI', 'HBSS']:
            #print source.zflag, source.field
            if zbins[j] < source.z < zbins[j+1]:
                n = n + 1
        else:  
            x = np.array(source.xPDF_z)
            y = np.array(source.yPDF_z)
            condition = (x < zbins[j]) | (x >= zbins[j+1]) 
            m = ma.masked_where(condition, x, copy=True)
            if ma.getmask(m).all() != True :
                x1, x2 = ma.flatnotmasked_edges( m )
#                if x1 == x2:
#                    #n = n + y[x1]*0.01
#                    
#                else:    
#                    print m
#                    
                xPDF = x[max(0, x1-1):x2+1]
                yPDF = y[max(0, x1-1):x2+1]
#                print xPDF
                Pr = simps(yPDF, xPDF)
                n = n + Pr
                
                #if Pr>0.1: print Pr 
    #print j, "nbin=", n
    N.append(n)

#plt.hist(redshift, bins, fill=False, histtype='step',color='k')
red, bi = np.histogram(redshift, zbins)
print np.mean(N-red), np.median(N-red),np.sum(red), np.sum(N)
print zbins
print bi
ax.plot(bi[:-1]+0.5*(bi[1]-bi[0]), red, 'k')
ax.plot(zbins[:-1]+dz*0.5, N, 'r')
#from XKCDify import XKCDify as xkcd
#xkcd(ax)
plt.show()