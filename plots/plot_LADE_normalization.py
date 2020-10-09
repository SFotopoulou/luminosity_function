import numpy as np
import sys
# Append the module path to the sys.path list
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/no_uncertainties_included/MLE/MLE_modules/')
from Source import Source
from parameters import Parameters
from LFunctions import Models
import matplotlib.pyplot as plt
import itertools
colors1 = itertools.cycle(['r','g','b','magenta','k','cyan'])
params = Parameters()
Lmin, Lmax = Parameters.L(params)
zmin, zmax = Parameters.z(params)
m = Models()
Lx = np.linspace(40.0, 47.0,100)
redshift = [0.0, 1.0, 2.0, 3.0]

fig = plt.figure(figsize=(15,10))
fig.subplots_adjust(left=0.10, right=0.97, top=0.98,wspace=0.34, hspace=0.15)

fig.add_subplot(1,2,1)
LADE = np.log10(m.LADE(Lx,0,45.14, 1.1, 4.5, 3, -1.5, 1.8, -0.19))-6
plt.plot(Lx, LADE, 'black')
LADE = np.log10(m.LADE(Lx,1,43.8+np.log10(np.power(2.8, 3)+np.power(2.8,-1.5)), 1.1, 4.5, 3, -1.5, 1.8, -0.19))-6
plt.plot(Lx, LADE, 'blue')
LADE = np.log10(m.LADE(Lx,3,45.14, 1.1, 4.5, 3, -1.5, 1.8, -0.19))-6
plt.plot(Lx, LADE, 'red')
plt.xticks([42, 43, 44, 45, 46])
plt.xlim([42, 46 ])
plt.ylim([-11.5,-0.5])

fig.add_subplot(1,2,1)
LADE = np.log10(m.LADEc(Lx,0,43.8, 1.1, 4.5, 3, -1.5, 1.8, -0.19))-6
plt.plot(Lx, LADE, 'black')
LADE = np.log10(m.LADEc(Lx,1,43.8, 1.1, 4.5, 3, -1.5, 1.8, -0.19))-6
plt.plot(Lx, LADE, 'blue', lw = 15, alpha=0.1)
LADE = np.log10(m.LADEc(Lx,3,43.8, 1.1, 4.5, 3, -1.5, 1.8, -0.19))-6
plt.plot(Lx, LADE, 'red')
plt.xticks([42, 43, 44, 45, 46])
plt.xlim([42, 46 ])
plt.ylim([-11.5,-0.5])

#plt.yticks([-10, -8, -6, -4, -2])

#   bbox_to_anchor = (x, y, width, height)        
#plt.legend(bbox_to_anchor=(-2.30, 2.4, 3., 0.1,), loc=2, ncol=3, mode="expand", borderaxespad=0.)
#plt.draw()
#for ext in ['pdf','eps','png','svg']:
#plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_plots/model_comparison/model_comp.svg')
plt.show()
