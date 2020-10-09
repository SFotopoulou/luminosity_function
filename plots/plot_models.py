import numpy as np
import sys
# Append the module path to the sys.path list
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/no_uncertainties_included/MLE/MLE_modules/')
from LFunctions import Models
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import pyplot, mpl

m = Models()
Lx = np.linspace(40.0, 47.0,100)
steps = 100.
redshift = list(np.linspace(0,3.0,steps))

fig = plt.figure(figsize=(11.9,7.93))
# Make a figure and axes with dimensions as desired.
ax1 = fig.add_axes([0.735, 0.3, 0.23, 0.05])
# Set the colormap and norm to correspond to the data for which
# the colorbar will be used.
cmap = mpl.cm.spectral
norm = mpl.colors.Normalize(vmin=np.min(redshift), vmax=np.max(redshift))

# ColorbarBase derives from ScalarMappable and puts a colorbar
# in a specified axes, so it has everything needed for a
# standalone colorbar.  There are many more kwargs, but the
# following gives a basic continuous colorbar with ticks
# and labels.
cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                   norm=norm,
                                   orientation='horizontal')
cb1.set_label('redshift')
cb1.set_ticks([0,1,2,3])
for t in cb1.ax.get_xticklabels():
     t.set_fontsize('x-small')

fig.subplots_adjust(left=0.13, right=0.97, top=0.99, bottom=0.14,wspace=0.43, hspace=0.23)


ax = fig.add_subplot(2,3,1)
for z,i in zip(redshift, np.arange(len(redshift))):
    PLE = np.log10(m.PLE(Lx,z,43.8, 1.1, 4.5, 3.0, -1.5, 1.8))-6
    plt.plot(Lx, PLE,color=cm.spectral(i/steps,1))
plt.xticks([42, 43, 44, 45, 46],fontsize='x-small')
plt.yticks(fontsize='x-small')
plt.xlim([42, 46 ])
plt.ylim([-10.5,-1.5])
ax.annotate("PLE", (0.5, 0.8) , xycoords='axes fraction', fontstyle='normal', fontsize='small', )

ax = fig.add_subplot(2,3,2)
for z,i in zip(redshift, np.arange(len(redshift))):
    PDE = np.log10(m.PDE(Lx,z,43.8, 1.1, 4.5, 3.0, -1.5, 1.8))-6
    plt.plot(Lx, PDE, color=cm.spectral(i/steps,1))
plt.xticks([42, 43, 44, 45, 46],fontsize='x-small')
plt.yticks(fontsize='x-small')
plt.xlim([42, 46 ])
plt.ylim([-10.5,-1.5])
ax.annotate("PDE", (0.5, 0.8) , xycoords='axes fraction', fontstyle='normal', fontsize='small', )

ax = fig.add_subplot(2,3,3)
for z,i in zip(redshift, np.arange(len(redshift))):
    ILDE = np.log10(m.ILDE(Lx,z,43.8, 1.1, 4.5, 3, -1.5))-6
    plt.plot(Lx, ILDE, color=cm.spectral(i/steps,1))
plt.xticks([42, 43, 44, 45, 46],fontsize='x-small')
plt.yticks(fontsize='x-small')
plt.xlim([42, 46 ])
plt.ylim([-10.5,-1.5])
ax.annotate("ILDE", (0.5, 0.8) , xycoords='axes fraction', fontstyle='normal', fontsize='small', )

#fig.add_subplot(2,3,4)
#LADE = np.log10(m.LADE(Lx,0,43.8, 1.1, 4.5, 3, -1.5, 1.8, -0.19))-6
#plt.plot(Lx, LADE, 'black')
#LADE = np.log10(m.LADE(Lx,1,43.8, 1.1, 4.5, 3, -1.5, 1.8, -0.19))-6
#plt.plot(Lx, LADE, 'blue')
#LADE = np.log10(m.LADE(Lx,3,43.8, 1.1, 4.5, 3, -1.5, 1.8, -0.19))-6
#plt.plot(Lx, LADE, 'red')
#plt.xticks([42, 43, 44, 45, 46])
#plt.xlim([42, 46 ])
#plt.ylim([-10.5,-1.5])

ax = fig.add_subplot(2,3,4)
for z,i in zip(redshift, np.arange(len(redshift))):
    LADE = np.log10(m.LADEc(Lx,z,43.8, 1.1, 4.5, 3, -1.5, 1.8, -0.19))-6
    plt.plot(Lx, LADE,color=cm.spectral(i/steps,1) )
plt.xticks([42, 43, 44, 45, 46],fontsize='x-small')
plt.yticks(fontsize='x-small')
plt.xlim([42, 46 ])
plt.ylim([-10.5,-1.5])
ax.annotate("LADE", (0.5, 0.8) , xycoords='axes fraction', fontstyle='normal', fontsize='small', )
plt.xlabel('Luminosity', fontsize='small')
ax.xaxis.set_label_coords(1.9, -0.2)
plt.ylabel(r'Differential Luminosity Function', fontsize='small')
ax.yaxis.set_label_coords(-0.4, 1.075)

#fig.add_subplot(2,3,5)
#for z,i in zip(redshift, np.arange(len(redshift))):
#    Ueda = np.log10(m.Ueda(Lx,0,43.8, 1.1, 4.5, 3, -1.5, 1.8, 44.6, 0.3))-6
#    plt.plot(Lx, Ueda, 'black')
#plt.xticks([42, 43, 44, 45, 46])
#plt.xlim([42, 46 ])
#plt.ylim([-10.5,-1.5])

ax = fig.add_subplot(2,3,5)
for z,i in zip(redshift, np.arange(len(redshift))):
    Fotopoulou = np.log10(m.Fotopoulou(Lx,z,43.8, 1.1, 4.5, 4.5, -1.5, 1.8, 44.6, 0.2))-6
    plt.plot(Lx, Fotopoulou, color=cm.spectral(i/steps,1))
plt.xticks([42, 43, 44, 45, 46],fontsize='x-small')
plt.yticks(fontsize='x-small')
plt.ylim([-10.5,-1.5])
plt.xlim([42, 46 ])
ax.annotate("LDDE", (0.5, 0.8) , xycoords='axes fraction', fontstyle='normal', fontsize='small', )
#for ext in ['pdf','eps','png','jpg','svg']:
#    plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/model_comparison/models.'+ext)

plt.show()