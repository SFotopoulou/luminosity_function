import sys
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')

import itertools
import pyfits
from numpy import arange,log10,sqrt,array, power,ones
import matplotlib.pyplot as plt
from Source import Source
from matplotlib.ticker import MultipleLocator

fig = plt.figure(figsize=(10,10))
fig.subplots_adjust(left=0.175, top=0.98,right=0.95, wspace=0.05, hspace=0.05)
ax = fig.add_subplot(111)

##
def get_area(fluxx,areaa,fx):
    omega = 0.
    for item in arange(0,len(fluxx)-1):
        if fx > fluxx[item] and fx < fluxx[item+1]:
            omega = ( areaa[item] + areaa[item+1] ) / 2.
            return omega
        elif fx > max(fluxx):
            omega = max(areaa)
        elif fx < min(fluxx):
            omega = 0.0
            return omega
#
lstyles = itertools.cycle(['-','-',':','-.','--'])
mstyles = itertools.cycle(['o','s','v','o','v'])

lcolors = itertools.cycle(['gray','black','gray','black','black'])
mcolors = itertools.cycle(['red','black','gray','white','black'])
medgecolors = itertools.cycle(['red','black','gray','black','black'])
z_order = itertools.cycle([1,3,2])
lcolors_c = itertools.cycle(['red','blue','cyan','green','magenta','black'])
mcolors_c = itertools.cycle(['red','blue','cyan','green','magenta','black'])
medgecolors_c = itertools.cycle(['black','black','black','black','magenta','black'])

### Fx - z
in_path='/home/sotiria/workspace/Luminosity_Function/input_files/'
data_name='Input_Data.fits'
dfile = in_path+data_name
f = pyfits.open(dfile)
fdata = f[1].data

HBSS = fdata[fdata.field('field')==1]
COSMOS = fdata[fdata.field('field')==2]
LH = fdata[fdata.field('field')==3]
CDFS = fdata[fdata.field('field')==4]
MAXI = fdata[fdata.field('field')==5]

HBSS_S = log10(HBSS.field('Flux'))
COSMOS_S = log10(COSMOS.field('Flux'))
LH_S = log10(LH.field('Flux'))
CDFS_S = log10(CDFS.field('Flux'))
MAXI_S = log10(MAXI.field('Flux'))

HBSS_z = HBSS.field('redshift')
COSMOS_z = COSMOS.field('redshift')
LH_z = LH.field('redshift')
CDFS_z = CDFS.field('redshift')
MAXI_z = MAXI.field('redshift')


#### Plot L-z
field = itertools.cycle([COSMOS_S,LH_S,CDFS_S])
field_z = itertools.cycle([COSMOS_z,LH_z,CDFS_z])
field_name = itertools.cycle(['COSMOS','LH','CDFS'])
#plt.figure(4,figsize=(12,12))
#ax4 = plt.subplot(111)
import numpy as np
import numpy.ma as ma
s = Source('none')
for i in [1,2,3]:
    F = 10.**field.next()
    z = field_z.next()
    #print F,z
    Lx, Lx_err = s.get_luminosity(F,ones(len(F)),z)
    mask_lum = np.where(np.array(Lx)>42,0,1)
    Lx = ma.masked_array(np.array(Lx),mask_lum)
    plt.errorbar(z,10.**(array(Lx)),label=field_name.next(),markersize = 9.0,marker=mstyles.next(),markeredgecolor=medgecolors.next(),markerfacecolor=mcolors.next(),ecolor=lcolors.next(),ls=' ',zorder=z_order.next())
plt.legend(loc=4)
plt.yscale('log',nonposy='clip')
#plt.xscale('log',nonposx='clip')
plt.xlim([0.95,3.05])
plt.ylim([3e42, 2e45])
plt.ylabel('$\mathrm{L_x /erg\cdot s^{-1}}$', fontsize='x-large')
plt.xlabel('$\mathrm{Redshift}$', fontsize='x-large')
plt.draw()
for ext in ['eps','pdf','png','jpg']:
    plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/Lx_z_small.'+ext)

plt.show()