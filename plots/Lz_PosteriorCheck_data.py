import sys
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
from AGN_LF_config import LF_config
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
from scipy import interpolate
import itertools
from matplotlib.ticker import NullFormatter

fig = plt.figure(figsize=(10,10))
nullfmt   = NullFormatter()         # no labels
# definitions for the axes
left, width = 0.11, 0.65
bottom, height = 0.11, 0.65
bottom_h = left_h = left+width

rect_2dhist = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(1, figsize=(8,8))

ax2Dhist = plt.axes(rect_2dhist)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)
# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)


lum_data = pyfits.open(LF_config.data_out_name)[1].data

data_Z = []
data_logL = []
for fld in ['MAXI', 'HBSS', 'COSMOS', 'LH', 'X_CDFS', 'AEGIS']: #LF_config.fields:
    z_data = lum_data[lum_data.field('Z_'+fld)>0]
    use_data = z_data[z_data.field('Z_'+fld)<=4.5]
    data_logL.extend( use_data.field( 'Lum_'+fld ) )
    data_Z.extend( use_data.field( 'Z_'+fld ) ) 
    
zbin = 25
Lbin = 25
N, yedges, xedges = np.histogram2d(data_logL, data_Z, bins=(Lbin, zbin))
im = ax2Dhist.imshow(N, origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='bone_r', aspect='auto')

ax2Dhist.set_xlim( (0.00, 4.5) )
ax2Dhist.set_ylim( (41, 46) )
im.set_clim(0,20)


ax2Dhist.set_ylabel('$\mathrm{log(\,L_x /erg\cdot s^{-1})}$', fontsize='large')
ax2Dhist.set_xlabel('$\mathrm{z}$', fontsize='large')
ax2Dhist.text(3.5, 41.5, '$\mathrm{data}$')
ax2Dhist.text(4.85, 46.5, '$\mathrm{N='+str(len(data_Z))+'}$')


axHistx.hist(data_Z, bins = zbin, histtype='step', color='k', normed=False)
axHistx.set_xlim( ax2Dhist.get_xlim() )
axHistx.set_ylim((0,120))
axHistx.set_yticks([])

axHisty.hist(data_logL, bins = Lbin, orientation='horizontal', histtype='step', color='k', normed=False)
axHisty.set_ylim( ax2Dhist.get_ylim() )
axHisty.set_xlim((0,150))
axHisty.set_xticks([])

plt.savefig('plots/Lz_2dhist_data.pdf', dpi=300)
plt.show()