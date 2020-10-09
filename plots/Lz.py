import sys
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
from AGN_LF_config import LF_config
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
from scipy import interpolate
import itertools

fig = plt.figure(figsize=(10,10))
fig.subplots_adjust(left=0.175, top=0.97,right=0.95, wspace=0.05, hspace=0.05)

ax = fig.add_subplot(111)
lum_file = '/home/Sotiria/Dropbox/transfer/XLF_output_files/LDDE_withUnc_MHXCLAXC_ztype_all_04_4146.out.fits'
lum_data = pyfits.open(lum_file)[1].data
colors = itertools.cycle(['#00AA00', '#0099FF', '#0066FF', '#0000AA','black', '#FF0000', '#990000','#440000'])
fields = ['MAXI','HBSS','LH','XMM_COSMOS','XMM_CDFS','AEGIS', 'Chandra_COSMOS', 'Chandra_CDFS']  

for fld in fields:
    
    logL = lum_data.field( 'Lum_'+fld )
    L = np.power(np.zeros(len(logL)) + 10, logL)
    Z = lum_data.field( 'Z_'+fld )
    
    lbl = fld
    if fld == 'XMM_CDFS': lbl = 'X-CDFS'
    if fld == 'XMM_COSMOS': lbl = 'X-COSMOS'
    if fld == 'LH': lbl = 'X-LH'
    if fld == 'HBSS': lbl = 'X-HBSS'
    if fld == 'AEGIS': lbl = 'C-AEGIS'
    if fld == 'Chandra_CDFS': lbl = 'C-CDFS'
    if fld == 'Chandra_COSMOS': lbl = 'C-COSMOS'
    color = colors.next()
    
    plt.errorbar(Z, L, \
                 label='$\mathrm{'+lbl+'}$', \
                 markersize = 7.0, \
                 marker = LF_config.mstyles.next(),\
                 markeredgecolor = color, \
                 #LF_config.medgecolors.next(), \
                 markerfacecolor = color,\
                 ecolor = color,\
                 ls=' ',\
                 zorder = fields.index(fld))
    
plt.legend(loc=0)

#plt.xlim([LF_config.zmin, LF_config.zmax+0.5])
#plt.ylim([np.power(10., LF_config.Lmin), np.power(10., LF_config.Lmax+0.1)])

#plt.xlim([-0.2, 4.2])
#plt.xticks([0,1,2,3,4])
plt.xlim([0.005, 10])
plt.xticks([0.01, 0.1, 1])
plt.ylim([1e41, 3e46])

plt.xscale('log')
plt.yscale('log')

plt.ylabel('$\mathrm{L_x /erg\cdot s^{-1}}$', fontsize='x-large')
plt.xlabel('$\mathrm{redshift}$', fontsize='x-large')

plt.savefig('plots/Lz.pdf', dpi=300, bbox_inches='tight')
plt.show()