import sys
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
from AGN_LF_config import LF_config
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
from scipy import interpolate
import itertools
from matplotlib.ticker import MultipleLocator

fig = plt.figure(figsize=(10,10))
fig.subplots_adjust(left=0.175, top=0.98,right=0.95, wspace=0.05, hspace=0.05)
ax = fig.add_subplot(111)

colors = itertools.cycle(['#00AA00', '#0099FF', '#0066FF', '#0000AA','black', '#FF0000', '#990000','#440000'])

for fld in LF_config.fields:
    curve_data = pyfits.open( LF_config.inpath + fld + LF_config.acname)[1].data
    flux = curve_data.field( fld + '_FLUX' )
    area = curve_data.field( fld + '_AREA' )
    if fld == 'MAXI': 
        flux = flux[2:]
        area = area[2:]
    if fld == 'XMM_CDFS': fld = 'X-CDFS'
    if fld == 'XMM_COSMOS': fld = 'X-COSMOS'
    if fld == 'LH': fld = 'X-LH'
    if fld == 'HBSS': fld = 'X-HBSS'
    if fld == 'AEGIS': fld = 'C-AEGIS'
    if fld == 'Chandra_CDFS': fld = 'C-CDFS'
    if fld == 'Chandra_COSMOS': fld = 'C-COSMOS'
    flux = [min(flux)] + list(flux)
    area = [1e-4] + list(area)
    
    flux = list(flux) + [1e-9]
    area = list(area) + [max(area)]
    color = colors.next()

    plt.plot( flux, area, \
              label='$\mathrm{'+fld+'}$', \
              ls = LF_config.lstyles.next(), \
              color= color, lw=3)

#flux, area = np.loadtxt(LF_config.inpath + 'Total_acurve_coherent.txt', unpack=True)
#area = area / 0.00030461742

#plt.plot( flux, area, \
#          label='$\mathrm{all\,fields}$', \
#          ls = '-', \
#          color= 'black', lw=2)

plt.legend(loc=2)

plt.xlim([3e-16, 1e-9])
plt.ylim([1e-3, 5e4])

plt.xscale('log')
plt.yscale('log')

plt.ylabel('$\mathrm{area /deg^{2}}$', fontsize='x-large')
plt.xlabel('$\mathrm{F_x /erg\cdot s^{-1}\cdot cm^{-2}}$', fontsize='x-large')
#ax.yaxis.set_major_locator(MultipleLocator(10))
#ax.yaxis.set_minor_locator(MultipleLocator(0.5))
#ax.xaxis.set_major_locator(MultipleLocator(1))
#ax.xaxis.set_minor_locator(MultipleLocator(0.5)) 


plt.savefig('plots/acurve.pdf', dpi=300)
plt.show()