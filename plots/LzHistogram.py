import sys
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
from astroML.plotting import hist
from AGN_LF_config import LF_config
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
from scipy import interpolate
import itertools
from matplotlib.backends.backend_pdf import PdfPages
import itertools
fields= LF_config.fields # ['MAXI','HBSS','COSMOS', 'LH', 'X_CDFS', 'AEGIS']

fig = plt.figure(figsize=(10,10))
fig.subplots_adjust(left=0.09, top=0.96, bottom=0.10, right=0.98, wspace=0.22, hspace=0.22)   
colors = itertools.cycle(['#00AA00', '#0099FF', '#0066FF', '#0000AA','black', '#FF0000', '#990000','#440000'])

bins = np.linspace(0,4, 4/0.25)#'freedman'

spec_z = {}
photo_z = {}

all_specz = []
all_photoz = []
all_z = []

file_name = '/home/Sotiria/Dropbox/transfer/XLF_output_files/LDDE_withUnc_MHXCLAXC_ztype_all_04_4146.out.fits'
incat = pyfits.open(file_name)[1].data

for fld in fields:    
    spec_z[fld] = incat[incat.field('Z_flag_'+fld) == 1].field('Z_'+fld)
    photo_z[fld] = incat[incat.field('Z_flag_'+fld) == 2].field('Z_'+fld)
    all_field = list(spec_z[fld])
    all_field.extend(list(photo_z[fld]))    
    color = colors.next()
    lstyle = LF_config.lstyles.next()
    all_specz.extend( list(spec_z[fld]) )
    all_photoz.extend( list(photo_z[fld]) )
    all_z.extend(list(spec_z[fld]))
    all_z.extend(list(photo_z[fld]))

    xall = []
    yall = []
    
    print all_field
    n = hist(all_field, bins=bins, histtype='step',label=fld, lw=3, normed=False,\
              color= color)
    
    plt.xlabel( '$\mathrm{redshift}$', size = 'large' )
    plt.ylabel( '$\mathrm{Number\,of\,sources}$', size = 'large' )

    xmin = 0
    xmax = 4
    
    plt.xlim([0, 4])
    plt.xticks([0, 1, 2, 3, 4])
    #plt.ylim([ymin, ymax])
    plt.legend()
    #plt.text(xmin + (xmax-xmin)*0.5, ymax + (ymax-ymin)*0.05, fld, size='small')        

#if len(all_specz)>0:
#    n = hist(all_specz, bins=bins, histtype='step',label=fld, lw=4, normed=False, color='black')
#if len(photo_z)>0:
#    n = hist(all_photoz, bins=bins, histtype='step',label=fld, lw=4, normed=False, color='gray')
        
#plt.draw()
#plt.savefig('plots/histo_z.pdf', dpi=200)
    plt.show()