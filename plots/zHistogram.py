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

fields= LF_config.fields # ['MAXI','HBSS','COSMOS', 'LH', 'X_CDFS', 'AEGIS']

fig = plt.figure(figsize=(15,10))
fig.subplots_adjust(left=0.09, top=0.96, bottom=0.10, right=0.98, wspace=0.22, hspace=0.22)   

bins = np.linspace(0,4, 4/0.25)#'freedman'

spec_z = {}
photo_z = {}

all_specz = []
all_photoz = []
all_z = []

for fld in fields:    
    file_name = '/home/Sotiria/Dropbox/transfer/XLF_output_files/LDDE_withUnc_MHXCLAXC_ztype_all_04_4146.out.fits'
 #LF_config.inpath + fld #+ LF_config.fname
    incat = pyfits.open(file_name)[1].data
    spec_z[fld] = incat[incat.field('Z_flag_'+fld) == 1].field('Z_'+fld)
    photo_z[fld] = incat[incat.field('Z_flag_'+fld) == 2].field('Z_'+fld)    
    
    all_specz.extend( list(spec_z[fld]) )
    all_photoz.extend( list(photo_z[fld]) )
    all_z.extend(list(spec_z[fld]))
    all_z.extend(list(photo_z[fld]))

for fld in fields:    
    indx = fields.index(fld)+1
    ax = fig.add_subplot(2, 4, indx)
    
    xall = []
    yall = []
    
    if len(spec_z[fld])>0:
        n = hist(spec_z[fld], bins=bins, histtype='step',label=fld, lw=4, normed=False, color='black')
    
    x = n[1]
    y = n[0]
    
    if len(photo_z[fld])>0:
        n = hist(photo_z[fld], bins=bins, histtype='step',label=fld, lw=4, normed=False, color='gray')
    
    if indx == 5: 
        plt.xlabel( '$\mathrm{redshift}$', size = 'large' )
        ax.xaxis.set_label_coords(9, -0.25)
    if indx == 1: 
        plt.ylabel( '$\mathrm{Number\,of\,sources}$', size = 'large' )
        ax.yaxis.set_label_coords(-0.2, -0.25)

    xmin = 0
    xmax = 4
    
    ymin = np.min( y )
    ymax = np.max( y ) + 2.5
    
    plt.xlim([0, 4])
    plt.xticks([0, 1, 2, 3, 4])
    plt.ylim([ymin, ymax])
    
    plt.text(xmin + (xmax-xmin)*0.5, ymax + (ymax-ymin)*0.05, fld, size='small')        
        
#plt.draw()
#plt.savefig('plots/histo_z.pdf', dpi=200)
plt.show()
    
print 'done'