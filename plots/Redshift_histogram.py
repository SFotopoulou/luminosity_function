import sys
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/no_uncertainties_included/MLE/MLE_modules/')
import itertools
import pyfits
from numpy import arange,log10,sqrt,array, power,ones
import matplotlib.pyplot as plt
from Source import Source
from matplotlib.ticker import MultipleLocator
import numpy.ma as ma
import numpy as np

in_path='/home/sotiria/workspace/Luminosity_Function/input_files/'
data_redshift='Input_Data.fits'
dfile = in_path+data_redshift
f = pyfits.open(dfile)
fdata = f[1].data
zmin = 0.01
zmax = 4
Lmin = 42.0
Lmax = 46.0

HBSS = fdata[fdata.field('field')==1]
COSMOS = fdata[fdata.field('field')==2]
LH = fdata[fdata.field('field')==3]
CDFS = fdata[fdata.field('field')==4]
MAXI = fdata[fdata.field('field')==5]

HBSS = HBSS[HBSS.field('redshift')<zmax]
COSMOS = COSMOS[COSMOS.field('redshift')<zmax]
LH = LH[LH.field('redshift')<zmax]
CDFS = CDFS[CDFS.field('redshift')<zmax]
MAXI = MAXI[MAXI.field('redshift')<zmax]

HBSS = HBSS[HBSS.field('redshift')>zmin]
COSMOS = COSMOS[COSMOS.field('redshift')>zmin]
LH = LH[LH.field('redshift')>zmin]
CDFS = CDFS[CDFS.field('redshift')>zmin]
MAXI = MAXI[MAXI.field('redshift')>zmin]

HBSS_S = log10(HBSS.field('Flux'))
COSMOS_S = log10(COSMOS.field('Flux'))
LH_S = log10(LH.field('Flux'))
CDFS_S = log10(CDFS.field('Flux'))
MAXI_S = log10(MAXI.field('Flux'))

HBSS_zFlag = HBSS.field('redshift_flag')
COSMOS_zFlag = COSMOS.field('redshift_flag')
LH_zFlag = LH.field('redshift_flag')
CDFS_zFlag = CDFS.field('redshift_flag')
MAXI_zFlag = MAXI.field('redshift_flag')
#
HBSS_z = HBSS.field('redshift')
COSMOS_z = COSMOS.field('redshift')
LH_z = LH.field('redshift')
CDFS_z = CDFS.field('redshift')
MAXI_z = MAXI.field('redshift')
#
d = Source('raw')
field_redshift = itertools.cycle(['MAXI','HBSS','COSMOS','LH','CDFS','all fields'])
Flux = [MAXI_S,HBSS_S,COSMOS_S,LH_S,CDFS_S] 
Phz = [MAXI_z,HBSS_z,COSMOS_z,LH_z,CDFS_z]
PhzFlag = [MAXI_zFlag,HBSS_zFlag,COSMOS_zFlag,LH_zFlag,CDFS_zFlag]

all_fields = np.concatenate(Phz)
all_flags = np.concatenate(PhzFlag)
all_fluxes = np.concatenate(Flux)

Flux = [MAXI_S,HBSS_S,COSMOS_S,LH_S,CDFS_S,all_fluxes] 
Phz = [MAXI_z,HBSS_z,COSMOS_z,LH_z,CDFS_z,all_fields]
PhzFlag = [MAXI_zFlag,HBSS_zFlag,COSMOS_zFlag,LH_zFlag,CDFS_zFlag,all_flags]

fig=plt.figure(1, figsize=(15,10))
fig.subplots_adjust(left=0.105, bottom=0.11, right=0.98, top=0.98,wspace=0.30, hspace=0.23)

for flux, redshift, flag in zip(Flux, Phz, PhzFlag):
    #print len(redshift)
    i = Phz.index(redshift)
    fredshift = field_redshift.next()
    
    luminosities, err = d.get_luminosity(np.power(10.,flux),len(flux)*[0],redshift)
    print len(luminosities)
    
    mask_lum = np.where(np.array(luminosities)>42,0,1)
#    print mask_lum
    All = ma.masked_array(redshift, mask_lum)
#    print len(All.compressed())
    ax = fig.add_subplot(2,3,i+1)
    ax.hist(All, bins=np.linspace(0,4,10),  histtype='step', color= 'black',normed=False,label=fredshift, ls='solid')

    mask_spec = np.where(( (luminosities>42.0) & (flag==1)),0,1)

    
    Sp_redshift = ma.masked_array(redshift, mask_spec)
#    print len(Sp_redshift)
#    print len(Sp_redshift.compressed())
    if len(Sp_redshift.compressed())==len(Sp_redshift):
        pass
    elif len(Sp_redshift.compressed())>0:
        ax.hist(Sp_redshift.compressed(), bins=np.linspace(0,4,10), color= 'gray', normed=False,histtype='step', ls='dashed')

    mask_phot = np.where( ((luminosities>42.0) & (flag==2) ),0,1)    
    Ph_redshift = ma.masked_array(redshift, mask_phot)
#    print len(Ph_redshift)
#    print len(Ph_redshift.compressed())
    if len(Ph_redshift.compressed())>0:
        ax.hist(Ph_redshift.compressed(), bins=np.linspace(0,4,10),color= 'red', normed=False, histtype='step', ls='dotted')

    ax.annotate(fredshift, (0.5, 0.85) , xycoords='axes fraction', fontstyle='oblique', fontsize='medium', )
    
    if i == 4:
        plt.xlabel('Redshift', fontsize='x-large')
        ax.xaxis.set_label_coords(0.5, -0.15)
    if i == 3:
        plt.ylabel(r'Number of sources', fontsize='x-large')
        ax.yaxis.set_label_coords(-0.25, 1.0)
    
  # if i == 0 or i==3:
  #      plt.yticks(np.linspace(0,100,5))
#    if i == 3 or i==4 or i==5:
    plt.xticks([0.0, 1.0,  2.0,  3.0, 4.0])
#    plt.xlim([-0.5, 4.5])  
  #  plt.ylim([0,60])

for ext in ['jpg','pdf','eps','png']:
    plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/zHisto.'+ext)

plt.show()
