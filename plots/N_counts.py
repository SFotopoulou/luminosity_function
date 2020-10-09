import sys
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/no_uncertainties_included/MLE/MLE_modules/')

import itertools
import pyfits
from numpy import arange,log10,sqrt,array, power,ones
import matplotlib.pyplot as plt
from Source import Source
from matplotlib.ticker import MultipleLocator

fig = plt.figure(figsize=(10,10))
fig.subplots_adjust(left=0.175, right=0.95, wspace=0.05, hspace=0.05)
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
mstyles = itertools.cycle(['o','o','s','o','v'])

lcolors = itertools.cycle(['gray','black','gray','black','black'])
mcolors = itertools.cycle(['gray','black','gray','white','black'])
medgecolors = itertools.cycle(['black','black','gray','black','black'])

lcolors_c = itertools.cycle(['red','blue','cyan','green','magenta','black'])
mcolors_c = itertools.cycle(['red','blue','cyan','green','magenta','black'])
medgecolors_c = itertools.cycle(['black','black','black','black','magenta','black'])

### Fx - z
in_path='/home/sotiria/workspace/Luminosity_Function/input_files/'
data_name='Input_Data.fits'
dfile = in_path+data_name
f = pyfits.open(dfile)
fdata = f[1].data
zmin = 0.01
zmax = 5

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

HBSS_z = HBSS.field('redshift')
COSMOS_z = COSMOS.field('redshift')
LH_z = LH.field('redshift')
CDFS_z = CDFS.field('redshift')
MAXI_z = MAXI.field('redshift')
#


#
##### Area Curve
area_name = 'area_curve.fits'
afile = in_path+area_name
a = pyfits.open(afile)
adata = a[1].data
###
HBSS_a = adata[adata.field('HBSS_FLUX')>0]
COSMOS_a = adata[adata.field('COSMOS_FLUX')>0]
LH_a = adata[adata.field('LH_FLUX')>0]
CDFS_a = adata[adata.field('CDFS_FLUX')>0]
MAXI_a = adata[adata.field('MAXI_FLUX')>0]

HBSS_aFlux = log10(HBSS_a.field('HBSS_FLUX'))
HBSS_aArea = HBSS_a.field('HBSS_AREA')
#print "HBSS:", min(HBSS_aFlux), max(HBSS_aFlux)

COSMOS_aFlux = log10(COSMOS_a.field('COSMOS_FLUX'))
COSMOS_aArea = COSMOS_a.field('COSMOS_AREA')
#print "COSMOS:", min(COSMOS_aFlux), max(COSMOS_aFlux)

LH_aFlux = log10(LH_a.field('LH_FLUX'))
LH_aArea = LH_a.field('LH_AREA')
#print "LH:", min(LH_aFlux), max(LH_aFlux)

CDFS_aFlux = log10(CDFS_a.field('CDFS_FLUX'))
CDFS_aArea = CDFS_a.field('CDFS_AREA')
#print "CDFS:", min(CDFS_aFlux), max(CDFS_aFlux)

MAXI_aFlux = log10(MAXI_a.field('MAXI_FLUX'))
MAXI_aArea = MAXI_a.field('MAXI_AREA')

### logN - logS
if zmin<0.2:
    fields = [MAXI_S,HBSS_S,COSMOS_S,LH_S,CDFS_S]
    field_name = itertools.cycle(['MAXI','HBSS','COSMOS','LH','CDFS'])
    area = itertools.cycle([MAXI_aArea,HBSS_aArea,COSMOS_aArea,LH_aArea,CDFS_aArea])
    area_Flux = itertools.cycle([MAXI_aFlux,HBSS_aFlux,COSMOS_aFlux,LH_aFlux,CDFS_aFlux])

else:
    fields = [HBSS_S,COSMOS_S,LH_S,CDFS_S]
    field_name = itertools.cycle(['HBSS','COSMOS','LH','CDFS'])
    area = itertools.cycle([HBSS_aArea,COSMOS_aArea,LH_aArea,CDFS_aArea])
    area_Flux = itertools.cycle([HBSS_aFlux,COSMOS_aFlux,LH_aFlux,CDFS_aFlux])


for fx in fields:
    Area = area.next()
    A_Flux = area_Flux.next()
    fname = field_name.next()
    N = []
    Nerr = []
    print(fname)

    if fname=='HBSS' and len(fx)>0:
        bins = arange(min(fx)-0.1, max(fx)+0.1, 0.33)
    if fname=='COSMOS' and len(fx)>0:
        bins = arange(min(fx)-0.1, max(fx)+0.1, 0.3)
    if fname=='LH' and len(fx)>0:
        bins = arange(min(fx)-0.1, log10(5e-14)+0.1, 0.3)
    if fname=='CDFS' and len(fx)>0:
        bins = arange(min(fx)-0.1, max(fx)+0.1, 0.3)
    if fname=='MAXI' and len(fx)>0:
        bins = arange(min(fx)-0.1, max(fx)+0.1, 0.25)
    
    bin = bins[:-1]
    #bin = bin + logFx_step/2.0

    for i in arange(0,len(bins)-1):
        count = 0.
        sum = 0.
        sum_err = 0.
        for f in fx:
            if f > bins[i]:
                omega = get_area(A_Flux,Area,f)
                if omega>0:
                    count = count+1
                    Omg = 1./omega
                    sum = sum+Omg
                    sum_err = sum_err+(Omg)**2.
        
        print(count)
        N.append(sum*((10.**bin[i])/(1.0e-14))**1.5)
        Nerr.append(sqrt(sum_err)*((10.**bin[i])/(1.0e-14))**1.5)
    plt.errorbar(10.**bin,N,yerr=Nerr,label=fname,markersize = 9.5,marker=mstyles.next(),markeredgecolor=medgecolors.next(),markerfacecolor=mcolors.next(),ecolor=lcolors.next(),ls=' ')
plt.legend()
plt.yscale('log',nonposy='clip')
plt.xscale('log',nonposx='clip')
plt.ylim([1.2,990])
plt.ylabel(r'$\log(N(>S)\cdot S^{1.5}_{14} /deg^{-2})$')
plt.xlabel(r'$\log(S_{5-10keV}/erg\cdot s^{-1}\cdot cm^{-2})$')
plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/tlogNlogS.jpg',dpi=300)
plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/tlogNlogS.eps')
plt.show()
    #plt.errorbar(10.**bin,N,yerr=Nerr,label=field_name.next(),markersize = 10,marker=mstyles.next(),markeredgecolor=medgecolors_c.next(),markerfacecolor=mcolors_c.next(),ecolor=lcolors_c.next(),ls=' ')
#plt.legend()
#plt.yscale('log',nonposy='clip')
#plt.xscale('log',nonposx='clip')
##plt.ylim([1.2,990])
#plt.ylabel(r'$\log(N(>S)\cdot S^{1.5}_{14} /deg^{-2})$')
#plt.xlabel(r'$\log(S_{5-10keV}/erg\cdot s^{-1}\cdot cm^{-2})$')
#plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/logNlogS.jpg',dpi=300)
#plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/logNlogS.eps')
#plt.show()
