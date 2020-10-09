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

#
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

lstyles = itertools.cycle(['--','-','-','-.','-.'])
mstyles = itertools.cycle(['o','o','s','^','v'])

lcolors = itertools.cycle(['black','gray','black','black','gray'])
mcolors = itertools.cycle(['gray','black','gray','black','white'])
medgecolors = itertools.cycle(['black','black','gray','black','black'])
ecolors = itertools.cycle(['black','black','gray','black','black'])


lcolors_c = itertools.cycle(['blue','red','black','gray','green','magenta'])
mcolors_c = itertools.cycle(['green','blue','red','black','gray','magenta'])
medgecolors_c = itertools.cycle(['green','blue','red','black','gray','magenta'])
ecolors_c = itertools.cycle(['green','blue','red','black','gray','magenta'])

## Fx - z
in_path='/home/sotiria/workspace/Luminosity_Function/input_files/'
data_name='Input_Data.fits'
dfile = in_path+data_name
f = pyfits.open(dfile)
ffdata = f[1].data

fdata = ffdata[ffdata.field('redshift')>0]

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

#    Flux - redshift
###plt.plot(MAXI_z, MAXI_S,'o',label='MAXI')
###plt.plot(HBSS_z, HBSS_S,'o',label='HBSS')
###plt.plot(COSMOS_z, COSMOS_S,'o',label='COSMOS')
###plt.plot(LH_z, LH_S,'o',label='LH')
###plt.plot(CDFS_z, CDFS_S,'o', label='CDFS')
###plt.legend()
###plt.show()
###HBSS_Rc = HBSS.field('mag')
###COSMOS_Rc = COSMOS.field('mag')
###LH_Rc = LH.field('mag')
###CDFS_Rc = CDFS.field('mag')
###
###### For papers
###plt.figure(1,figsize=(12,12))
###ax = plt.subplot(111)
###s1 = plt.errorbar(HBSS_z, 10.**HBSS_S, ls = ' ',markersize = 8.5,color=mcolors.next(), marker=mstyles.next(),markeredgecolor=medgecolors.next())
###s2 = plt.errorbar(COSMOS_z, 10.**COSMOS_S, ls = ' ',markersize = 8.5,color=mcolors.next(), marker=mstyles.next(),markeredgecolor=medgecolors.next())
###s3 = plt.errorbar(LH_z, 10.**LH_S, ls = ' ',markersize = 8.5,color=mcolors.next(), marker=mstyles.next(),markeredgecolor=medgecolors.next())
###s4 = plt.errorbar(CDFS_z, 10.**CDFS_S, ls = ' ',markersize = 8.5,color=mcolors.next(), marker=mstyles.next(),markeredgecolor=medgecolors.next())
###plt.legend([s1,s2,s3,s4],['HBSS','COSMOS','LH','CDFS'],scatterpoints=1,loc=3)
###
###ax.set_yscale('log')
###ax.set_xscale('log')
###plt.xlabel(r'$\log(z)$')
###plt.ylabel(r'$\log(F_{5-10keV}/erg\cdot s^{-1}\cdot cm^{-2})$')
###plt.xlim([1e-2,7])
###plt.ylim([1e-16,1e-11])
###plt.draw()
###plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/Fx_z.jpg',dpi=300)
###plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/Fx_z.eps')
###plt.show()

#
##    For presentations
#plt.figure(1,figsize=(12,12))
#ax = plt.subplot(111)
#s1_c = plt.errorbar(HBSS_z, 10.**HBSS_S, ls = ' ',markersize = 10,color=mcolors_c.next(), marker=mstyles.next(),markeredgecolor=medgecolors_c.next())
#s2_c = plt.errorbar(COSMOS_z, 10.**COSMOS_S, ls = ' ',markersize = 10,color=mcolors_c.next(), marker=mstyles.next(),markeredgecolor=medgecolors_c.next())
#s3_c = plt.errorbar(LH_z, 10.**LH_S, ls = ' ',markersize = 10,color=mcolors_c.next(), marker=mstyles.next(),markeredgecolor=medgecolors_c.next())
#s4_c = plt.errorbar(CDFS_z, 10.**CDFS_S, ls = ' ',markersize = 10,color=mcolors_c.next(), marker=mstyles.next(),markeredgecolor=medgecolors_c.next())
#plt.legend([s1_c,s2_c,s3_c,s4_c],['HBSS','COSMOS','LH','CDFS'],scatterpoints=1,loc=3)
#
#ax.set_yscale('log')
#ax.set_xscale('log')
#plt.xlabel(r'$\log(z)$')
#plt.ylabel(r'$\log(F_{5-10keV}/erg\cdot s^{-1}\cdot cm^{-2})$')
#plt.xlim([1.5e-2,7])
#plt.ylim([2e-16,2e-12])
#plt.draw()
#plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/Fx_z_c.jpg',dpi=300)
#plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/Fx_z_c.eps')
#plt.show()

#### Area Curve
area_name = 'area_curve.fits'
afile = in_path+area_name
a = pyfits.open(afile)
adata = a[1].data
##

HBSS_a = adata[adata.field('HBSS_FLUX')>0]
COSMOS_a = adata[adata.field('COSMOS_FLUX')>0]
LH_a = adata[adata.field('LH_FLUX')>0]
CDFS_a = adata[adata.field('CDFS_FLUX')>0]
MAXI_a = adata[adata.field('MAXI_FLUX')>0]

HBSS_aFlux = log10(HBSS_a.field('HBSS_FLUX'))
HBSS_aArea = HBSS_a.field('HBSS_AREA')
print "HBSS:", min(HBSS_aFlux), max(HBSS_aFlux)

COSMOS_aFlux = log10(COSMOS_a.field('COSMOS_FLUX'))
COSMOS_aArea = COSMOS_a.field('COSMOS_AREA')
print "COSMOS:", min(COSMOS_aFlux), max(COSMOS_aFlux)

LH_aFlux = log10(LH_a.field('LH_FLUX'))
LH_aArea = LH_a.field('LH_AREA')
print "LH:", min(LH_aFlux), max(LH_aFlux)

CDFS_aFlux = log10(CDFS_a.field('CDFS_FLUX'))
CDFS_aArea = CDFS_a.field('CDFS_AREA')
print "CDFS:", min(CDFS_aFlux), max(CDFS_aFlux)

MAXI_aFlux = log10(MAXI_a.field('MAXI_FLUX'))
MAXI_aArea = MAXI_a.field('MAXI_AREA')
print "MAXI:", min(MAXI_aFlux), max(MAXI_aFlux)
#
#plt.figure(2,figsize=(12,12))
#ax2 = plt.subplot(111)
a1 = plt.plot(10.**HBSS_aFlux, HBSS_aArea, lw=4, ls=lstyles.next(),color=lcolors.next())
a2 = plt.plot(10.**COSMOS_aFlux, COSMOS_aArea, lw=4,ls=lstyles.next(),color=lcolors.next())
a3 = plt.plot(10.**LH_aFlux, LH_aArea, lw=4,ls=lstyles.next(),color=lcolors.next())
a4 = plt.plot(10.**CDFS_aFlux, CDFS_aArea, lw=4,ls=lstyles.next(),color=lcolors.next())
a5 = plt.plot(10.**MAXI_aFlux, MAXI_aArea, lw=4,ls=lstyles.next(),color=lcolors.next())

plt.legend([a5,a1,a2,a3,a4],['MAXI','HBSS','COSMOS','LH','CDFS'],loc=2)


plt.ylabel(r'$\log(area/deg^2)$',fontsize=32)
plt.xlabel(r'$\log(F_{5-10keV}/erg\cdot s^{-1}\cdot cm^{-2})$',fontsize=32)
plt.ylim([1e-2,5e4])
plt.xlim([1e-16,1e-9])
ax.set_yscale('log')
ax.set_xscale('log')
plt.draw()
plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/area_curve.jpg',dpi=300)
plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/area_curve.eps')
plt.show()
#
ax2 = plt.subplot(111)
a1 = plt.plot(10.**HBSS_aFlux, HBSS_aArea, lw=4,color=lcolors_c.next())
a2 = plt.plot(10.**COSMOS_aFlux, COSMOS_aArea, lw=4,color=lcolors_c.next())
a3 = plt.plot(10.**LH_aFlux, LH_aArea, lw=4,color=lcolors_c.next())
a4 = plt.plot(10.**CDFS_aFlux, CDFS_aArea, lw=4,color=lcolors_c.next())
a5 = plt.plot(10.**MAXI_aFlux, MAXI_aArea, lw=4,color=lcolors_c.next())

plt.legend([a5,a1,a2,a3,a4],['MAXI', 'HBSS','COSMOS','LH','CDFS'],loc=2)
ax2.set_yscale('log')
ax2.set_xscale('log')
plt.ylabel(r'$\log(area/deg^2)$',fontsize=32)
plt.xlabel(r'$\log(F_{5-10keV}/erg\cdot s^{-1}\cdot cm^{-2})$',fontsize=32)
plt.ylim([1e-2,5e4])
plt.xlim([1e-16,1e-9])
plt.draw()
plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/area_curve_c.jpg',dpi=300)
plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/area_curve_c.eps')
plt.show()
#
## logN - logS
fields = [MAXI_S,HBSS_S,COSMOS_S,LH_S,CDFS_S]
field_name = itertools.cycle(['MAXI','HBSS','COSMOS','LH','CDFS'])
area = itertools.cycle([MAXI_aArea,HBSS_aArea,COSMOS_aArea,LH_aArea,CDFS_aArea])
area_Flux = itertools.cycle([MAXI_aFlux,HBSS_aFlux,COSMOS_aFlux,LH_aFlux,CDFS_aFlux])
#plt.figure(3,figsize=(12,12))
#ax3 = plt.subplot(111)

for fx in fields:
    Area = area.next()
    A_Flux = area_Flux.next()
    fname = field_name.next()
    N = []
    Nerr = []
    print fname
    if fname=='HBSS':
        bins = arange(min(fx)-0.1, max(fx)+0.1, 0.33)
    if fname=='COSMOS':
        bins = arange(min(fx)-0.1, max(fx)+0.1, 0.3)
    if fname=='LH':
        bins = arange(min(fx)-0.1, log10(5e-14)+0.1, 0.3)
    if fname=='CDFS':
        bins = arange(min(fx)-0.1, max(fx)+0.1, 0.3)
    if fname=='MAXI':
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
        
        print count
        N.append(sum*((10.**bin[i])/(1.0e-14))**1.5)
        Nerr.append(sqrt(sum_err)*((10.**bin[i])/(1.0e-14))**1.5)
    plt.errorbar(10.**bin,N,yerr=Nerr,label=fname,markersize = 9.5,marker=mstyles.next(),markeredgecolor=medgecolors.next(),markerfacecolor=mcolors.next(),ecolor=ecolors.next(),ls=' ')
plt.legend()
plt.yscale('log',nonposy='clip')
plt.xscale('log',nonposx='clip')
plt.ylim([1.2,990])
plt.ylabel(r'$\log(N(>S)\cdot S^{1.5}_{14} /deg^{-2})$',fontsize=32)
plt.xlabel(r'$\log(S_{5-10keV}/erg\cdot s^{-1}\cdot cm^{-2})$',fontsize=32)
plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/logNlogS.jpg',dpi=300)
plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/logNlogS.eps')
plt.show()
#    plt.errorbar(10.**bin,N,yerr=Nerr,label=fname,markersize = 9.5,marker=mstyles.next(),markeredgecolor=medgecolors_c.next(),markerfacecolor=mcolors_c.next(),ecolor=ecolors_c.next(),ls=' ')
#
#plt.legend()
#plt.yscale('log',nonposy='clip')
#plt.xscale('log',nonposx='clip')
##plt.ylim([1.2,990])
#plt.ylabel(r'$\log(N(>S)\cdot S^{1.5}_{14} /deg^{-2})$',fontsize=32)
#plt.xlabel(r'$\log(S_{5-10keV}/erg\cdot s^{-1}\cdot cm^{-2})$',fontsize=32)
#plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/logNlogS_c.jpg',dpi=300)
#plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/logNlogS_c.eps')
#plt.show()

### Plot L-z
field = itertools.cycle([MAXI_S, HBSS_S,COSMOS_S,LH_S,CDFS_S])
field_z = itertools.cycle([MAXI_z, HBSS_z,COSMOS_z,LH_z,CDFS_z])
field_name = itertools.cycle(['MAXI','HBSS','COSMOS','LH','CDFS'])
plt.figure(4,figsize=(10,10))
import numpy.ma as ma
import numpy as np
s = Source('none')
for i in [1,2,3,4,5]:
    F = 10.**field.next()
    z = field_z.next()
    Lx, Lx_err = s.get_luminosity(F,ones(len(F)),z)
    maskz = np.where(np.array(z)>0.0099, 0 , 1)
    z=ma.masked_array(np.array(z),maskz)

    Lx = np.array(Lx)
    maskL = np.where(np.array(Lx)>42.0, 0 , 1)
    Lx = ma.masked_array(np.array(Lx),maskL)
    
    plt.errorbar(z,power(10.0,Lx),label=field_name.next(),markersize = 9.0,marker=mstyles.next(),markeredgecolor=medgecolors.next(),markerfacecolor=mcolors.next(),ecolor=ecolors.next(),ls=' ')

plt.legend(loc=4)
plt.yscale('log',nonposy='clip')
plt.xscale('log',nonposx='clip')
plt.ylim([3e41, 1e46])
plt.xlim([1e-2,1e1])
plt.ylabel(r'$\log(L_X /erg\cdot s^{-1})$', fontsize=32)
plt.xlabel(r'$\log(z)$',fontsize=32)
plt.draw()
plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/Lx_z.jpg',dpi=300)
plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/Lx_z.eps')
plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/Lx_z.png')

plt.show()

field = itertools.cycle([MAXI_S, HBSS_S,COSMOS_S,LH_S,CDFS_S])
field_z = itertools.cycle([MAXI_z, HBSS_z,COSMOS_z,LH_z,CDFS_z])
field_name = itertools.cycle(['MAXI','HBSS','COSMOS','LH','CDFS'])
plt.figure(4,figsize=(10,10))
ax4 = plt.subplot(111)

s = Source('none')
for i in [1,2,3,4,5]:
    F = 10.**field.next()
    z = field_z.next()
   # print F,z
    Lx, Lx_err = s.get_luminosity(F,ones(len(F)),z)
    maskz = np.where(np.array(z)>0.0099, 0 , 1)
    z=ma.masked_array(np.array(z),maskz)

    Lx = np.array(Lx)
    maskL = np.where(np.array(Lx)>42.0, 0 , 1)
    Lx = ma.masked_array(np.array(Lx),maskL)
    
    #print Lx
    plt.errorbar(z,power(10.0,Lx),label=field_name.next(),markersize = 9.0,marker=mstyles.next(),markeredgecolor=medgecolors_c.next(),markerfacecolor=mcolors_c.next(),ecolor=ecolors_c.next(),ls=' ')
plt.legend(loc=4)
plt.yscale('log',nonposy='clip')
plt.xscale('log',nonposx='clip')

plt.ylabel(r'$\log(L_X /erg\cdot s^{-1})$', fontsize=32)
plt.xlabel(r'$\log(z)$',fontsize=32)
plt.ylim([3e41, 1e46])
plt.xlim([1e-2,1e1])
plt.draw()
plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/Lx_z_c.jpg',dpi=300)
plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/Lx_z_c.eps')
plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/Lx_z_c.png')

plt.show()
# For eRosita
##"""
##Make a compund path -- in this case two simple polygons, a rectangle
##and a triangle.  Use CLOSEOPOLY and MOVETO for the different parts of
##the compound path
##"""
##from matplotlib.path import Path
##from matplotlib.patches import PathPatch
##
##vertices = []
##codes = []
##
##codes = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
##vertices = [(0.01,2.1e40), (0.01,2.21e46), (5, 1.51e52), (5, 1.43e46), (0,0)]
##
##vertices = array(vertices, float)
##path = Path(vertices, codes)
##
##pathpatch = PathPatch(path, facecolor='yellow', edgecolor='yellow')
##
##
##ax.add_patch(pathpatch)
##
##ax.dataLim.update_from_data_xy(vertices)
##
##plt.legend(loc=2)
##plt.yscale('log',nonposy='clip')
##plt.xscale('log',nonposx='clip')
##plt.xlim([0.015,4.5])
##plt.ylim(1e40,6e45)
##plt.ylabel(r'$\log(L_X /erg\cdot s^{-1})$')
##plt.xlabel(r'$\log(z)$')
##
##plt.show()

## Plot fx-fopt
#plt.figure(5,figsize=(12,12))
ax5 = plt.subplot(111)
field_R = itertools.cycle([HBSS_Rc,COSMOS_Rc,LH_Rc,CDFS_Rc])
for i in [1,2,3,4]:
    Fx = 10.**field.next()
    # z = field_z.next()
    Rc = field_R.next()
#    plt.errorbar(Fx,Rc,label=field_name.next(),markersize = 8.5,marker=mstyles.next(),markeredgecolor=medgecolors.next(),markerfacecolor=mcolors.next(),ecolor=lcolors.next(),ls=' ')
    plt.errorbar(Fx,Rc,label=field_name.next(),markersize = 10,marker=mstyles.next(),markeredgecolor=medgecolors_c.next(),markerfacecolor=mcolors_c.next(),ecolor=lcolors_c.next(),ls=' ')

plt.legend()#loc=4)
plt.xlim([4e-16,5e-12])
plt.ylim(14,28)
ax5.set_xscale('log')
ax5.set_yscale('linear')
plt.axhline(y=22.5,color='black',linestyle=':')
plt.axvline(x=8e-15,color='black',linestyle='--')
plt.xticks(visible=True)
plt.yticks(visible=True)

R=arange(10,30,1)
plt.plot(10**(-4.5-R/2.5),R,color='k')
plt.plot(10**(-6.5-R/2.5),R,color='k')

ax5.yaxis.set_major_locator(MultipleLocator(2))
ax5.yaxis.set_minor_locator(MultipleLocator(0.5))

plt.xlabel(r'$\log(F_X /erg\cdot s^{-1}\cdot cm^{-2})$')
plt.ylabel(r'$Rc\, mag\, (AB)$')
plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/fx-fopt.jpg',dpi=300)
plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/fx-fopt.eps')
plt.show()
        
plt.axhline(y=22.5,color='black',linestyle=':')
plt.axvline(x=8e-15,color='black',linestyle='--')
plt.xticks(visible=True)
plt.yticks(visible=True)

R=arange(10,30,1)
plt.plot(10**(-4.5-R/2.5),R,color='k')
plt.plot(10**(-6.5-R/2.5),R,color='k')

ax5.yaxis.set_major_locator(MultipleLocator(2))
ax5.yaxis.set_minor_locator(MultipleLocator(0.5))

plt.xlabel(r'$\log(F_X /erg\cdot s^{-1}\cdot cm^{-2})$')
plt.ylabel(r'$Rc\, mag\, (AB)$')
plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/fx-fopt_c.jpg',dpi=300)
plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/fx-fopt_c.eps')
plt.show()