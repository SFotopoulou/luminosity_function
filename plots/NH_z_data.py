from xspec import *
from numpy import arange,mean
import matplotlib.pyplot as plt
import itertools
import pyfits
import numpy as np
import numpy.ma as ma
import matplotlib.ticker as ticker
######## MAXI
MAXI_path='/home/sotiria/Documents/Luminosity_Function/data/MAXI/'
MAXI_name='master_catalog_MAXI.fits'
MAXI_file = MAXI_path+MAXI_name
MAXI_f = pyfits.open(MAXI_file)
MAXIdata = MAXI_f[1].data

MAXI_z = MAXIdata.field('redshift')
MAXI_NH = np.log10(MAXIdata.field('Nh')*1e22)
mask = np.where(MAXI_z>0.01,0,1)
MAXI_z = ma.masked_array(MAXI_z,mask)

######## CDFS
CDFS_path='/home/sotiria/Documents/Luminosity_Function/data/Chandra-CDFS/catalogs/'
CDFS_name='master_catalog_CDFS.fits'
CDFS_file = CDFS_path+CDFS_name
CDFS_f = pyfits.open(CDFS_file)
CDFSdata = CDFS_f[1].data
##CDFSSoft = CDFSdata.field('ctr_05_2')
##CDFSHard = CDFSdata.field('ctr_2_10')*1.021
CDFSSoft = CDFSdata.field('SCts')/CDFSdata.field('SExp')
CDFSHard = CDFSdata.field('HCts')*1.007/CDFSdata.field('HExp')
CDFS_HR= (CDFSHard-CDFSSoft)/(CDFSSoft+CDFSHard) # (0.5-2)-(2-8)

CDFS_z = CDFSdata.field('zadopt')
CDFS_NH = np.log10(CDFSdata.field('NH')*1e22)
CDFS_eNH = 0.434*CDFSdata.field('e_NH')/CDFSdata.field('NH')
#print mean(CDFS_eNH)
######## HBSS
HBSS_path='/home/sotiria/Documents/Luminosity_Function/data/XMM_HBS/catalogs/'
HBSS_name='NH_master_catalog_HBSS.fits'
HBSS_file = HBSS_path+HBSS_name
HBSS_f = pyfits.open(HBSS_file)
HBSSdata = HBSS_f[1].data
HBSSSoft = HBSSdata.field('Rate_1')
HBSSHard = HBSSdata.field('Rate_2')
HBSS_HR= (-HBSSSoft*1e-2*7.89e-1+HBSSHard*1e-3*5.165)/(HBSSSoft*1e-2*7.89e-1+HBSSHard*1e-3*5.165)
HBSS_HR2 = HBSSdata.field('HR2') # (0.5-2)-(2-4.5)
#plt.plot(HBSS_HR2, HBSS_HR-0.25,'ko')
#plt.show()
HBSS_z = HBSSdata.field('z_2')
HBSS_NH = HBSSdata.field('LogNH')
####### Lockman Hole
LH_path='/home/sotiria/Documents/Luminosity_Function/data/XMM_LH/catalogs/'
LH_name='master_catalog_LH.fits'
LH_file = LH_path+LH_name
LH_f = pyfits.open(LH_file)
LH_fdata = LH_f[1].data
LHdetect = LH_fdata[LH_fdata.field('Ldet3')>10]
LHHR1 = LHdetect.field('HR1')# (0.5-2)-(2-10)
LH_z = LHdetect.field('redshift')
LH_NH = LHdetect.field('log(Abs)')
LH_eNH = LHdetect.field('e_log(Abs)_2')
#print mean(LH_eNH)
####### COSMOS
path='/home/sotiria/Documents/Luminosity_Function/data/XMM_COSMOS/catalogs/'
name='master_catalog.COSMOS.fits'
file = path+name
f = pyfits.open(file)
fdata = f[1].data

UHdetect = fdata[fdata.field('L5-10')>12]
soft = UHdetect.field('Ct.5-2')
soft_ct = UHdetect.field('Ct.5-2')/UHdetect.field('Exp.5-2')
hard = UHdetect.field('Ct2-10')
hard_ct = UHdetect.field('Ct2-10')/UHdetect.field('Exp2-10')
uhard = UHdetect.field('Ct5-10')
uhard_ct = UHdetect.field('Ct5-10')/UHdetect.field('Exp5-10')
COSMOS_z = UHdetect.field('redshift_2a')
COSMOS_HR = (hard_ct-soft_ct)/(hard_ct+soft_ct)# (0.5-2)-(2-10)
COSMOS_NH = UHdetect.field('Nh')
COSMOS_NH2 = UHdetect.field('logNH')


################################################################
#      Simulate absorption in the 5-10keV band with pyspec     #
################################################################
#             cosmology parameters ( H0, q0, lambda0 )         #
#                                                              #
#                   Using COSMO:  70. .0 .73                   # 
################################################################
Xset.chatter = 5

NHbin=np.logspace(-2,2,50) # x1e22
HighECutOff = 100 # keV
redshift = np.linspace(0,4,2)
lstyle = itertools.cycle(["dotted","dotted","dashdot","dashed","solid"])     
fig=plt.figure(1,figsize=(10,10))
fig.subplots_adjust(left=0.17, bottom=0.11, right=0.99, top=0.99,wspace=0.34, hspace=0.15)
ax = fig.add_subplot(111)       

#plt.plot(HBSS_NH,HBSS_HR-0.25,label='HBSS',linestyle='',marker='o',markersize=9,markerfacecolor='k',markeredgecolor='k')
#plt.plot(COSMOS_NH,COSMOS_HR,linestyle='',marker='s',markersize=9,markerfacecolor='gray',markeredgecolor='k')
#plt.plot(COSMOS_NH2,COSMOS_HR,label='COSMOS',linestyle='',marker='s',markersize=9,markerfacecolor='gray',markeredgecolor='k')
#plt.plot(LH_NH,LHHR1,label='LH',linestyle='^',marker='o',markersize=9,markerfacecolor='w',markeredgecolor='k')
#plt.plot(CDFS_NH,CDFS_HR,label='CDFS',linestyle='',marker='v',markersize=9,markerfacecolor='k',markeredgecolor='k')

plt.plot(HBSS_NH,HBSS_HR-0.25,label='HBSS',linestyle='',marker='o',markersize=9,markerfacecolor='b',markeredgecolor='b')
plt.plot(COSMOS_NH,COSMOS_HR,linestyle='',marker='s',markersize=9,markerfacecolor='r',markeredgecolor='r')
plt.plot(COSMOS_NH2,COSMOS_HR,label='COSMOS',linestyle='',marker='s',markersize=9,markerfacecolor='r',markeredgecolor='r')
plt.plot(LH_NH,LHHR1,label='LH',linestyle='^',marker='^',markersize=9,markerfacecolor='k',markeredgecolor='k')
plt.plot(CDFS_NH,CDFS_HR,label='CDFS',linestyle='',marker='v',markersize=9,markerfacecolor='gray',markeredgecolor='gray')

PhotonIndex_G = [1.2]
style=itertools.cycle(['-','--'])
colors=itertools.cycle(['k','gray'])

for photonindex in PhotonIndex_G:
    l_style = style.next()

    for Redshift in redshift:
        l_color = colors.next()
        HR=[]
        m0 = Model("zphabs*cutoffpl")#+const*cutoffpl+zgauss+pexrav") 
        m0.zphabs.Redshift = Redshift
        m0.cutoffpl.PhoIndex = photonindex
        m0.cutoffpl.HighECut = HighECutOff 
        m0.cutoffpl.norm = 1e6
        for NH in NHbin:
            m0.zphabs.nH = NH
            AllModels.calcFlux("0.5 2")
            XMM_Cnts052 = m0.flux[0]*1.857e11
            
            AllModels.calcFlux("2 10")
            XMM_Cnts210 = m0.flux[0]*4.53e10 
            #print (XMM_Cnts210-XMM_Cnts052)/(XMM_Cnts210+XMM_Cnts052)
            HR.append((XMM_Cnts210-XMM_Cnts052)/(XMM_Cnts210+XMM_Cnts052))
    
    
        plt.plot(np.log10(np.array(NHbin)*1e22), HR, ls =l_style,color=l_color, )
    
ax.set_ylim([-1.,1.1])
ax.set_xlim([19.5,24.5])
ax.set_xlabel("$\log(N_H/cm^{-2})$",fontsize=32)
ax.set_ylabel("HR (C$_{2-10}$ - C$_{0.5-2}$)/(C$_{2-10}$ + C$_{0.5-2}$)",fontsize=32)
plt.legend(loc=2)
#for ext in ['jpg','pdf','eps','png']:
#    plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/NH_HR_comp.'+ext)

plt.show()
#
#
##fig=plt.figure(2,figsize=(10,10))
##fig.subplots_adjust(left=0.13, bottom=0.11, right=0.99, top=0.98,wspace=0.34, hspace=0.15)
##
##ax = fig.add_subplot(111) 
##plt.plot(MAXI_z,MAXI_NH,label='MAXI',linestyle='',marker='o',markersize=9,markerfacecolor='gray',markeredgecolor='k')
##plt.plot(HBSS_z, HBSS_NH,label='HBSS',linestyle='',marker='o',markersize=9,markerfacecolor='k',markeredgecolor='k')
##plt.plot(COSMOS_z, COSMOS_NH,linestyle='',marker='s',markersize=9,markerfacecolor='gray',markeredgecolor='k')
##plt.plot(COSMOS_z,COSMOS_NH2,label='COSMOS',linestyle='',marker='s',markersize=9,markerfacecolor='gray',markeredgecolor='k')
##plt.plot(LH_z,LH_NH,label='LH',linestyle='^',marker='o',markersize=9,markerfacecolor='w',markeredgecolor='k')
##plt.plot(CDFS_z,CDFS_NH,label='CDFS',linestyle='',marker='v',markersize=9,markerfacecolor='k',markeredgecolor='k')
##ax.set_xscale('log')
##ax.set_ylim([19.5,25])
##ax.set_xlim([1e-2,6])
##ax.set_xlabel("Redshift",fontsize=32)
##ax.set_ylabel("$\log(N_H/cm^{-2})$",fontsize=32)
##plt.legend(loc=2)
##for ext in ['jpg','pdf','eps','png']:
##    plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/NH_z_comp.'+ext)
##
##plt.show()
#
#
#fig=plt.figure(3,figsize=(10,10))
#fig.subplots_adjust(left=0.17, bottom=0.11, right=0.99, top=0.99,wspace=0.34, hspace=0.15)
#ax = fig.add_subplot(111)       
#
#plt.plot(HBSS_NH,HBSS_HR-0.25,label='HBSS',linestyle='',marker='o',markersize=9,markerfacecolor='b',markeredgecolor='b')
#plt.plot(COSMOS_NH,COSMOS_HR,linestyle='',marker='s',markersize=9,markerfacecolor='r',markeredgecolor='r')
#plt.plot(COSMOS_NH2,COSMOS_HR,label='COSMOS',linestyle='',marker='s',markersize=9,markerfacecolor='r',markeredgecolor='r')
#plt.plot(LH_NH,LHHR1,label='LH',linestyle='^',marker='^',markersize=9,markerfacecolor='k',markeredgecolor='k')
#plt.plot(CDFS_NH,CDFS_HR,label='CDFS',linestyle='',marker='v',markersize=9,markerfacecolor='gray',markeredgecolor='gray')
#ax.set_ylim([-1.,1.1])
#ax.set_xlim([19.5,24.5])
#ax.set_xlabel("$\log(N_H/cm^{-2})$",fontsize=32)
#ax.set_ylabel("HR (C$_{2-10}$ - C$_{0.5-2}$)/(C$_{2-10}$ + C$_{0.5-2}$)",fontsize=32)
#plt.legend(loc=2)
#for ext in ['jpg','pdf','eps','png']:
#    plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/NH_HR_comp_c.'+ext)
#
#plt.show()
#
##
##fig=plt.figure(4,figsize=(10,10))
##fig.subplots_adjust(left=0.13, bottom=0.11, right=0.99, top=0.98,wspace=0.34, hspace=0.15)
##
##ax = fig.add_subplot(111)
##plt.plot(MAXI_z,MAXI_NH,label='MAXI',linestyle='',marker='o',markersize=9,markerfacecolor='green',markeredgecolor='green') 
##plt.plot(HBSS_z, HBSS_NH,label='HBSS',linestyle='',marker='o',markersize=9,markerfacecolor='b',markeredgecolor='b')
##plt.plot(COSMOS_z, COSMOS_NH,linestyle='',marker='s',markersize=9,markerfacecolor='r',markeredgecolor='r')
##plt.plot(COSMOS_z,COSMOS_NH2,label='COSMOS',linestyle='',marker='s',markersize=9,markerfacecolor='r',markeredgecolor='r')
##plt.plot(LH_z,LH_NH,label='LH',linestyle='^',marker='^',markersize=9,markerfacecolor='k',markeredgecolor='k')
##plt.plot(CDFS_z,CDFS_NH,label='CDFS',linestyle='',marker='v',markersize=9,markerfacecolor='gray',markeredgecolor='gray')
##ax.set_xscale('log')
##ax.set_ylim([19.5,25])
##ax.set_xlim([1e-2,6])
##ax.set_xlabel("Redshift",fontsize=32)
##ax.set_ylabel("$\log(N_H/cm^{-2})$",fontsize=32)
##plt.legend(loc=2)
##for ext in ['jpg','pdf','eps','png']:
##    plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/5-10data/NH_z_comp_c.'+ext)
##
##plt.show()