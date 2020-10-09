import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.integrate import quad, simps
import matplotlib.pyplot as plt
import warnings
import itertools
from scipy import interpolate
from astropy.io import fits

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
   

    ## External data ###
    fig = plt.figure(figsize=(10,10))
    fig.subplots_adjust(left=0.18, bottom=0.18, right=0.95, wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(111)
    lstyles = itertools.cycle(['--','-'])
    mstyles = itertools.cycle(['o','s'])
    
    lcolors_c = itertools.cycle(['blue','orange'])
    mcolors_c = itertools.cycle(['blue','orange'])
    medgecolors_c = itertools.cycle(['blue','orange'])
    ecolors_c = itertools.cycle(['blue','orange'])
    
    def credible_interval(distribution, level, bin_=100):
        pdf, bins = np.histogram(distribution, bins = bin_, normed=True)
        # credible interval
        bins = bins[:-1]
        binwidth = bins[1]-bins[0]
        idxs = pdf.argsort()
        idxs = idxs[::-1]
        credible_interval = idxs[(np.cumsum(pdf[idxs])*binwidth < level/100.).nonzero()]
        idxs = idxs[::-1] # reverse
        low = min( sorted(credible_interval) )
        high = max( sorted(credible_interval) )
        min_val = bins[low]
        max_val = bins[high]
        # mode
        dist_bin = np.array([bins[i]+(bins[1]-bins[0])/2. for i in range(0,len(bins)-1)])
        mode = dist_bin[np.where(pdf==max(pdf))][0]
        
        return min_val, max_val, mode    
    
    
    def get_area(fx):
        return a_curve[fx]
    
    fields = ['X_North', 'X_South']
    field_name = itertools.cycle(['XXL-N','XXL-S'])
    area_curve = '/home/Sotiria/Documents/XXL/area_curve/area.txt'
    
    Hard_flux, North_area, South_area = np.loadtxt(area_curve,unpack=True,usecols=(0,3,4))
    
    a_curve = {}
    a_curve["North_flux"] = Hard_flux
    N_area = interpolate.interp1d(np.log10(Hard_flux), North_area, bounds_error=False, fill_value=0.0)
    S_area = interpolate.interp1d(np.log10(Hard_flux), South_area, bounds_error=False, fill_value=0.0)
    
    
    North_data = fits.open('/home/Sotiria/Documents/XXL/1000_brightest/best_photoz/merge_cat/North_Ilbert_1000_AGN_photoz.fits')[1].data
    South_data = fits.open('/home/Sotiria/Documents/XXL/1000_brightest/best_photoz/merge_cat/South_Ilbert_1000_AGN_photoz.fits')[1].data
    
    North_flux = North_data.field('CDflux_1')
    South_flux = South_data.field('CDflux_1')

    bins = np.linspace(np.log10(4.7e-14), np.log10(2e-12), 15)#area_xxl = itertools.cycle([26.9, 23.6])
    for field in fields:
        fname = field_name.next()
        if fname=='XXL-N':
            fx = np.log10(North_flux)    
            N = []
            Nerr = []
        if fname=='XXL-S':
            fx = np.log10(South_flux)    
            N = []
            Nerr = []
        bin = bins[:-1]
    
        for i in np.arange(0,len(bins)-1):
            count = 0.
            sum = 0.
            sum_err = 0.
            for f in fx:
                if f > bins[i]:
                    if field=='X_North':
                        omega = N_area(f)
                        
                    if field=='X_South':
                        omega = S_area(f)    
                        
                    if omega>0:
                        count = count+1
                        #print omega, count
    
                        Omg = 1./omega
                        sum = sum+Omg
                        sum_err = sum_err+(Omg)**2.
            
            
            N.append(sum)#*((10.**bin[i])/(1.0e-14))**1.5)
            Nerr.append(np.sqrt(sum_err))#*((10.**bin[i])/(1.0e-14))**1.5)
        if fname=='XXL-N':
            np.savetxt('XXL1000_North.dat', zip(bin, np.log10(N), np.log10(Nerr)))
            north = plt.errorbar(10.**bin, N, yerr=Nerr, label=fname,markersize = 12.5,marker=mstyles.next(),markeredgecolor=medgecolors_c.next(),markerfacecolor=mcolors_c.next(),ecolor=ecolors_c.next(),ls=' ')
            
        if fname=='XXL-S':
            np.savetxt('XXL1000_South.dat', zip(bin, np.log10(N), np.log10(Nerr)))
    
            south = plt.errorbar(10.**bin, N, yerr=Nerr, label=fname,markersize = 12.5,marker=mstyles.next(),markeredgecolor=medgecolors_c.next(),markerfacecolor=mcolors_c.next(),ecolor=ecolors_c.next(),ls=' ')
    print N

    from scipy.integrate import quad
    
    def line(X, A, B):
        return 10**(A*X+B)
    
    alpha = -2.37
    alpha_err = 0.08
    beta = -20.09
    beta_err = 1.03
    
    mean = np.array([ alpha, beta])
    cov_matrix = np.array([[ alpha_err*alpha_err ,  0.],\
                  [ 0,  beta_err*beta_err        ]])
        
    plt.legend([north, south, cosmos, atlas, cdfs], ['XXL-N', 'XXL-S', 'COSMOS', 'ATLAS', 'XMM-CDFS'])
    plt.yscale('log',nonposy='clip')
    plt.xscale('log',nonposx='clip')
    plt.ylabel(r'$\rm{N(>S)/deg^{2}}$',fontsize=32)
    plt.xlabel(r'$\rm{S_{5-10keV}/erg\cdot s^{-1}\cdot cm^{-2}}$',fontsize=32)
    plt.show()
    
    
    
    






