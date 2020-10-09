import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.integrate import quad, simps
import matplotlib.pyplot as plt
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
      
    
    def get_PDF(distribution_, min_, max_p, bins=100):
        """ create PDF from the sampled distribution"""
        xpdf = np.linspace(min_p, max_p, bins)
        smoothed = stats.gaussian_kde(distribution_, bw_method=0.15)
        integral = quad(smoothed, min_p, max_p)[0]
        normed_ypdf = smoothed(xpdf)/integral
        PDF_ = interp1d(xpdf, normed_ypdf, bounds_error=False, fill_value=1e-99)
        
        return xpdf, PDF_
    
    my_prior = [[41.0, 46.0],\
                [-2.0, 5.0],\
                [-2.0, 5.0],\
                [0.0, 10.0],\
                [-10.0, 3.0],\
                [0.01, 4.0],\
                [41.0, 46.0],\
                [0.0, 1.0],\
                [-10.0, -2.0]]
    print my_prior[0][0]     
    def uniform_prior(i):
        min_p, max_p = my_prior[i][0], my_prior[i][1]
        return 1.0/(max_p - min_p)
    #in_run = '/home/Sotiria/Dropbox/transfer/XLF_output_files/individual_fields/Chandra_CDFS/1-.txt'
    in_run = '/home/Sotiria/Dropbox/transfer/XLF_output_files/combination_fields/All_XMM/1-.txt'
    
    in_dist = np.loadtxt(in_run)
    
    logL0 = in_dist[:,2]
    bins = 100
    for i in range(2, len(in_dist[0,:])):

        min_p, max_p = my_prior[i-2][0], my_prior[i-2][1]
        prior_PDF = np.array( bins * [uniform_prior(i-2)] )
        
               
        draws = in_dist[:,i]
        x_PDF, posterior_PDF = get_PDF(draws, min_p, max_p) # generator
        prior_PDF = np.array( len(x_PDF) * [uniform_prior(i-2)] )
        
        #print min( posterior_PDF(draws) ), max( posterior_PDF(draws) )
        #print min( prior_PDF ), max( prior_PDF )
        #print min( posterior_PDF(draws)/prior_PDF ), max( posterior_PDF(draws)/prior_PDF )
        
        integrand = posterior_PDF(x_PDF) * np.log2( posterior_PDF(x_PDF)/prior_PDF )
        DKL = interp1d(x_PDF, integrand, bounds_error=False, fill_value=1e-99)
        
        info_gain = simps(DKL(x_PDF), x_PDF)
     
        #plt.plot(x_PDF, integrand, '.')
        #plt.axhline(0)
        #plt.show()
        
        print round(info_gain,2)
        
        #plt.plot(sorted(draws), posterior_PDF(sorted(draws)), 'b-')
        #plt.plot(x_PDF, prior_PDF, 'r-')
        #plt.title('information gain: '+str(round(info_gain,2))+' bits')
        #plt.show()