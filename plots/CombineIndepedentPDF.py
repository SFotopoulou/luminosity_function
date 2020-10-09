import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.integrate import quad, simps
import matplotlib.pyplot as plt
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    def get_myCDF(array_CDF, xlist, min_, max_, min_fill, max_fill):
        min_CDF = min_
        max_CDF = max_
    
        CDF_low = np.where(xlist<min_CDF, min_fill*np.ones(len(array_CDF)), array_CDF)
        CDF_high = np.where(xlist>max_CDF, max_fill*np.ones(len(array_CDF)), CDF_low)
    
        return CDF_high

    def draw_distribution(incurve, inrange):
        
        dx = inrange[1] - inrange[0]
        cumulative = np.cumsum(incurve)*dx
    
        inverse = interp1d(cumulative, inrange, bounds_error=False, fill_value=-999.0)
        random = np.random.random(100000)
    
        new_dist = get_myCDF(inverse(random), random, np.min(cumulative), np.max(cumulative), np.min(inrange), np.max(inrange))
        #plt.clf()
        #plt.plot(inrange, incurve)
        #plt.hist(new_dist, bins=100, normed=True)
        #plt.draw()
        #plt.show()
        #plt.plot(random, inverse(random))
        #plt.show()
    
        return new_dist

    def credible_interval(distribution, level=68, bins_=10):
        # find best binning
        std = np.std(distribution)
        d_range = np.max(distribution) - np.min(distribution)
        bins_ = (20*d_range/std)

        pdf, bins = np.histogram(distribution, bins = bins_, normed=True)
        # credible interval
        bins = bins[:-1]
        binwidth = bins[1]-bins[0]

        idxs = pdf.argsort()
        idxs = idxs[::-1]
        credible_interval_ = idxs[(np.cumsum(pdf[idxs])*binwidth < level/100.).nonzero()]
        idxs = idxs[::-1] # reverse

        low = min( sorted(credible_interval_) )
        high = max( sorted(credible_interval_) )
        
        min_val = bins[low]
        max_val = bins[high]
        # mode
        dist_bin = np.array([bins[i]+(bins[1]-bins[0])/2. for i in range(0,len(bins))])
        mode = dist_bin[np.where(pdf==max(pdf))][0]
        #print "mode:", min_val, max_val, mode
        
        return min_val, max_val, mode    

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
    
    def uniform_prior(i):
        min_p, max_p = my_prior[i][0], my_prior[i][1]
        return 1.0/(max_p - min_p)
                
    in_runs = ['/home/Sotiria/Dropbox/transfer/XLF_output_files/individual_fields/MAXI/1-.txt',
               '/home/Sotiria/Dropbox/transfer/XLF_output_files/individual_fields/XMM_HBSS/1-.txt',
               '/home/Sotiria/Dropbox/transfer/XLF_output_files/individual_fields/XMM_COSMOS/1-.txt',
               '/home/Sotiria/Dropbox/transfer/XLF_output_files/individual_fields/XMM_LH/1-.txt',
               '/home/Sotiria/Dropbox/transfer/XLF_output_files/individual_fields/XMM_CDFS/1-.txt',
               '/home/Sotiria/Dropbox/transfer/XLF_output_files/individual_fields/Chandra_COSMOS/1-.txt',
               '/home/Sotiria/Dropbox/transfer/XLF_output_files/individual_fields/Chandra_AEGIS/1-.txt',
               '/home/Sotiria/Dropbox/transfer/XLF_output_files/individual_fields/Chandra_CDFS/1-.txt']
    
    in_dist = {}
    for run in in_runs:
        r = in_runs.index(run)
        in_dist[r] = np.loadtxt(run)
        
    bins = 100
    for i in range(2, len(in_dist[0][0,:])):
        
        min_p, max_p = my_prior[i-2][0], my_prior[i-2][1]
        prior_PDF = np.array( bins * [uniform_prior(i-2)] )
        
        combined_posterior = np.ones(bins)
            
        for r in range(0, len(in_runs)):
            draws = in_dist[r][:,i]
            
            x_PDF, posterior_PDF = get_PDF(draws, min_p, max_p) # generator
            combined_posterior = combined_posterior * posterior_PDF(x_PDF)
            
            plt.plot(x_PDF, posterior_PDF(x_PDF))
        
        norm = simps(combined_posterior, x_PDF)
        normalized_posterior = combined_posterior/norm
        
        
        new_posterior = draw_distribution(normalized_posterior, x_PDF)
        post_68min, post_68max, post_mode = credible_interval(new_posterior)
        
        plt.plot(x_PDF, normalized_posterior,'k--')
    
        integrand = normalized_posterior * np.log2( normalized_posterior/prior_PDF )
        DKL = interp1d(x_PDF, integrand, bounds_error=False, fill_value=1e-99)
        
        info_gain = simps(DKL(x_PDF), x_PDF)
        print i, min_p, max_p, round(info_gain,2), post_68min, post_68max, post_mode
        plt.show()