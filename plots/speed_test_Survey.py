import sys
# Add the module path to the sys.path list
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/no_uncertainties_included/MLE/MLE_modules/')
#
from numpy import arange,vectorize, array
import numpy as np
import matplotlib.pyplot as plt
from parameters import Parameters
import pyfits

class Survey:
    
    def __init__(self,plot_curves=False):
        params = Parameters()
        self.fields = Parameters.fields(params)
        hdulist = pyfits.open("/home/sotiria/workspace/Luminosity_Function/input_files/area_curve.fits")
        acurve = hdulist[1].data
        self.surveys = []
#    5th curve
        if 5 in self.fields:
            MAXI = acurve[ acurve.field('MAXI_FLUX')>0 ]
            self.fx_5 = array( MAXI.field('MAXI_FLUX') )
            self.area_5 = array( MAXI.field('MAXI_AREA') )
            if plot_curves==True: 
                plt.plot(self.fx_5,self.area_5,label='MAXI',color='gray',ls='-')
            self.surveys.append( (self.fx_5, self.area_5) )
#    1st curve
        if 1 in self.fields:
            HBSS = acurve[ acurve.field('HBSS_FLUX')>0 ]
            self.fx_1 = array( HBSS.field('HBSS_FLUX') )
            self.area_1 = array( HBSS.field('HBSS_AREA') )
            if plot_curves==True: 
                plt.loglog(self.fx_1,self.area_1,label='HBSS',color='k',ls='-') 
            self.surveys.append( (self.fx_1, self.area_1) )    
#    2nd curve
        if 2 in self.fields:
            COSMOS = acurve[ acurve.field('COSMOS_FLUX')>0 ]
            self.fx_2 = array( COSMOS.field('COSMOS_FLUX') )
            self.area_2 = array( COSMOS.field('COSMOS_AREA') )
            if plot_curves==True: 
                plt.plot(self.fx_2,self.area_2,label='COSMOS',color='k',ls='--')
            self.surveys.append( (self.fx_2, self.area_2) )
#    3rd curve
        if 3 in self.fields:
            LH = acurve[ acurve.field('LH_FLUX')>0 ]
            self.fx_3 = array( LH.field('LH_FLUX') )
            self.area_3 = array( LH.field('LH_AREA') )
            if plot_curves==True: 
                plt.plot(self.fx_3,self.area_3,label='LH',color='k',ls='-.')
            self.surveys.append( (self.fx_3, self.area_3) )    
#    4th curve
        if 4 in self.fields:
            CDFS = acurve[ acurve.field('CDFS_FLUX')>0 ]
            self.fx_4 = array( CDFS.field('CDFS_FLUX') )
            self.area_4 = array( CDFS.field('CDFS_AREA') )        
            if plot_curves==True: 
                plt.plot(self.fx_4,self.area_4,label='CDFS',color='k',ls=':')
            self.surveys.append( (self.fx_4, self.area_4) )
        if plot_curves==True: 
            plt.title("Area curves")
            plt.xlabel("Fx (erg/s/cm$^2$)")
            plt.ylabel("Area (deg$^2$)")     
            plt.legend(loc=0)
            plt.xscale('log')
            plt.yscale('log')
            plt.draw()
            plt.savefig('acurve.jpg',dpi=300)
            plt.savefig('acurve.eps')
            #plt.clf()
            plt.show()            
        pass
    
    def find_nearest(self, flux, area, value):
        idx=(np.abs(flux-value)).argmin()
        return area[idx]
    
    def return_area(self,flux_in):
        if type(flux_in) is float or np.isscalar(flux_in) :
            area_out = 0.
            for flux, area in self.surveys:
                nearest = self.find_nearest(flux, area, flux_in) 
                area_out = area_out + nearest     
            return area_out*0.00030461742 # in steradians     
        else:
            f = array(flux_in)
            a = []
            for value in f:
                area_out = 0.
                for flux, area in self.surveys:
                    nearest = self.find_nearest(flux, area, value) 
                    area_out = area_out +  nearest     
                a.append(area_out*0.00030461742) # in steradians     
            return a
