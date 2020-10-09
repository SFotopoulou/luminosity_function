import sys
import time
import math
import numpy as np
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
from AGN_LF_config import LF_config
from Source import *
from Survey import *
from scipy.integrate import simps,dblquad
from make_PDFz import Spectrum
import matplotlib.pyplot as plt

class AGN:
    def __init__(self, name, counts, e_counts, f, ef, mag, redshift, flag, field):
        #print "I am AGN class"
        #print name, redshift, flag, field
        self.ID = name
        self.field = field
        self.counts = counts
        self.e_counts = e_counts
        self.flux = f
        self.e_flux = ef
        self.mag = mag
        self.z = redshift
        self.zflag = flag    
        self.l = get_luminosity(self.flux, self.e_flux, self.z, power_law=LF_config.pl)
        #self.e_l = get_luminosity([self.flux], [self.e_flux], [self.z], power_law=LF_config.pl)[1][0]
        self.s = Spectrum()
#        self.vPDFgauss = vectorize(self.PDFgauss)
        if self.field == 'MAXI':
            self.e_zspec = 0.0001
        else:
            self.e_zspec = 0.001
        self.zsteps = 15
        self.Lpoints = 10
        pass
    
    def PDFgauss(self, x, mu, sigma):
        s2 = sigma*sigma
        t2 = (x-mu)*(x-mu)
        result = (np.exp(-t2 / (2.0 * s2))) / np.sqrt(2.0 * math.pi * s2)
        return result

    def PDFz(self):
#    Redhisft Probability Distribution Function
#    PDFz from photoz, or gaussian for zspec
        #print self.zflag
        if self.zflag == 2 and (self.field in LF_config.photozFields):
            #print self.field, self.ID
            self.xPDF_z, self.yPDF_z = self.s.return_PDF(self.field, self.ID) 
            
            if self.xPDF_z[0] == 0.0 :
                self.xPDF_z = self.xPDF_z[1:]
                self.yPDF_z = self.yPDF_z[1:]
            self.zpoints = len(self.xPDF_z)
        elif self.zflag < 0 :
            # flat PDF across redshift
            self.xPDF_z = np.linspace(LF_config.zmin, LF_config.zmax, 200)
            self.yPDF_z = np.array( [ 1./(LF_config.zmax - LF_config.zmin) ] * len(self.xPDF_z) )
            self.yPDFz = np.tile(self.yPDF_z, self.Lpoints)
            self.xPDFz = self.xPDF_z            
        else:    
            self.xPDF_z = np.linspace(self.z - 5.0 * self.e_zspec, self.z + 5.0 * self.e_zspec, self.zsteps)
            self.yPDF_z = self.PDFgauss(self.xPDF_z, self.z, self.e_zspec)
            self.zpoints = self.zsteps 
        self.yPDFz = np.tile(self.yPDF_z, self.Lpoints)
        self.xPDFz = self.xPDF_z
        #plt.plot(self.xPDF_z, self.yPDF_z)
        #plt.title(self.ID)
        #plt.savefig(str(self.ID)+'.jpg')
        #plt.clf()
#        print self.ID, self.xPDF_z[ np.where(self.yPDF_z==np.max(self.yPDF_z))[0][0] ]
#        print
    
    def make_grid(self):
#    Make individual grid for each object
        zmin = min(self.xPDFz)
        zmax = max(self.xPDFz)
        Lmin, Lmin_err = get_luminosity([self.flux], [self.e_flux], [zmin])
        Lmax, Lmax_err = get_luminosity([self.flux], [self.e_flux], [zmax])
        LL = np.array([np.ones( len(self.xPDFz), float )*item for item in np.linspace(Lmin[0] - 5.*Lmin_err[0], Lmax[0] + 5.*Lmax_err[0], self.Lpoints)])
        self.L = LL.ravel() #    make LL 1D
        self.Z = np.tile(self.xPDFz, self.Lpoints) # repeat as many times as Lpoints
#
        data_grid = Dz_DV(self.L, self.Z)
        self.Fx_grid, DVc_grid = zip(*data_grid)
        self.DVc_grid = np.array(DVc_grid)
        self.dz = self.xPDFz[1]-self.xPDFz[0]
        Ltemp = self.L[::self.Lpoints]
        self.dL = Ltemp[1] - Ltemp[0]
        pass

    def make_data_vol(self):
        dv = np.vectorize(dif_Vc)
        self.dvc = dv(self.xPDF_z)
        
    def make_lumis(self):
        #print self.flux, self.e_flux, self.xPDF_z
        self.L_PDF = np.array( [get_luminosity(self.flux, self.e_flux, zphot)[0] for zphot in self.xPDF_z] )
#        plt.plot(self.xPDF_z, self.L_PDF)
#        plt.xlim([0,7])
#        plt.ylim([40,47])
#        plt.show()
        
    def PDFf(self):
#    Flux Probability Distribution Function, for the moment gaussian
        self.xPDFf = np.array( self.Fx_grid )
        self.yPDFf = self.PDFgauss(self.xPDFf, self.flux, self.e_flux)
       
    def return_PDFz(self):
        return self.xPDF_z, self.yPDF_z
    
    def return_PDFf(self):
        return self.xPDFf, self.yPDFf    
    
    def return_grid(self):
        return self.L, self.Z
    
    def return_MultipliedGrid_zL(self):
        return self.DVc_grid*self.yPDFz*self.yPDFf

    def return_MultipliedGrid(self):
        return self.dvc*self.yPDF_z
    
    def return_points(self):
        return self.Lpoints, self.zpoints

    def return_lumis(self):
        #print self.xPDF_z
        return self.L_PDF, np.array( self.xPDF_z )
    
    def return_data(self):
        return self.ID, self.field, self.counts, self.e_counts, self.flux, self.e_flux, self.mag, self.z, self.zflag

if __name__ == "__main__":
    agn = AGN(206575, 5.82e-15, 1.06e-15, 1.745, 2, 'LH')
    print agn.return_PDFz()
    