import sys
import time
import math
from numpy import array,log,sqrt,linspace,vectorize,ones,tile,exp,sqrt,where, power
from speed_Source import Source
from parameters import Parameters
from LFunctions import Models
from scipy.integrate import simps,dblquad
from make_PDFz import Spectrum
import matplotlib.pyplot as plt

class AGN:
    def __init__(self, name, f, ef, redshift, flag, field):
        self.ID = name
        self.flux = f
        self.e_flux = ef
        self.z = redshift
        self.zflag = flag
        self.field = field
#    
        self.s = Spectrum()
        self.model = Models()
#
#        self.vPDFgauss = vectorize(self.PDFgauss)
        self.e_zspec = 0.01
        self.zsteps = 15
        self.Lpoints = 10
        pass
    
    def PDFgauss(self, x, mu, sigma):
        s2 = sigma*sigma
        t2 = (x-mu)*(x-mu)
        return (exp(-t2/(2.0*s2)))/sqrt(2.0*math.pi*s2)

    def PDFz(self):
#    Redhisft Probability Distribution Function
#    PDFz from photoz, or gaussian for zspec
        if self.zflag == 2 and (self.field == 2 or self.field == 3):
            self.xPDF_z, self.yPDF_z = self.s.return_PDF(self.field, self.ID)  
            if self.xPDF_z[0] == 0.0:
                self.xPDF_z = self.xPDF_z[1:]
                self.yPDF_z = self.yPDF_z[1:]
            
            self.zpoints = len(self.xPDF_z)
        else:    
            self.xPDF_z = linspace(self.z-5.0*self.e_zspec, self.z+5.0*self.e_zspec, self.zsteps)
            self.yPDF_z = self.PDFgauss(self.xPDF_z, self.z, self.e_zspec)
            self.zpoints = self.zsteps 
        self.yPDFz = tile(self.yPDF_z, self.Lpoints)
        self.xPDFz = self.xPDF_z
        pass
    
    def make_grid(self):
#    Make individual grid for each object
        dat = Source('data')
        zmin = min(self.xPDFz)
        zmax = max(self.xPDFz)
        Lmin, Lmin_err = dat.get_luminosity([self.flux], [self.e_flux], [zmin])
        Lmax, Lmax_err = dat.get_luminosity([self.flux], [self.e_flux], [zmax])
        LL = array([ones( len(self.xPDFz), float )*item for item in linspace(Lmin[0]-Lmin_err[0], Lmax[0]+Lmax_err[0], self.Lpoints)])
        self.L = LL.ravel() #    make LL 1D
        self.Z = tile(self.xPDFz, self.Lpoints) # repeat as many times as Lpoints
#
        data_grid = dat.Dz_area(self.L,self.Z)
        self.Fx_grid, DVc_grid = zip(*data_grid)
        self.DVc_grid = array(DVc_grid)
        self.dz = self.xPDFz[1]-self.xPDFz[0]
        Ltemp = self.L[::self.Lpoints]
        self.dL = Ltemp[1]-Ltemp[0]
        pass
    
    def PDFf(self):
#    Flux Probability Distribution Function, for the moment gaussian
        self.xPDFf = array(self.Fx_grid)
        self.yPDFf = self.PDFgauss(self.xPDFf, self.flux, self.e_flux)
        pass 
  
    def return_PDFz(self):
        return self.yPDFz
    
    def return_PDFf(self):
        return self.xPDFf, self.yPDFf    
    
    def return_grid(self):
        return self.L, self.Z#, self.DVc_grid, self.dz, self.dL

    def return_MultipliedGrid(self):
        return self.DVc_grid*self.dz*self.yPDFf*self.yPDFz
    
    def return_points(self):
        return self.Lpoints, self.zpoints
