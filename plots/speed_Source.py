import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/no_uncertainties_included/MLE/MLE_modules/')

from numpy import arange,pi,log10,log,sqrt,vectorize,zeros,linspace,asarray
from parameters import Parameters
from cosmology import Distance
from speed_test_Survey import Survey
import pyfits
import scipy.integrate

class Source:
    """usage: Source('data') from file='input_data.txt', provides the observed data points
       caution: Lx is logarithm, Fx is linear        """
    def __init__(self,flavor):
        self.infile='/home/sotiria/workspace/Luminosity_Function/input_files/Input_Data.fits'
#       Setup parameters
        params = Parameters()
        self.fields = Parameters.fields(params)
        self.c_light = Parameters.speed_light(params)
        self.zmin, self.zmax = Parameters.z(params)
        self.pl = Parameters.pl(params)
#       cosmological distances 
        self.cosmology = Distance()
        self.s = Survey(plot_curves=False)
        self.mode = flavor
        return
          
    def get_data(self,zmin=0,zmax=100):
        fin = pyfits.open(self.infile)
        ID = []
        F = []
        F_err = []
        Z = []
        Z_flag = []
        Field = []
        fdata = fin[1].data
        for element in self.fields:
#            Select data
            fdata = fdata[fdata.field('redshift_flag') > 0 ]
            mask_field = fdata[fdata.field('field') == element ]
            mask_zmin = mask_field[mask_field.field('redshift')>zmin]
            mask_zmax = mask_zmin[mask_zmin.field('redshift')<zmax]
#            Read data
            ID.extend(mask_zmax.field('ID'))
            F.extend(mask_zmax.field('Flux'))
            F_err.extend(mask_zmax.field('e_Flux'))
            Z.extend(mask_zmax.field('redshift'))
            Z_flag.extend(mask_zmax.field('redshift_flag'))
            Field.extend(mask_zmax.field('field'))
        return ID, F, F_err, Z, Z_flag, Field

    def get_luminosity(self, Fx ,Fx_err, z, z_err=0):
        Lx = []
        Lx_err = []
        if z_err == 0:
            z_err = zeros(len(z))
        for i in arange(0,len(z)):
            dL = self.cosmology.luminosity_dist(z[i])
            Ddl = self.cosmology.derivative_lum_dist(z[i])
            Lum = Fx[i]*4.0*pi*(dL**2.0)*(1.0+z[i])**(self.pl-1.0)
            logL = log10(Lum)
            #if logL>42.0:
            Lx.append(logL)
            delta_L = sqrt( (4.0 * pi * dL * (1.0 + z[i]) * Fx[i] * ( 2.0 * Ddl * (1.0 + z[i])**(self.pl - 2.0) + dL * (self.pl - 1.0)) * z_err[i])**2.0
                            + (( 4.0 * pi * dL**2.0 * (1.0 + z[i])**(self.pl - 1.0) )*Fx_err[i])**2.0 )
            Lx_err.append( delta_L / ( Lum * log(10.0) ) )
        return Lx, Lx_err
    
    def get_flux(self,Lx,z):
        dL = self.cosmology.luminosity_dist(z)
        return ((10.**Lx)/(4.*pi*(dL**2.)*(1.+z)**(self.pl-1.)))
    
    def get_V(self, Lx, zmin, zmax, zstep): 
        s = Survey(plot_curves=False)    
        redshifts = linspace(zmin, zmax, (zmax-zmin)/zstep)
        dv = []   
        z = []
        for red in redshifts:
            a = s.return_area( self.get_flux(Lx, red) )
            V = self.cosmology.dif_comoving_Vol(red, a)*3.4036771e-74
            if V<1.0e-60: break
            dv.append( V ) # vol in Mpc^3
            z.append(red)
        return sum(dv)*zstep

    def get_Vint(self, Lx, zmin, zmax, zstep): 
        s = Survey(plot_curves=False)    
        redshifts = linspace(zmin, zmax, (zmax-zmin)/zstep)
        dv = []   
        z = []
        for red in redshifts:
            a = s.return_area( self.get_flux(Lx, red) )
            V = self.cosmology.dif_comoving_Vol(red, a)*3.4036771e-74
            if V<1.0e-60: break
            dv.append( V ) # vol in Mpc^3
            z.append(red)
        integral = scipy.integrate.simps(dv, z)
        return integral

    def dV_dz(self,l,z):
        s = Survey(plot_curves=False) 
        flux = self.get_flux(l, z)
        area = s.return_area(flux)
        return self.cosmology.dif_comoving_Vol(z, area)*3.4036771e-74 # vol in Mpc^3
    

    def return_data(self):
        Lz = []
        for i in arange(0,len(self.z)):
            Lz.append([self.Lx[i], self.z[i], self.Lx_err[i], self.z_err[i]])
        return self.Lx, self.z, self.Lx_err, self.z_err
    
    def Dz_area(self, LLx, zz, flux=False,plot_acurves=False):
#    Data DO NOT include the area curve.

        if self.mode == 'data':      
            Flux = vectorize(self.get_flux)
            temp_Fx = Flux(LLx, zz)
            #area = self.s.return_area(temp_Fx)   
            if flux==False:
                dv = vectorize(self.cosmology.dif_Vc)
                dvc = dv(zz)
                Data = zip(temp_Fx, dvc*3.4036771e-74) # vol in Mpc^3 
                return Data
            if flux==True:
                new_grid = zip(temp_Fx, LLx, zz)  
                return new_grid
#    The grid includes the area curve, gives the survey efficiency
        if self.mode == 'grid':      
            s = Survey(plot_curves=plot_acurves)
            Flux = vectorize(self.get_flux)
            temp_Fx = Flux(LLx, zz)
            area = s.return_area(temp_Fx)   
            if flux==False:
                Data = []
                DV = vectorize(self.cosmology.dif_comoving_Vol)
                DVc = DV(zz,area)
                Data = DVc*3.4036771e-74 # vol in Mpc^3, per unit area
                return Data
            if flux==True:
                new_grid = zip(temp_Fx, LLx, zz, area)              
                return new_grid
