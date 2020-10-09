import sys
# Add the module path to the sys.path list
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/no_uncertainties_included/MLE/MLE_modules/')
#
import time, itertools
from numpy import hstack,vstack,arange,savetxt, linspace,logspace,sqrt,log10,array,ones,tile, median, asarray, genfromtxt, power, column_stack
from Source import Source
import matplotlib.pyplot as plt
import scipy.integrate
from LFunctions import Models
from scipy.integrate import simps
from parameters import Parameters

params = Parameters()
Lmin, Lmax = Parameters.L(params)
zmin, zmax = Parameters.z(params)
model = Models()

def modelF_Phi(Lx,z,L0,g1,g2,p1,p2,zc,La,a):    
    """ 
    The luminosity function model 
    """
    return model.Fotopoulou(Lx,z,L0,g1,g2,p1,p2,zc,La,a)

draws = genfromtxt('/home/sotiria/workspace/Luminosity_Function/src/including_uncertainties/Bayesian/APEMoST/LDDE_Fotop/analysis_results/Fotopoulou.prob')

MCMC = genfromtxt('/home/sotiria/workspace/Luminosity_Function/src/including_uncertainties/Bayesian/APEMoST/LDDE_Fotop/analysis_results/Fotopoulou_MCMC_params')
L0_F = MCMC[0,1]
g1_F = MCMC[1,1]
g2_F = MCMC[2,1]
p1_F = MCMC[3,1]
p2_F = MCMC[4,1]
zc_F = MCMC[5,1]
La_F = MCMC[6,1]
a_F = MCMC[7,1]
norm_F = power(10.0, MCMC[8,1])
# Set up grid for data
## L sample
#Lpoints = 50 # no need to go too thin
## z sample
#zpoints = 400 # governed by the photoz step
#
#LL = array([ones( (zpoints), float )*item for item in linspace(Lmin,Lmax,Lpoints)])
#L = LL.ravel() #    make LL 1D
#Z = tile(linspace(zmin, zmax, zpoints), Lpoints) # repeat as many times as Lpoints
## Set up grid for survey integral
#g = Source('grid')
#int_grid = g.Dz_area(L,Z)
#DVcA = int_grid
#redshift_int = Z[0:zpoints]
#Luminosity_int = linspace(Lmin,Lmax,Lpoints)

# Observations
s = Source('data')
ID, F, F_err, Z_i, Z_flag, Field = s.get_data(zmin, zmax)
Lx_i, Lx_err = s.get_luminosity(F, F_err, Z_i)

#    Sort according to luminosity

Lx_s, z_s, F_s = zip(*sorted(zip(Lx_i, Z_i, F))) 
   

zbin = [0.01, 0.2, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 3.0, 4.0,7.0]
#Lbin = [41.25, 42.0, 42.5, 43.0, 43.25, 43.50, 44.0, 44.65]               
Lbin = [42.0, 43.0, 44.0, 45.0, 46.0]
linecolor = (0., 0., 0.)
fillcolor = (0.75, 0.75, 0.75)
pointcolor = (0., 0., 0.)
linestyles = itertools.cycle( ['-',':','-.','steps'] )
colors = itertools.cycle( ['k', 'k','k','gray'] )
markers = itertools.cycle( ['o', 's', '^' , 'v'] )
markerface = itertools.cycle( ['k', 'w', 'k','gray'] )
ecolors = itertools.cycle( ['k', 'k','k','gray'] )

zlabel = []
save_data = []
#ll = linspace(41.0,48.0)
## LF at z=0
#LF0 = []
#for lx in ll:
#    PF = modelF_Phi(lx, 0.0, L0_F, g1_F, g2_F, p1_F, p2_F, zc_F, La_F, a_F)*norm_F
#    LF0.append(log10(PF))
#
#redshift_bin = itertools.cycle([1.040e-01,  3.449e-01, 6.278e-01, 8.455e-01, 1.161e+00, 1.465e+00, 1.799e+00, 2.421e+00, 3.376e+00])
fig_size = [10, 10]
fig = plt.figure(figsize=fig_size)
fig.subplots_adjust(left=0.175,  right=0.95, bottom=0.10, top=0.95, wspace=0.0, hspace=0.0)
zspace = linspace(0,4.5,20)

for i in xrange(0, len(Lbin)-1):
    name = '/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/Ndensity_files/Ndensity_'+str(Lbin[i])+'.dat'    
    values = hstack(( zspace ))
    print Lbin[i]
    ll = linspace(Lbin[i], Lbin[i+1], 6)
    min_Nmodel = []
    max_Nmodel = []
    
    for z in zspace:
        Nmodel = []
        for k in arange(0, len(draws[:,0])):
            L0_d, g1_d, g2_d, p1_d, p2_d, zc_d, La_d, a_d, norm_d = draws[k,:]
            LF_model = []
            for lx in ll:
                PF = modelF_Phi(lx, z, L0_d, g1_d, g2_d, p1_d, p2_d, zc_d, La_d, a_d)*power(10.0, norm_d)
                LF_model.append(PF)
            integral = log10( simps(LF_model, ll) )
            Nmodel.append(integral)
        min_Nmodel.append( min( asarray(Nmodel) ) )
        max_Nmodel.append( max( asarray(Nmodel) ) )    
    
    values = column_stack((values, min_Nmodel , max_Nmodel))
    savetxt(name, values)
    plt.fill_between(zspace, min_Nmodel, max_Nmodel, color=fillcolor, label=str(Lbin[i]), alpha = 0.2)
    
    mcmc_Nmodel = []
    for z in zspace:
        LF_model = []
        for lx in ll:
            PF = modelF_Phi(lx, z, L0_F, g1_F, g2_F, p1_F, p2_F, zc_F, La_F, a_F)*norm_F
            LF_model.append(PF)
        integral = log10( simps(LF_model, ll) )
        mcmc_Nmodel.append(integral)
    plt.plot(zspace, mcmc_Nmodel, color=colors.next(), ls=linestyles.next(), lw=3, label=str(Lbin[i])+'$<logL_x<$'+str(Lbin[i+1]) )       

    data = []
    for j in arange(0,len(zbin)-1):
        count = 0
        dN_dV = 0.0
        err_dN_dV = 0.0
        Ll = []
        Zz = []
        for Lx, z, f in zip(Lx_s, z_s, F_s):
            if zbin[j] <= z < zbin[j+1] and Lbin[i] <= Lx < Lbin[i+1]:
                count = count + 1
                Vmax =  s.get_V(Lx, zbin[j], zbin[j+1], 0.01) 
                dN_dV = dN_dV + 1.0/Vmax
                err_dN_dV = err_dN_dV + power( (1.0/Vmax), 2.0)
                
                Ll.append(Lx)        
                Zz.append(z)
        err_dN_dV = sqrt( err_dN_dV )
        if count>0:
            print count
            #print median(Zz), median(Ll), count, log10(dN_dV), 0.434*err_dN_dV/dN_dV, median(Ll)-Lbin[i], Lbin[i+1]-median(Ll)
            datum = [median(Zz), median(Ll), count, log10(dN_dV), 0.434*err_dN_dV/dN_dV, median(Ll)-Lbin[i], Lbin[i+1]-median(Ll), median(Zz)-zbin[j], zbin[j+1]-median(Zz)]
            data.append(datum)
    data = array(data)        
    
    redshift = data[:,0]
    e_red_l = data[:,7]
    e_red_h = data[:,8]
    
    Number_density = data[:,3]
    err_Number_density = data[:,4]
    face = markerface.next() 
    plt.errorbar(redshift, Number_density, yerr = err_Number_density, xerr=[e_red_l, e_red_h], ls= ' ', marker=markers.next(), markersize=14, markerfacecolor=face, ecolor=ecolors.next())
#plt.yscale('log')
plt.legend()
plt.xlim([0,4.5])
plt.ylim([-10,-2.5])
plt.xlabel('$Redshift$', fontsize='x-large')
plt.ylabel('$Number\,Density\,(Mpc^{-3}$)', fontsize='x-large')
#plt.draw()
#for extension in ['.eps', '.jpg', '.pdf', '.png']:
#    plt.savefig('/home/sotiria/workspace/Luminosity_Function/src/LF_Plots/Ndensity_files/Ndensity'+extension)
plt.show()