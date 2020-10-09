import sys
# Add the module path to the sys.path list
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/no_uncertainties_included/MLE/MLE_modules')
#
import time
import math   
from numpy import array,log,sqrt,linspace,vectorize,ones,tile,where,sum, genfromtxt, power, log10,savetxt
from Source import Source
from LFunctions import Models
from scipy.integrate import simps,dblquad
import minuit
from class_source import AGN
from make_PDFz import Spectrum
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt

model = Models()
params = Params()

def Phi(Lx,z,params):    
    """ 
    The luminosity function model 
    """
    return model.Fotopoulou(Lx,z,L0,g1,g2,p1,p2,zc,La,a)*power(10.0, Normal)

def Source_sublist(list_in,params,queue):
    """Source detection probability phi*dVc*Source_prob"""
    result = []
    for source in list_in:

        zPDF_y = source.return_PDFz()
        fPDF_z, fPDF_y = source.return_PDFf()
        Luminosity, Redshift, DVc_grid, dz, dL = source.return_grid()
        PHI_Lz = Phi(Luminosity, Redshift, params)
        Lpoint, zpoint = source.return_points()
        
        x = DVc_grid*zPDF_y*fPDF_y*PHI_Lz*dz
        
        y = []
        for count in range(0, Lpoint):
            startz = count*zpoint
            endz = startz + zpoint            
        
            integrand = x[startz:endz]
            redshift_i = Redshift[startz:endz]    
            #print len(integrand), len(redshift_i)       
            int1 = simps(integrand,redshift_i,even='last')
            y.append(int1)
        
        luminosity_i = Luminosity[::zpoint] 

        int2 = simps(y,luminosity_i,even='last')
        result.append(int2)
    
    result = array(result)
    result_pos = where(result>0, result, 1.0)
    result_log = log(result_pos)
    res = sum(result_log) # sum for each sublist

    queue.put(res)    

def Marshall_Likelihood(L0, g1, g2, p1, p2, zc, La, a, Norm):
    params = Params()
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p1
    params.zc = zc
    params.La = La
    params.a = a
    params.Norm = Norm

    """
    Likelihood for MLE L = -2.0*ln(L)
    """
    """ 
    Data probability
    """
    for queue_in, list_in in zip(queue_list, list_list):
        Process(target=Source_sublist, args=(list_in,params,queue_in)).start()

    result = []
    for q in queue_list:
        result.append( q.get() )
        
    source_d = sum(result) # sum of all sublists

    """
       Survey detection probability Phi*dV*Area_curve
    """
    PHI_Lz = Phi(L,Z,params)
    V = DVcA 
    X = V*PHI_Lz # integrand
    
    y = []
    for count in range(0, Lpoints):
        startz = count*zpoints
        endz = startz + zpoints
        x = X[startz:endz]
        
        int1 = simps(x,Redshift_int, even='last')
        y.append(int1)
   
    survey_d = simps(y,Luminosity_int, even='last')
    """ Marshall Likelihood including errors"""
    
    Like = - source_d + survey_d # Difference, no area curve in the data prob.
    return 2.0*Like

start_time = time.time()
############### Observations ###############
# Prepare individual grid from each datum ##
############################################
Lpoints = int((Lmax-Lmin)/0.01)
zpoints = int((zmax-zmin)/0.01)

try:
    L, Z, DVcA  = genfromtxt('input_integral_MAXI.dat', unpack=True)
except IOError as e:
    print 'Oh dear. Generating data file 1, be patient...'   
    LL = array([ones( (zpoints), float )*item for item in linspace(Lmin,Lmax,Lpoints)])
    L = LL.ravel() #    make LL 1D
    Z = tile(linspace(zmin, zmax, zpoints), Lpoints) # repeat as many times as Lpoints
    # Set up grid for survey integral
    g = Source('grid')
    DVcA = g.Dz_area(L,Z)
    integr = zip(L, Z, DVcA)
    savetxt('input_integral_MAXI.dat', integr)    # save to file


Redshift_int = Z[0:zpoints]
Luminosity_int = linspace(Lmin,Lmax,Lpoints)

#############################################################################################3
#    Prepare data
print "Preparing data"
d = Source('data')
ID, Fx, e_Fx, z, z_flag, field = d.get_data()
n = len(ID)
source_list = []
t1 = time.time()
for i in range(0,n):
    source = AGN( ID[i], Fx[i], e_Fx[i], z[i], z_flag[i], field[i] )
    source.PDFz()
    source.make_grid()
    source.PDFf()
    
    source_list.append(source)

print "Source list creation:", time.time()-t1,"sec"
print "-------------------------------------------"
###########################################################################################
# Set up for multi processing
q1 = Queue()
q2 = Queue()

list1 = source_list[:280]
list2 = source_list[280:]

queue_list = [q1, q2]
list_list = [list1, list2]
##########################################################################################
m = minuit.Minuit(Marshall_Likelihood)
#    Accuracy
m.printMode = 1
m.strategy = 1
m.up = 1.0
m.tol = 10

#    Careful! Do not use integers!
#    Initial Values
m.values["L0"] = LF_config.L0
m.values["g1"] = LF_config.g1
m.values["g2"] = LF_config.g2
m.values["p1"] = LF_config.p1
m.values["p2"] = LF_config.p2
m.values["zc"] = LF_config.zc
m.values["La"] = LF_config.La
m.values["a"] = LF_config.a
m.values["Norm"] = LF_config.Norm

print "------------------------------"
print "Running Last MLE"

m.migrad()

params.Norm = m.values["Norm"]
params.L0=m.values["L0"]
params.g1=m.values["g1"]
params.g2=m.values["g2"]
params.p1=m.values["p1"]
params.p2=m.values["p2"]
params.zc=m.values["zc"]
params.La=m.values["La"]
params.a=m.values["a"]

print "MLE converged"
print "Time lapsed =",round(time.time()-start_time,2),"sec"
print "------------------------------"

print "Number of function calls:",m.ncalls
print "Vertical distance to minimum", m.edm
print "Ndata=",n
Nparameters = 9.0
Ndata = n
output = open('Fotopoulou_MAXI.out','w')
output.write("No. model parameters = "+str(Nparameters)+"\n")
output.write( "Least likelihood value = "+str(m.fval)+"\n")
output.write( "with parameters : "+str(m.values)+"\n")
output.write( "Minimum distance between function calls : "+str(m.edm)+"\n")
output.write( "Vertical distance to minimum"+str(m.edm)+"\n")
output.write( "AIC="+str(2.*Nparameters+m.fval)+"\n")
output.write( "AICc="+str(2.*Nparameters+m.fval+(2.*Nparameters*(Nparameters+1.))/(Ndata-Nparameters-1.))+"\n")
output.write( "BIC="+str(m.fval+Nparameters*log10(Ndata))+"\n")
#    Calculate errors and covariance matrix
print "calilng Hesse"
m.hesse()
output.write( "\nLinear Errors\n" )
output.write( str(m.errors) )
output.write( "Covariance Matrix\n" )
output.write( str(m.covariance) )
## calculate non linear errors
print "calling MINOS"
m.minos()
output.write( "\nMINOS Errors\n" )
output.write( str(m.merrors) )
output.close()

redshift = [0.26, 0.73, 1.44, 2.42, 3.37]
Luminosity = linspace(42.0, 45.5)
import itertools
colors = itertools.cycle(['blue', 'green', 'red', 'cyan', 'black'])

for item, color in zip(redshift, colors):
    dPhi = log10( Phi(Luminosity, item, params) )
    plt.plot(Luminosity, dPhi, color = color)

plt.ylim([-8,-3])
plt.xlim([41.5, 45.5])
#plt.yscale("log")
plt.title("Luminosity Dependent Density Evolution")
plt.draw()
plt.savefig("Fotop_LDDE_MAXI.eps")
plt.savefig("Fotop_LDDE_MAXI.pdf")
plt.savefig("Fotop_LDDE_MAXI.jpg")
plt.show()
