import sys
# Add the module path to the sys.path list
sys.path.append('/home/sotiria/workspace/Luminosity_Function/src/no_uncertainties_included/MLE/MLE_modules')
#
import time
import math   
from numpy import array,log,sqrt,linspace,vectorize,ones,tile,where,sum, genfromtxt, power, log10,savetxt,mean
from speed_Source import Source
from parameters import Parameters
from LFunctions import Models
from scipy.integrate import simps,dblquad
import minuit
from speed_class_source import AGN
from make_PDFz import Spectrum
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
from test_Survey import Survey
s = Survey(plot_curves=True) 
params = Parameters()
Lmin, Lmax = Parameters.L(params)
zmin, zmax = Parameters.z(params)

model = Models()

def Phi(Lx,z,L0,g1,g2,p1,p2,zc,La,a,Normal):    
    """ 
    The luminosity function model 
    """
    return model.Fotopoulou(Lx,z,L0,g1,g2,p1,p2,zc,La,a)*power(10.0, Normal)
    
def Source_sublist(list_in,L0,g1,g2,p1,p2,zc,La,a,Normal,queue):
    """Source detection probability phi*dVc*Source_prob"""
    result = []
    for source in list_in:
        Grid = source.return_MultipliedGrid()
        Luminosity, Redshift = source.return_grid()
        PHI_Lz = Phi(Luminosity, Redshift, L0,g1,g2,p1,p2,zc,La,a,Normal)
        Lpoint, zpoint = source.return_points()
        
        x = Grid*PHI_Lz
        
        y = []
        count_range = xrange(0, Lpoint)
        for count in count_range:
            startz = count*zpoint
            endz = startz + zpoint            
            integrand = x[startz:endz]
            redshift_i = Redshift[startz:endz]    
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
 

def Marshall_Likelihood(L0,g1,g2,p1,p2,zc,La,a,Normal):
    """
    Likelihood for MLE L = -2.0*ln(L)
    """
    """ 
    Data probability
    """
    processes = []
    for queue_in, list_in in zip(queue_list, list_list):
        p = Process(target=Source_sublist, args=(list_in,L0,g1,g2,p1,p2,zc,La,a,Normal,queue_in))
        p.start()
        processes.append(p)
        
    result = []
    for q in queue_list:
        result.append( q.get() )
                
    for p in processes:
        p.join()
            
    source_d = sum(result) # sum of all sublists

    """
       Survey detection probability Phi*dV*Area_curve
    """
    PHI_Lz = Phi(L,Z,L0,g1,g2,p1,p2,zc,La,a,Normal)
    X = DVcA*PHI_Lz # integrand
    
    y = []
    count_r = xrange(0, Lpoints)
    for count in count_r:
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
g = Source('grid')
vectFlux = vectorize(g.get_flux)
vectArea = vectorize(s.return_area)
try:
    L, Z, DVcA  = genfromtxt('integral.dat', unpack=True)
except IOError as e:
    print 'Oh dear. Generating data file 1, be patient...'   
    LL = array([ones( (zpoints), float )*item for item in linspace(Lmin,Lmax,Lpoints)])
    L = LL.ravel() #    make LL 1D
    Z = tile(linspace(zmin, zmax, zpoints), Lpoints) # repeat as many times as Lpoints
    # Set up grid for survey integral
    
    DVcA = g.Dz_area(L,Z)
    temp_Fx = vectFlux(L,Z)
    
    area = vectArea(temp_Fx)  
    integr = zip(L, Z, DVcA, temp_Fx, area)
    savetxt('integral.dat', integr)    # save to file


Redshift_int = Z[0:zpoints]
Luminosity_int = linspace(Lmin,Lmax,Lpoints)

#############################################################################################3
#    Prepare data
print "Preparing data"
d = Source('data')
ID, Fx, e_Fx, z, z_flag, field = d.get_data()
n = len(ID)
t1 = time.time()
################## for data creation #############################
def source_creation(source_list_in, source_queue_in):
    source_list = []
    for ID, Fx, e_Fx, z, z_flag, field in source_list_in:
        source = AGN( ID, Fx, e_Fx, z, z_flag, field )
        source.PDFz()
        source.make_grid()
        source.PDFf()
        source_list.append(source)
    source_queue_in.put(source_list)

################## multiprocessing for data #############################
Nprocess = 8
step = int( len(ID)//Nprocess )
source_queue_list = []
source_list_list = []
if Nprocess > 1:
    n_range = xrange(0, Nprocess)     
    for n in n_range:
        start_point = n*step
        end_point = (n+1)*step
        if n < Nprocess-1:
            temp_list = zip(ID[start_point:end_point], Fx[start_point:end_point], e_Fx[start_point:end_point], z[start_point:end_point], z_flag[start_point:end_point], field[start_point:end_point])
        else:
            temp_list = zip(ID[start_point:], Fx[start_point:], e_Fx[start_point:], z[start_point:], z_flag[start_point:], field[start_point:])
        source_list_list.append(temp_list)
        source_queue_list.append(Queue())
else:
    source_list_list.append( zip(ID[:], Fx[:], e_Fx[:], z[:], z_flag[:], field[:]) )
    source_queue_list.append( Queue() )    

############################### create data ########################################################
source_processes = []
for source_queue_in, source_list_in in zip(source_queue_list, source_list_list):
    p = Process(target=source_creation, args=(source_list_in, source_queue_in))
    p.start()
    source_processes.append(p)
    
source_list = []
for q in source_queue_list:
    source_list.extend( q.get() )
            
for p in source_processes:
    p.join()
        
print "Source list creation:", time.time()-t1,"sec"
print "-------------------------------------------"
############################## multiprocessing for likelihood ########################
Nprocess = 8
step = int( len(source_list)//Nprocess )
queue_list = []
list_list = []
if Nprocess > 1:
    n_range = xrange(0, Nprocess)
    for n in n_range:
        queue_list.append(Queue())
        if n < Nprocess-1:
            temp_list = source_list[n*step:(n+1)*step]
        else:
            temp_list = source_list[n*step:]
        list_list.append(temp_list)
else:
    list_list.append( source_list[:] )
    queue_list.append( Queue() )
##########################################################################################
def myprior(mp, oldparams):
    return 0.

def myloglike(mp, oldparams):
    m = mp.contents
    params = [m.params.contents.data[i] for i in range(m.n_params)]
    LogL = Marshall_Likelihood(*params)
    return -LogL
       
###########################################################################
import scipy
import numpy
import math
import sys
import matplotlib.pyplot as plt
import pyapemost

def show(filepath):
	""" open the output (pdf) file for the user """
	import subprocess, os
	if os.name == 'mac': subprocess.call(('open', filepath))
	elif os.name == 'nt': os.startfile(filepath)
	elif os.name == 'posix': subprocess.call(('xdg-open', filepath))

pyapemost.set_function(myloglike, myprior)

if len(sys.argv) < 2:
	#print "SYNOPSIS: %s [calibrate|run|analyse]" % sys.argv[0]
	cmd = ["calibrate", "run", "analyse"]
else:
	cmd = sys.argv[1:]

if "calibrate" in cmd:
	pyapemost.calibrate()
# run APEMoST
if "run" in cmd:
	#w = pyapemost.watch.ProgressWatcher()
	#w.start()
	pyapemost.run(max_iterations = 100000, append = False)
	#w.stop()

# lets analyse the results
if "analyse" in cmd:
	plotter = pyapemost.analyse.VisitedAllPlotter()
	plotter.plot()
	show("chain0.pdf")
	histograms = pyapemost.create_histograms()
	i = 1
	plt.clf()
	plt.figure(figsize=(7, 4 * len(histograms)))
	for k,(v,stats) in histograms.iteritems():
		plt.subplot(len(histograms), 1, i)
		plt.plot(v[:,0], v[:,2], ls='steps--', label=k)
		plt.legend()
		print k, stats
		i = i + 1
	plt.savefig("marginals.pdf")
	show("marginals.pdf")
	
	print pyapemost.model_probability()
