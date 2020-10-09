import numpy as np
from scipy.integrate import simps
from multiprocessing import Process, Queue
import time
from math import log10
import matplotlib.pyplot as plt
#
from AGN_LF_config import LF_config
from cosmology import dif_Vc
from LFunctions import *
from SetUp_data import Set_up_data
from SetUp_grid import set_up_grid


setup_data = Set_up_data()
source_list = setup_data.get_data()[0]

L, Z, DVcA, temp_Fx, area, Redshift_int, Luminosity_int = set_up_grid()

def Ndata():
    return len(source_list) 

################ Multiprocessing for data ################
Nprocess = LF_config.Nprocess
step = int( len(source_list)//Nprocess )
queue_list = []
list_list = []

if Nprocess > 1:
    n_range = range(0, Nprocess)
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
################################################      
def Phi(Lx, z, params):    
    """ 
    The luminosity function model 
    """

    if LF_config.model == 'LDDE':
        return Fotopoulou(Lx, z, params)
        
def Source_sublist(list_in,params,queue):
    """Source detection probability phi*dVc*Source_prob"""
    if LF_config.z_unc == True:
        result = []
        for source in list_in:
            xpdf = source.return_PDFz()[0]
            Grid = source.return_MultipliedGrid()
            Luminosity, Redshift = source.return_lumis()
            PHI_Lz = Phi(Luminosity, Redshift, params)            
            x = Grid*PHI_Lz
            int1 = simps(x, xpdf,even='avg')
            result.append(int1)

        result = np.array(result)
        result_log = np.log(result)
        res = sum(result_log) # sum for each sublist
        queue.put(res) 

    elif LF_config.z_unc == False:
        result = []
        for source in list_in:
            ID, field, counts, e_counts, flux, e_flux, mag, z, zflag = source.return_data()
            Lum = source.l
            t1 = time.time()
            PHI_Lz = Phi(Lum, z, params)
            t1 = time.time()
            Vol = dif_Vc(z)
            result.append( PHI_Lz*Vol )

        queue.put(sum(np.log10(result)))
    
def Marshall_Likelihood(params):
    """
    Likelihood for MLE L = -2.0*ln(L)
    """
    """ 
    Data probability
    """
    processes = []
    for queue_in, list_in in zip(queue_list, list_list):
        p = Process(target=Source_sublist, args=(list_in,params,queue_in))
        p.start()
        processes.append(p)
        
    result = []
    for q in queue_list:
        result.append( q.get() )
                
    for p in processes:
        p.join()
            
    source_d = sum(result) 
    PHI_Lz = Phi(L,Z,params)
    X = DVcA*PHI_Lz 
   
    y = []
    count_r = range(0, LF_config.Lpoints)

    simps_last = []
    tt = time.time()
    for count in count_r:

        startz = count * LF_config.zpoints
        endz = startz + LF_config.zpoints
        x = X[startz:endz]
        
        int1 = simps(x, Redshift_int, even='avg')
        simps_last.append(int1)
        y.append(int1)
    survey_d = simps(y, Luminosity_int, even='avg')
    
    Like = -source_d + survey_d 
    
    return 2.0*Like + LF_config.Likelihood_offset
    
class Params: pass
params = Params()

    
def Fotopoulou_Likelihood(L0, g1, g2, p1, p2, zc, La, a, Norm):
    params.L0 = L0
    params.g1 = g1
    params.g2 = g2
    params.p1 = p1
    params.p2 = p2
    params.zc = zc
    params.La = La
    params.a = a
    params.Norm = Norm
    return Marshall_Likelihood(params)

