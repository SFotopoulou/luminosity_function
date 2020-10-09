import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
import copy
# Add the ptdraft folder path to the sys.path list
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/models')
from AGN_LF_config import LF_config
import astroML as ML
from scipy import interpolate, integrate
from astroML.plotting import hist 
from scipy.stats.kde import gaussian_kde
fields = ['MAXI','HBSS','COSMOS', 'LH', 'X_CDFS', 'AEGIS']
all_ztypes= [ ('all', 'True')]
model_parameters = ['L0', 'g1', 'g2', 'p1', 'p2', 'zc', 'La', 'a', 'Norm']

marginals_path = '/home/Sotiria/workspace/Luminosity_Function/src/MultiNest/'
ztype = 'all'
z_unc = 'True'
bins = 'freedman'
marginals = {}
PDF = {}
parameters = range(2, 11)

fig = plt.figure(figsize=(10,10))
fig.subplots_adjust(left=0.12, top=0.87, bottom=0.07, right=0.98, wspace=0.15, hspace=0.29) 

for param_indx in parameters: 
    param_name = model_parameters[param_indx-2]
    all_fields_folder = '_'.join(LF_config.fields) + '_' +'_ztype_' + \
                        ztype+"_"+ LF_config.model + '_pl' + str(LF_config.pl) + '_zunc_' + str(z_unc)
    
    all_fields_file = marginals_path + all_fields_folder + '/1-.txt'
    marginals['all'] = np.loadtxt(all_fields_file)
    n = hist(marginals['all'][:, param_indx], bins=bins, histtype='bar', label='all fields', color='gray', ec='gray', normed=True, zorder=-1)
    x = n[1][:-1] + (n[1][1] - n[1][0])/2.
    y = n[0][:]
    
    xmin = np.min(x)
    xmax = np.max(x)
    param_min, param_max = LF_config.Prior[param_name]

    if xmin > param_min:
        x = np.insert(x, 0, param_min)
        y = np.insert(y, 0, 0 )
    if xmax < param_max:        
        x = np.append(x, param_max)
        y = np.append(y, 0)

    
    PDF['all',param_name] = interpolate.interp1d(x, y)                    


    for fld in fields:    
        
        folder_name = fld + '_' + '_ztype_' + ztype + '_' + LF_config.model + '_pl' + \
                      str(LF_config.pl) + '_zunc_' + str(z_unc)
        
        file_name = marginals_path + folder_name + "/1-.txt"
        marginals[fld] = np.loadtxt(file_name)
        n = hist(marginals[fld][:, param_indx], bins=bins, histtype='step',label=fld, lw=2, normed=True)
        x = n[1][:-1] + (n[1][1] - n[1][0])/2.
        y = n[0][:]
        
        xmin = np.min(x)
        xmax = np.max(x)
        
        if xmin > param_min:
            x = np.insert(x, 0, param_min)
            y = np.insert(y, 0, 1e-10 )
        if xmax < param_max:        
            x = np.append(x, param_max)
            y = np.append(y, 1e-10)

#        if fld=='MAXI':
#            print y
        y[y==0]=1e-10
#            print y
#            print
        PDF[fld, param_name] = interpolate.interp1d(x, y,'linear')
plt.clf()

for field1 in fields:
    out_file = open('KLD_'+field1+'.txt','w')
    out_file.write('# parameter field ' + ' '.join(fields)+' ' + '_err '.join(fields)+'_err\n')

    for param_indx in parameters: 
        param_name = model_parameters[param_indx-2]
        param_min, param_max = LF_config.Prior[param_name]
        #xx = np.linspace(param_min, param_max, 100)
        print param_name, param_min, param_max


        Q = PDF[field1, param_name]
        KLD = []
        KLDerr = []
        for field2 in fields:
            P = PDF[field2, param_name]
            #x = np.linspace(param_min, param_max)
#            print
#            print param_name
#            print field1, Q(param_min), Q(param_max)
#            print field2, P(param_min), P(param_max) 
#            
#            print P(np.linspace(param_min, param_max))/Q(np.linspace(param_min, param_max))
#  
            def f(x):
                return np.log( P(x)/Q(x) ) * P(x)
            
#            plt.plot( x, f(x)) 
#            plt.show()
#            if param_name=='Norm':
#                print field2, P(param_min), P(param_max)
#                print field1, Q(param_min), Q(param_max)
            D1, D2 = integrate.quad( f, param_min, param_max, limit=2000)
            KLD.append( str(D1) )
            KLDerr.append( str(D2) )
            if D1<0:
                print field1, field2, D1
                x = np.linspace(param_min, param_max,20)
                print Q(x)
                print P(x)
                #plt.plot(x, f(x))
                #plt.show()
        out_file.write( param_name + ' ' + field1 + ' ' + str(' '.join(KLD)) + ' ' + str(' '.join(KLDerr)) + '\n')
    out_file.close()