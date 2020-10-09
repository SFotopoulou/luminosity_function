import sys
import os
# Add the ptdraft folder path to the sys.path list
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/models')
import numpy as np
from AGN_LF_config import LF_config
import matplotlib.pyplot as plt

folder_name = '_'.join(LF_config.fields) + '_' + LF_config.model + '_pl' + str(LF_config.pl) + '_zunc_' + str(LF_config.z_unc)
fields = LF_config.fields
#print folder_name
#input_files = ['/home/Sotiria/workspace/Luminosity_Function/src/MultiNest/'+ field + '_' + LF_config.model + '_pl' + str(LF_config.pl) + '_zunc_' + str(LF_config.z_unc)+'/1-.txt' for field in fields]
#input_files.append('/home/Sotiria/workspace/Luminosity_Function/src/MultiNest/AEGIS_' + LF_config.model + '_pl' + str(LF_config.pl) + '_zunc_False/1-.txt')

input_files= ['/home/Sotiria/workspace/Luminosity_Function/src/MultiNest/MAXI_LH_HBSS_COSMOS_X_CDFS_AEGIS_LDDE_pl0.9_zunc_True/1-.txt']
#print input_file
    #print marginals[:,2]

parameters = LF_config.parameters
n_params = len(parameters)

x_plots = 3
y_plots = x_plots
#print x_plots, y_plots
fig = plt.figure(figsize=(20,20))
fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.14,wspace=0.5, hspace=0.5)

for i in range(n_params):
    
    plt.subplot(x_plots, y_plots, i+1)
    for infile in input_files:
        marginals = np.loadtxt(infile)
        plt.hist(marginals[:,i+2], normed=True, histtype='step')
        
    plt.xlabel(parameters[i])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #if i == 1 : plt.title(folder_name.split('_')[0]+' '+folder_name.split('_')[1])
    
plt.show()