import sys
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
from astroML.plotting import hist
from AGN_LF_config import LF_config
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
from scipy import interpolate
import itertools
from matplotlib.backends.backend_pdf import PdfPages

fields = ['MAXI','XMM-HBSS','XMM-COSMOS', 'XMM-LH', 'XMM-CDFS', 'Chandra-COSMOS', 'Chandra-AEGIS','Chandra-CDFS']
zorder = [10, 9, 8, 17, 6, 5, 3, 2, 1]
colors = itertools.cycle(['#00AA00', '#0099FF', '#0066FF', '#0000AA','black', '#FF0000', '#990000','#440000'])
all_ztypes= [ ('all', 'True')]

model_parameters = ['$\mathrm{\log{L_0}}$', '$\mathrm{\gamma_1}$', '$\mathrm{\gamma_2}$', '$\mathrm{p_1}$', '$\mathrm{p_2}$', '$\mathrm{z_c}$', '$\mathrm{\log{L_\\alpha}}$', '$\mathrm{\\alpha}$', '$\mathrm{\log{A}}$']
param_names = ['L_0', 'g_1', 'g_2', 'p_1', 'p_2', 'z_c', 'L_a', 'a', 'Norm']

marginals_path = '/home/Sotiria/workspace/Luminosity_Function/src/MultiNest/new/'

bins = 'freedman'
marginals = {}
ticks = {}
ticks['L_0'] = range(41,47)
ticks['g_1'] = [-1.5, 0, 1.5, 3.0, 4.5]
ticks['g_2'] = [-1.5, 0, 1.5, 3.0, 4.5]
ticks['p_1'] = [0, 2.5, 5, 7.5, 10]
ticks['p_2'] = [-9, -6, -3, 0, 3]
ticks['z_c'] = range(0, 5)
ticks['L_a'] = range(41, 47)
ticks['a'] = [0.0, 0.3, 0.6, 0.9]
ticks['Norm'] = [-10, -8, -6, -4, -2]

save=True
parameters = range(2, 11)

all_fields_file = '/home/Sotiria/Dropbox/transfer/XLF_output_files/combination_fields/All_Coherent/1-.txt'

marginals_file = {'MAXI':'/home/Sotiria/Dropbox/transfer/XLF_output_files/individual_fields/MAXI/1-.txt',
             'XMM-HBSS':'/home/Sotiria/Dropbox/transfer/XLF_output_files/individual_fields/XMM_HBSS/1-.txt',
             'XMM-COSMOS':'/home/Sotiria/Dropbox/transfer/XLF_output_files/individual_fields/XMM_COSMOS/1-.txt',
             'XMM-LH':'/home/Sotiria/Dropbox/transfer/XLF_output_files/individual_fields/XMM_LH/1-.txt',
             'XMM-CDFS':'/home/Sotiria/Dropbox/transfer/XLF_output_files/individual_fields/XMM_CDFS/1-.txt',
             'Chandra-COSMOS':'/home/Sotiria/Dropbox/transfer/XLF_output_files/individual_fields/Chandra_COSMOS/1-.txt',
             'Chandra-AEGIS':'/home/Sotiria/Dropbox/transfer/XLF_output_files/individual_fields/Chandra_AEGIS/1-.txt',
             'Chandra-CDFS':'/home/Sotiria/Dropbox/transfer/XLF_output_files/individual_fields/Chandra_CDFS/1-.txt'
    }


fig = plt.figure(figsize=(10,10))
fig.subplots_adjust(left=0.02, top=0.85, bottom=0.05, right=0.98, wspace=0.15, hspace=0.29) 
for param_indx in parameters: 
      
    xall = []
    yall = []
    for setup in all_ztypes:
        ztype = setup[0]
        z_unc = setup[1]
        
        indx = parameters.index(param_indx)+1
    
        fig.add_subplot(3, 3, indx)
    
        marginals['all'] = np.loadtxt(all_fields_file)
                    
        for fld in fields:    
            #folder_name = fld + '_' + '_ztype_' + ztype + '_' + LF_config.model + '_pl' + \
            #              str(LF_config.pl) + '_zunc_' + str(z_unc)
            
            #file_name = marginals_path + folder_name + "/1-.txt"
            marginals[fld] = np.loadtxt(marginals_file[fld])
            
            n = hist(marginals[fld][:, param_indx], bins=bins, histtype='step',label=fld, lw=2.5, normed=True, color=colors.next(), zorder=zorder[fields.index(fld)] )
            xall.extend( list(n[1]) )
            yall.extend( list(n[0]) )

        n = hist(marginals['all'][:, param_indx], bins=bins, histtype='bar', label='all fields', color='gray', ec='gray', normed=True, zorder=-1)
    
        xall.extend( list(n[1]) )
        yall.extend( list(n[0]) )
               
        xmin = np.min( xall ) - (np.max( xall )-np.min( xall ))*0.01
        ymin = np.min( yall )
    
        xmax = np.max( xall ) + (np.max( xall )-np.min( xall ))*0.005
        ymax = np.max( yall ) + (np.max( yall )-np.min( yall ))*0.00
        
        plt.text(xmin + (xmax-xmin)*0.08, ymax - (ymax-ymin)*0.2, model_parameters[param_indx-2], size='medium')

        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax+0.05*(ymax-ymin)])
        plt.yticks(visible=False)
        
        plt.xticks(ticks[param_names[param_indx-2]], size='x-small')
plt.legend(bbox_to_anchor=(-2.3, 4.10, 3.3, 0.1,), loc=2, ncol=3, mode="expand", borderaxespad=0.)
if save == True:
    plt.savefig('plots/20151030_LDDE_marginals_new.pdf', dpi=300)

plt.show()