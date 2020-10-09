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

#ztype = LF_config.ztype
#z_unc = LF_config.z_unc
#z_unc = 'True'
fields= ['MAXI','HBSS','COSMOS', 'LH', 'X_CDFS', 'AEGIS']
all_ztypes= [ ('measured', 'False'), ('measured', 'True'), ('all', 'True')]

model_parameters = ['L_0', 'g_1', 'g_2', 'p_1', 'p_2', 'z_c', 'L_a', 'a', 'Norm']

marginals_path = '/home/Sotiria/workspace/Luminosity_Function/src/MultiNest/'

bins = 'freedman'
marginals = {}

param_indx = range(2, 11)

pp = PdfPages('plots/'+LF_config.model+'_marginal_dist.pdf')    

for param_indx in param_indx: 
    fig = plt.figure(figsize=(17,10))
    fig.subplots_adjust(left=0.06, top=0.85, bottom=0.10, right=0.96, wspace=0.13, hspace=0.05)   
    xall = []
    yall = []
    for setup in all_ztypes:
        ztype = setup[0]
        z_unc = setup[1]
        
        indx = all_ztypes.index(setup)+1
 
        fig.add_subplot(1, len(all_ztypes), indx)
    
    
        all_fields_folder = '_'.join(LF_config.fields) + '_' +'_ztype_' + \
                            ztype+"_"+ LF_config.model + '_pl' + str(LF_config.pl) + '_zunc_' + str(z_unc)
        
        all_fields_file = marginals_path + all_fields_folder + '/1-.txt'
        marginals['all'] = np.loadtxt(all_fields_file)
                    
        for fld in fields:    
            
            folder_name = fld + '_' + '_ztype_' + ztype + '_' + LF_config.model + '_pl' + \
                          str(LF_config.pl) + '_zunc_' + str(z_unc)
            
            file_name = marginals_path + folder_name + "/1-.txt"
            marginals[fld] = np.loadtxt(file_name)
            
            n = hist(marginals[fld][:, param_indx], bins=bins, histtype='step',label=fld, lw=4, normed=True)
            xall.extend( list(n[1]) )
            yall.extend( list(n[0]) )

        if indx == 2: plt.xlabel( '$\mathrm{'+model_parameters[param_indx-2]+'}$' )
        
        n = hist(marginals['all'][:, param_indx], bins=bins, histtype='bar', label='all fields', color='gray', ec='gray', normed=True, zorder=-1)
        xall.extend( list(n[1]) )
        yall.extend( list(n[0]) )
            
        if indx !=1 :  plt.yticks(visible=False)

        xmin = np.min( xall ) - 0.1
        ymin = np.min( yall )

        xmax = np.max( xall ) + 0.1
        ymax = np.max( yall ) + 0.1
        
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        
        if ztype == 'measured':
            zname = 'sources with z'
        else:
            zname = 'all sources'
            
        plt.text(xmin + (xmax-xmin)*0.05, ymax - (ymax-ymin)*0.07, zname, size='medium')
        plt.text(xmin + (xmax-xmin)*0.05, ymax - (ymax-ymin)*0.12, 'z PDF '+z_unc, size='medium')        
        
    plt.legend(bbox_to_anchor=(-2.26, 1.06, 3.26, 0.1,), loc=2, ncol=4, mode="expand", borderaxespad=0.)
    plt.draw()
    pp.savefig()#'plots/LDDE_'+model_parameters[param_indx-2]+'.pdf', dpi=200)
    #plt.show()

pp.close()
    
print 'done'