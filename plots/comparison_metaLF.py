import sys
# Append the module path to the sys.path list
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from Source import get_flux

########################################################
# Data
########################################################

Lx = np.linspace(40.0, 47.0)
redshift = [0.104, 1.161, 2.421, 3.376]
zbin = [0.01, 0.2, 1.0, 1.2, 2.0, 3.0, 4.0]
zii = [0.1, 1.13157894737, 2.42105263158, 3.45263157895]

path = '/home/Sotiria/workspace/Luminosity_Function/src/LF_plots/forPaper/'
metaLF_file = '/home/Sotiria/workspace/Luminosity_Function/src/meta-luminosity_function/metaLF-results/16/output_extrapolated/collated.txt'
dtype = ['L','z','median','q01','q99','+1sigma','-1sigma','+3sigma','-3sigma','q10','q90']
d = np.loadtxt(metaLF_file, dtype=[(d,'f') for d in dtype])

x_plots = 2
y_plots = 2 
gray = 'gray'#(0.85, 0.85, 0.85)

models = ['LDDE', 'LADE', 'ILDE', 'PDE', 'PLE']

for model in models:  
    fig = plt.figure(figsize=(10,10))
    fig.subplots_adjust(left=0.16, right=0.97, wspace=0.05, hspace=0.05)

    LF_interval = np.loadtxt(path + 'data/dPhi_interval_'+model+'_for_metaLF.dat')
    
    for zi in zii :
        ax = fig.add_subplot(y_plots, x_plots, zii.index(zi)+1)
    
        e = d[d['z'] == zi]
        e = e[-np.isnan(e['median'])]
        plt.fill_between(x=e['L'], y1 =e['q01'], y2=e['q99'], color=(0.85, 0.85, 0.85))
        plt.fill_between(x=e['L'], y1 =e['q10'], y2=e['q90'], color='gray' )
        plt.plot(e['L'], e['median'], color='black', ls='-', lw=4)
        zz = np.where( np.abs(LF_interval[:,1] - zi)<0.001 )
        
        LF_ll = LF_interval[zz, 0][0]
        LF_mode = LF_interval[zz, 6][0]

        LF_low90 = LF_interval[zz, 2][0]
        LF_high90 = LF_interval[zz, 3][0]

        LF_low99 = LF_interval[zz, 4][0]
        LF_high99 = LF_interval[zz, 5][0]
        
#        vflux = np.vectorize(get_flux)
#        Flux = vflux(LF_ll, [zi]*len(LF_ll))
#        mask_low = np.where((np.array(Flux)>5e-16) & (np.array(Flux)<6e-11), 0, 1)
#        #print mask_low
#    
#        LF_ll = ma.masked_array(np.array(LF_ll) ,mask_low,hard_mask=True)    
#        LF_mode = ma.masked_array(np.array(LF_mode), mask_low,hard_mask=True)
#    
#        LF_low90 = ma.masked_array(np.array(LF_low90), mask_low,hard_mask=True)
#        LF_high90 = ma.masked_array(np.array(LF_high90), mask_low,hard_mask=True)
#
#        LF_low99 = ma.masked_array(np.array(LF_low99), mask_low,hard_mask=True)
#        LF_high99 = ma.masked_array(np.array(LF_high99), mask_low,hard_mask=True)


        plt.plot(LF_ll, LF_mode,  color='red', ls = '-', zorder=5)
            
        plt.plot(LF_ll, LF_low90,  color='red', ls = '--', zorder=5)
        plt.plot(LF_ll, LF_high90, color='red', ls = '--', zorder=5)
        
        plt.plot(LF_ll, LF_low99,  color='red', ls = '-.', zorder=5)
        plt.plot(LF_ll, LF_high99, color='red', ls = '-.', zorder=5)
    
        i = zii.index(zi)
    
        plt.xticks([43, 44, 45], visible=False)
        plt.yticks([-10, -8, -6, -4, -2], visible=False)
        
        if i == 0 or i == 2 : plt.yticks(visible=True)
        if i == 2 or i == 3 : plt.xticks(visible=True)

        if i == 3:
            i = i +2
            plt.xlabel('Luminosity', fontsize='x-large')
            ax.xaxis.set_label_coords(-0.075, -0.15)
        if i == 2:
            i = i +2
            plt.ylabel(r'd$\Phi$/dlogLx', fontsize='x-large')
            ax.yaxis.set_label_coords(-0.25, 1.0)
        if i ==1 :
            i = i+1
            
        
        ax.annotate(str(zbin[i])+"$< $"+"z"+"$ < $"+str(zbin[i+1]), (0.1, 0.1) , xycoords='axes fraction', fontstyle='oblique', fontsize='medium', )
    
    
        plt.ylim([-11.5,-0.5])
        plt.xlim([42, 46 ])

    #plt.suptitle(model)
    #for ext in ['jpg','pdf','eps','png']:    
    plt.savefig(path + 'plots/210_metaLF_with_'+model+'.pdf')
    
    #plt.show()
