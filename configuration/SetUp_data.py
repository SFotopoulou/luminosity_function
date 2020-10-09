import sys
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')
from AGN_LF_config import LF_config
from Source import *
from Survey import *
from class_source import AGN
import numpy as np
import time
from multiprocessing import Process, Queue
from astropy.io import fits as pyfits

import cPickle



class Set_up_data():
        
    def paste_dict_to_fits(self, outname, key_list, dict):
        """ Saves python dictionary to fits file"""
        tbhdu = pyfits.new_table(pyfits.ColDefs([ pyfits.Column(name = key,
                                                                format = 'E',
                                                                array = dict[key]) for key in key_list] ))

        # Overwrite file if existing
        tbhdu.writeto(outname, clobber=True)
                    
    def set_up_data(self):
        data_in = read_data()
        t1 = time.time()
        key_list = []
        source_list = []
        fig = plt.figure(figsize=(10,10))
        fig.subplots_adjust(left=0.175, right=0.98, wspace=0.05, hspace=0.05)
                
        for field in LF_config.fields:
            #print data_in
            name = data_in['ID_'+field]
            counts = data_in['Counts_'+field]
            e_counts = data_in['e_Counts_'+field]            
            flux = data_in['F_'+field]
            eflux = data_in['F_err_'+field]
            mag = data_in['mag_'+field]
            redshift = data_in['Z_'+field]
            redshift_flag = data_in['Z_flag_'+field]
            
            key_list.append('ID_'+field)
            key_list.append('Counts_'+field)
            key_list.append('e_Counts_'+field)
            key_list.append('F_'+field)
            key_list.append('F_err_'+field)
            key_list.append('mag_'+field)
            key_list.append('Z_'+field)
            key_list.append('Z_flag_'+field)
            key_list.append('Lum_'+field)
            key_list.append('Lum_err_'+field)

            for ID, cnt, e_cnt, Fx, e_Fx, R, z, z_flag in zip(name, counts, e_counts, flux, eflux,mag, redshift, redshift_flag):
                #print field, ID,Fx, e_Fx, z
                lum= get_luminosity(Fx, e_Fx, z)
                if LF_config.Lmin <= lum <= LF_config.Lmax and LF_config.zmin <= z <= LF_config.zmax :
                    source = AGN( ID, cnt, e_cnt, Fx, e_Fx, R, z, z_flag, field)
                    if LF_config.z_unc == True and LF_config.L_unc == True:
                        source.PDFz()
                        source.make_grid()
                        source.PDFf()
                    elif LF_config.z_unc == True and LF_config.L_unc == False:
                        source.PDFz()
                        source.make_data_vol()
                        source.make_lumis()
                    else:
                        pass
                    source_list.append(source)
                
            luminosity = get_luminosity(flux, eflux, redshift)
            
            data_in["Lum_"+field] = luminosity
            data_in["Lum_err_"+field] = luminosity
            #print flux, eflux, redshift, luminosity, err_L

            if LF_config.plot_Lz == True or LF_config.save_Lz == True:                   
                
                plt.errorbar(np.array(redshift),10.**np.array(luminosity), label = field, markersize = 9.0,
                             marker = LF_config.mstyles.next(), 
                             markeredgecolor = LF_config.medgecolors.next(),
                             markerfacecolor = LF_config.mcolors.next(),
                             ecolor = LF_config.lcolors.next(), ls = ' ',
                             zorder = LF_config.zorder.next())
                plt.legend(loc=2)
                plt.yscale('log',nonposy='clip')
                plt.xscale('log',nonposx='clip')
                plt.xlim([0.008,5.0])
                plt.ylim([0.5e42, 6e45])
                plt.ylabel('$\mathrm{L_x /erg\cdot s^{-1}}$', fontsize='x-large')
                plt.xlabel('$\mathrm{Redshift}$', fontsize='x-large')
            
                
        if LF_config.save_Lz == True:                 
            plt.draw()
            for ext in ['eps','pdf','png','jpg']:
                plt.savefig('/home/Sotiria/workspace/Luminosity_Function/output_files/plots/Lx_z.'+ext)
                
        if LF_config.plot_Lz == True:       
            plt.show()    
            #plt.close()
            pass
            
        self.paste_dict_to_fits(LF_config.data_out_name, key_list, data_in)
        self.return_data = data_in
        print "Source list creation:", time.time()-t1,"sec"
        print "Ndata=", len(source_list)
        print "-------------------------------------------"
        plt.show()
        return source_list
    
    def get_data(self):
        if LF_config.depickle_data == True:
            sourcelist = open('sourceList.pkl', 'rb')
            source_list = cPickle.load(sourcelist)
            #
            indatalist = open('indataList.pkl', 'rb')
            indata_list = cPickle.load(indatalist)
        else:               
            # AGN objects list
            source_list = self.set_up_data()

            sourcelist = open('sourceList.pkl', 'wb')
            cPickle.dump(source_list, sourcelist)
            sourcelist.close()
            # Input dictionary
            indata_list = self.return_data

            indatalist = open('indataList.pkl', 'wb')
            cPickle.dump(indata_list, indatalist)
            indatalist.close()
            
        return source_list, indata_list
    
    def get_Ndata(self):
        return len(self.source_list) 

if __name__ == "__main__":
    s = Set_up_data()
    s.set_up_data()