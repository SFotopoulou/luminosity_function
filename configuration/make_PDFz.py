from astropy.io import fits as pyfits
import numpy as np
import time
from numpy import *
import matplotlib.pyplot as plt
from scipy.integrate import simps


class Spectrum:
    def __init__(self):#,infile):
        #self.infile=open(infile,'r').readlines() # open and read the whole file
        self.start=13
        pass
        
    def getheader(self): # read the header
        self.indent=self.infile[1].split()[0]
        #self.zspec=float(self.infile[1].split()[1])
        self.zphot=float(self.infile[1].split()[2])
        self.Nfilt=float(self.infile[3].split()[1])
        self.Npdf=float(self.infile[5].split()[1])  
        #print self.indent, self.zphot                    
        pass
    
    def getPDF(self,out=False): # read the PDF
        count1=int(self.start+self.Nfilt)
        count2=int(count1+self.Npdf)
        self.xpdf=[]
        self.ypdf=[]
        while count1<count2:
#            if float(self.infile[count1].split()[1])>0:
            self.xpdf.append(float(self.infile[count1].split()[0]))
            self.ypdf.append(float(self.infile[count1].split()[1]))
            count1+=1
        if self.xpdf[-1]==0:
            self.xpdf = self.xpdf[:-1]
            self.ypdf = self.ypdf[:-1]    
        if out==True:
            return self.xpdf, self.ypdf

    def return_PDF(self,field,ID,filename=None):
        if field == 1:
            try:
                filelist = open(filename, 'r').readlines()
            except IOError:
                print"input file", filename, "not known"
        elif field == 'COSMOS' or field == 'XMM_COSMOS':
            filelist = open('/home/Sotiria/Documents/Luminosity_Function/data/XMM_COSMOS/catalogs/photoz/UH_zphot_cosmos/COSMOS_filelist','r').readlines()            
        elif field == 'LH':
            filelist = open('/home/Sotiria/Documents/Luminosity_Function/data/XMM_LH/catalogs/photoz/LH_filelist','r').readlines()     
        elif field == 'AEGIS':
            filelist = open('/home/Sotiria/Documents/Luminosity_Function/data/AEGIS/catalogs/photoz/AEGIS_filelist','r').readlines()            
        elif field == 'X_CDFS' or field == 'XMM_CDFS':
            filelist = open('/home/Sotiria/Documents/Luminosity_Function/data/XMM_CDFS/Li-Ting/specFile/XMM_CDFS_filelist','r').readlines()    
        elif field == 'Chandra_CDFS':
            filelist = open('/home/Sotiria/Documents/Luminosity_Function/data/Chandra_CDFS/PDZ_190115/Chandra_CDFS_filelist.dat','r').readlines()    
        elif field == 'Chandra_COSMOS':
            filelist = open('/home/Sotiria/Documents/Luminosity_Function/data/Chandra_COSMOS/zPDF/Chandra_COSMOS_filelist.dat','r').readlines()    
        elif field == 'XXL_North':
            file = pyfits.open('/home/Sotiria/workspace/Luminosity_Function/input_files/PDFs/XXL_North_photoz_PDF.fits')[1].data    
        elif field == 'XXL_South':
            file = pyfits.open('/home/Sotiria/workspace/Luminosity_Function/input_files/PDFs/XXL_South_photoz_PDF.fits')[1].data                 
        elif field == 'other':
            try:
                filelist = open(filename, 'r').readlines()
            except IOError:
                print"input file", filename, "not known"
        
        if 'XXL' not in field:
            for file in filelist:
                fin = file.split('\n')[0]
                ff= fin.split('/')[-1:]
    
                for f in ff:
                    first = f.split('.spec')[0]
                    fid = int( first.split('Id')[1] )
    
                if fid == ID:
                    if field=='X_CDFS' or field == 'XMM_CDFS' or field=="Chandra_COSMOS":
                        xpdf, ypdf = loadtxt(fin, unpack=True)    
                    else:    
                        self.infile = open(fin,'r').readlines()
                        sed = self.getheader()    
                        xpdf, ypdf = self.getPDF(out=True)
                    
                    
                    integral1 = simps(ypdf, xpdf, even='avg')
    
                    # normalize PDF
                    ypdfz = ypdf/integral1
    
                    #return xpdf, ypdfz
                    
    ##    Find limits of PDF inside Level of Confidence
                    sigma_1 = 0.1586555 # 68.2689%
                    sigma_2 = 0.022750132 # 95.4499%
                    sigma_3 = 0.00135 # 99.7300%
                    sigma_5 = 0.000000287 # 99.9999%
                    
                    level = 0.0
                    # lower limit            
                    previous_integral = 0.0
                    for i in range(1, len(ypdfz)):
                        
                        integral = simps( ypdfz[0:i], xpdf[0:i] )
    
                        if integral>level:
                            ##printintegral,previous_integral
                            if abs( integral - level ) < abs (previous_integral - level) :
                                a = i
                            else:
                                a = i - 1
                                #print"lower limit"
                                #printypdfz[a], xpdf[a]
                                break
                            
                        previous_integral = integral
                        
                    # upper limit
                    
                    previous_integral = 0.0                
                    for j in range(len(ypdfz)-1, 0, -1):
                        
                        integral = simps(ypdfz[j:len(ypdfz)], xpdf[j:len(ypdfz)])
                        
                        if integral>level:
                            ##printintegral,previous_integral
                            if abs( integral - level ) < abs (previous_integral - level) :
                                b = j 
                            else:
                                b = j + 1
                            #print"upper limit"
                            #printypdfz[b], xpdf[b]
                            break
                        
                        previous_integral = integral
                    
                    #print"a=",a, " b= ",b
                    #printsimps(ypdfz[a:b+1], xpdf[a:b+1], dx = 0.01, even='first')
                    #print
                    return xpdf[a:b+1], ypdfz[a:b+1]
        else:    
             xpdf = np.array(list(np.arange(0, 6, 0.01)) + list(np.arange(6, 7.1, 0.2))) 
             #print ID
             ypdf = file[file.field('NUMBER') == int(ID)][0][1:]
             #print ypdf
             #print len(xpdf), len(ypdf)
             
             #plt.plot(xpdf,ypdf)
             #plt.show()
            
             integral1 = simps(ypdf, xpdf, even='avg')

            # normalize PDF
             ypdfz = ypdf/integral1

            #return xpdf, ypdfz
            
##    Find limits of PDF inside Level of Confidence
             sigma_1 = 0.1586555 # 68.2689%
             sigma_2 = 0.022750132 # 95.4499%
             sigma_3 = 0.00135 # 99.7300%
             sigma_5 = 0.000000287 # 99.9999%
            
             level = 0.0
            # lower limit            
             previous_integral = 0.0
             for i in range(1, len(ypdfz)):
                
                integral = simps( ypdfz[0:i], xpdf[0:i] )

                if integral>level:
                    ##printintegral,previous_integral
                    if abs( integral - level ) < abs (previous_integral - level) :
                        a = i
                    else:
                        a = i - 1
                        #print"lower limit"
                        #printypdfz[a], xpdf[a]
                        break
                    
                previous_integral = integral
                
            # upper limit
            
             previous_integral = 0.0                
             for j in range(len(ypdfz)-1, 0, -1):
                
                integral = simps(ypdfz[j:len(ypdfz)], xpdf[j:len(ypdfz)])
                
                if integral>level:
                    ##printintegral,previous_integral
                    if abs( integral - level ) < abs (previous_integral - level) :
                        b = j 
                    else:
                        b = j + 1
                    #print"upper limit"
                    #printypdfz[b], xpdf[b]
                    break
                
                previous_integral = integral
            
            #print"a=",a, " b= ",b
            #printsimps(ypdfz[a:b+1], xpdf[a:b+1], dx = 0.01, even='first')
            #print
             return xpdf[a:b+1], ypdfz[a:b+1]