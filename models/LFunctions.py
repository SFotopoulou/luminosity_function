from numpy import where, power, log10, array, shape, ndarray, arange, exp
import warnings
from math import pi, cos
import sys
sys.path.append('/home/Sotiria/workspace/Luminosity_Function/src/LF_modules/configuration')

from AGN_LF_config import LF_config

""" All luminosities are inserted as logarithms
    Includes models with luminosity and density evolution"""

def Phi0( Lx, z, params, scale=1.):
    """Luminosity function at z=0: broken power law"""
    return  1. / ((((power(10., Lx))/(scale * (power(10., params.L0))))**params.g1)+
                  (((power(10., Lx))/(scale * (power(10., params.L0))))**params.g2))

#   Pure luminosity evolution - Ueda 2003
def PLE( L_in, z_in, params):
    if isinstance(L_in, float) and isinstance(z_in, float):
        eL = where(z_in <= params.zc, power((1. + z_in), params.p1), 
                   (power((1. + params.zc), params.p1))*
                   (power((1. + z_in) / (1.0 + params.zc), params.p2)))
        
        Phi = Phi0(L_in, z_in, params, eL)
        return Phi * power(10.0, params.Norm)
    else:
        if isinstance(L_in, float) and not isinstance(z_in, float):
            L_in = array([L_in]*len(z_in))
            z_in = array(z_in)
        elif isinstance(z_in, float) and isinstance(L_in, array): 
            z_in = array([z_in]*len(L_in))
            L_in = array(L_in)
        
        eL = where(z_in <= params.zc, power((1. + z_in), params.p1), 
                       (power((1. + params.zc), params.p1))*
                       (power((1. + z_in) / (1.0 + params.zc), params.p2)))
            
        return Phi0(L_in, z_in, params, eL) * power(10.0, params.Norm)
    

def halted_PLE( L_in, z_in, params):
    if isinstance(L_in, float) and isinstance(z_in, float):
        eL = where(z_in <= params.zc, power((1. + z_in), params.p1), power((1. + params.zc), params.p1))
        
        Phi = Phi0(L_in, z_in, params, eL)
        return Phi * power(10.0, params.Norm)
    else:
        if isinstance(L_in, float) and not isinstance(z_in, float):
            L_in = array([L_in]*len(z_in))
            z_in = array(z_in)
        elif isinstance(z_in, float) and isinstance(L_in, array): 
            z_in = array([z_in]*len(L_in))
            L_in = array(L_in)
        
        eL = where(z_in <= params.zc, power((1. + z_in), params.p1), power((1. + params.zc), params.p1))
        
        return Phi0(L_in, z_in, params, eL) * power(10.0, params.Norm)
        

#   modified Pure luminosity evolution - Silverman 2008
def mod_PLE(L_in, z_in, params):
    if isinstance(L_in, float) and isinstance(z_in, float):
        xi = log10((1.0 + z_in)/(1.0 + params.zc))
        params.L0 = params.L0 + params.p1 * xi + params.p2 * power(xi,2.0)
        eL = 1.0
        params.g2 = params.g2 * power((1.0 + z_in)/(1.0 + params.zc), params.a)        
        Phi = Phi0(L_in, z_in, params, eL)
        return Phi * power(10.0, params.Norm)
    else:
        if isinstance(L_in, float) and not isinstance(z_in, float):
            L_in = array([L_in]*len(z_in))
            z_in = array(z_in)
        elif isinstance(z_in, float) and isinstance(L_in, array): 
            z_in = array([z_in]*len(L_in))
            L_in = array(L_in)
        
        xi = log10((1.0 + z_in)/(1.0 + params.zc))
        params.L0 = params.L0 + params.p1 * xi + params.p2 * power(xi,2.0)
        eL = 1.0
        params.g2 = params.g2*power((1.0 + z_in)/(1.0 + params.zc), params.a)        
        return Phi0(L_in, z_in, params, eL) * power(10.0, params.Norm)
        

#   Pure luminosity evolution - Silverman 2008
def PLE2(L_in, z_in, params):
    if isinstance(L_in, float) and isinstance(z_in, float):
        xi = log10((1.0 + z_in)/(1.0 + params.zc))
        params.L0 = params.L0 + params.p1 * xi + params.p2 * power(xi, 2.0)
        eL = 1.0
        Phi = Phi0(L_in, z_in, params, eL)
        return Phi* power(10.0, params.Norm)
    else:
        if isinstance(L_in, float) and not isinstance(z_in, float):
            L_in = array([L_in]*len(z_in))
            z_in = array(z_in)
        elif isinstance(z_in, float) and isinstance(L_in, array): 
            z_in = array([z_in]*len(L_in))
            L_in = array(L_in)
        
        
        xi = log10((1.0 + z_in)/(1.0 + params.zc))
        params.L0 = params.L0 + params.p1 * xi + params.p2 * power(xi, 2.0)
        eL = 1.0
        return Phi0(L_in, z_in, params, eL) * power(10.0, params.Norm)
        
    
#   Pure luminosity evolution - Aird 2010
def PLE3(L_in, z_in, params):
    if isinstance(L_in, float) and isinstance(z_in, float):
        xi = (1.0 + params.zc)/(1.0 + z_in)
        params.L0 = params.L0 - log10(power(xi, params.p1) + power(xi, params.p2))
        eL = 1.0
        Phi = Phi0(L_in, z_in, params, eL)
        return Phi* power(10.0, params.Norm)
    else:
        if isinstance(L_in, float) and not isinstance(z_in, float):
            L_in = array([L_in]*len(z_in))
            z_in = array(z_in)
        elif isinstance(z_in, float) and isinstance(L_in, array): 
            z_in = array([z_in]*len(L_in))
            L_in = array(L_in)
        
        xi = (1.0 + params.zc)/(1.0 + z_in)
        params.L0 = params.L0 - log10(power(xi, params.p1) + power(xi, params.p2))
        eL = 1.0
        return Phi0(L_in, z_in, params, eL) * power(10.0, params.Norm)
        

#   Pure density evolution - Ueda 2003
def PDE( L_in, z_in, params):
    if isinstance(L_in, float) and isinstance(z_in, float):
        eD = where(z_in <= params.zc, power((1. + z_in), params.p1), 
                   (power((1. + params.zc), params.p1))*
                   (power((1. + z_in) / (1.0 + params.zc), params.p2)))
        
        Phi = Phi0(L_in, z_in, params)
        return Phi * eD * power(10.0, params.Norm)
    else:
        if isinstance(L_in, float) and not isinstance(z_in, float):
            L_in = array([L_in]*len(z_in))
            z_in = array(z_in)
        elif isinstance(z_in, float) and isinstance(L_in, array): 
            z_in = array([z_in]*len(L_in))
            L_in = array(L_in)
        
        eD = where(z_in <= params.zc, power((1. + z_in), params.p1), 
                   (power((1. + params.zc), params.p1))*
                   (power((1. + z_in) / (1.0 + params.zc), params.p2)))
        
        return Phi0(L_in, z_in, params) * eD * power(10.0, params.Norm)
    
       
#   Pure density evolution - Ueda 2003
def halted_PDE( L_in, z_in, params):
    if isinstance(L_in, float) and isinstance(z_in, float):
        eD = where(z_in <= params.zc, power((1. + z_in), params.p1), (power((1. + params.zc), params.p1)))
        
        Phi = Phi0(L_in, z_in, params)
        return Phi * eD * power(10.0, params.Norm)
    else:
        if isinstance(L_in, float) and not isinstance(z_in, float):
            L_in = array([L_in]*len(z_in))
            z_in = array(z_in)
        elif isinstance(z_in, float) and isinstance(L_in, array): 
            z_in = array([z_in]*len(L_in))
            L_in = array(L_in)
        
        eD = where(z_in <= params.zc, power((1. + z_in), params.p1), 
                   (power((1. + params.zc), params.p1)))
        
        return Phi0(L_in, z_in, params) * eD * power(10.0, params.Norm)

# Miyaji et al. 2000
def Miyaji( L_in, z_in, params):
    if isinstance(L_in, float) and isinstance(z_in, float):            
        if (z_in <= params.zc) and (L_in < params.La):
            e = (1. + z_in)**(max(params.pmin, params.p1 - params.a * (params.La - L_in)))
        elif (z_in <= params.zc) and (L_in >= params.La):
            e = (1. + z_in)**params.p1
        elif (z_in > params.zc) and (L_in < params.La):
            e = ((1. + params.zc)**(max(params.pmin, params.p1 - params.a * (params.La-L_in)))) * ((1. + z_in) / (1. + params.zc))**params.p2
        elif (z_in > params.zc) and (L_in >= params.La):
            e = ((1. + params.zc)**(params.p1)) * ((1. + z_in) / (1. + params.zc))**params.p2
        Phi = Phi0(L_in, z_in, params)
        return Phi * e * power(10.0, params.Norm)
    else:
        if isinstance(L_in, float):
            L_in = [L_in]*len(z_in)
        elif isinstance(z_in, float): 
            z_in = [z_in]*len(L_in)
        Phi_out = []
        for Lx, z in zip(L_in, z_in):
            if (z <= params.zc) and (Lx < params.La):
                e = (1. + z)**(max(params.pmin, params.p1 - params.a * (params.La - Lx)))
            elif (z <= params.zc) and (Lx >= params.La):
                e = (1. + z)**params.p1
            elif (z > params.zc) and (Lx < params.La):
                e = ((1. + params.zc)**(max(params.pmin, params.p1 - params.a * (params.La - Lx)))) * ((1. + z) / (1. + params.zc))**params.p2
            elif (z > params.zc) and (Lx >= params.La):
                e = ((1. + params.zc)**(params.p1)) * ((1. + z) / (1. + params.zc))**params.p2
            Phi_out.append(Phi0(Lx, z, params) * e * power(10.0, params.Norm))
        return Phi_out


def Miyaji15( L_in, z_in, params):
    if isinstance(L_in, float) and isinstance(z_in, float):
        p1L = params.p1 + params.b1 * (L_in - 44.)
        p2L = params.p2 + params.b2 * (L_in - 44.)
        
        Lx10 = power(10.0, L_in)
        La10 = power(10.0, params.Lb)
        
        zc_case1 = params.zb0
        zc_case2 = params.zb0 * power( Lx10 / La10, params.a )
        zcrit = where(L_in >= params.Lb, zc_case1, zc_case2)
        
        ec_case1 = power( (1. + z_in), p1L )
        ec_case2 = power( (1. + zcrit), p1L) * power( (1. + z_in) / (1. + zcrit), p2L)
        ec_case3 = power( (1. + params.zb2), p1L) * power( (1. + z_in) / (1. + params.zb2), params.p3)
        ec = where(z_in <= zcrit, ec_case1, ec_case2)
        ed = where(z_in > params.zb2, ec_case3, ec)
        
        Phi = Phi0(L_in, z_in, params)
        return Phi * ed * power(10.,params.Norm)    
    else:    
        if isinstance(L_in, float):
            L_in = array([L_in]*len(z_in))
            z_in = array(z_in)
        elif isinstance(z_in, float): 
            z_in = array([z_in]*len(L_in))
            L_in = array(L_in)
        
        p1L = params.p1 + params.b1 * (L_in - 44.)
        p2L = params.p2 + params.b2 * (L_in - 44.)
        
        Lx10 = power(10.0, L_in)
        La10 = power(10.0, params.Lb)
        
        zc_case1 = params.zb0
        zc_case2 = params.zb0 * power( Lx10 / La10, params.a )
        zcrit = where(L_in >= params.Lb, zc_case1, zc_case2)
        
        ec_case1 = power( (1. + z_in), p1L )
        ec_case2 = power( (1. + zcrit), p1L) * power( (1. + z_in) / (1. + zcrit), p2L)
        ec_case3 = power( (1. + zcrit), p1L) * power( (1. + params.zb2) / (1. + zcrit), p2L) * power( (1. + z_in) / (1. + params.zb2), params.p3)
        ec = where(z_in <= zcrit, ec_case1, ec_case2)
        ed = where(z_in > params.zb2, ec_case3, ec)
        
        return Phi0(L_in, z_in, params) * ed * power(10.,params.Norm)
        

#   Ueda et al. 2003
def Ueda( L_in, z_in, params):
    if isinstance(L_in, float) and isinstance(z_in, float):                
        Lx10 = power(10.0, L_in)
        La10 = power(10.0, params.La)
        
        zc_case1 = params.zc
        zc_case2 = params.zc * power( Lx10 / La10, params.a )
        zcrit = where(L_in >= params.La, zc_case1, zc_case2)
        
        ec_case1 = power( (1. + z_in), params.p1 )
        ec_case2 = power( (1. + zcrit), params.p1) * power( (1. + z_in) / (1. + zcrit), params.p2)
        ec = where(z_in <= zcrit, ec_case1, ec_case2)
        
        Phi = Phi0(L_in, z_in, params)
        return Phi * ec * power(10.0, params.Norm)
    else:
        if isinstance(L_in, float) and not isinstance(z_in, float):
            L_in = array([L_in]*len(z_in))
            z_in = array(z_in)
        elif isinstance(z_in, float) and isinstance(L_in, array): 
            z_in = array([z_in]*len(L_in))
            L_in = array(L_in)    
                
        Lx10 = power(10.0, L_in)
        La10 = power(10.0, params.La)
        
        zc_case1 = params.zc
        zc_case2 = params.zc * power( Lx10/La10, params.a )
        zcrit = where(L_in >= params.La, zc_case1, zc_case2)
        
        ec_case1 = power( (1. + z_in), params.p1 )
        ec_case2 = power( (1. + zcrit), params.p1) * power( (1. + z_in) / (1. + zcrit), params.p2)
        ec = where(z_in <= zcrit, ec_case1, ec_case2)
        
        return Phi0(L_in, z_in, params) * ec * power(10.0, params.Norm)
    
#   Ueda et al. 2014
def Ueda14( L_in, z_in, params):
    
    if isinstance(L_in, float) and isinstance(z_in, float):                
        Lx10 = power(10.0, L_in)
        La110 = power(10.0, params.La1)
        La210 = power(10.0, params.La2)
        p1_evol = params.p1 + params.beta * (L_in - params.Lp)
        
        zc_1 = params.zc1
        zc_2 = params.zc2

        zc_case1 = params.zc1 * power( Lx10 / La110, params.a1 )
        zc_case2= params.zc2 * power( Lx10 / La210, params.a2 )
        
        
        zcrit1 = where(L_in >= params.La1, zc_1, zc_case1)
        zcrit2 = where(L_in >= params.La2, zc_2, zc_case2)
        
        ec_case1 = power( (1. + z_in), p1_evol )
        ec_case2 = power( (1. + zcrit1), p1_evol) * power( (1. + z_in) / (1. + zcrit1), params.p2)
        ec_case3 = power( (1. + zcrit1), p1_evol) * power( (1. + zcrit2) / (1. + zcrit1), params.p2) * power( (1. + z_in) / (1. + zcrit2), params.p3)
        
        
        ec = where(z_in <= zcrit1, ec_case1, where(z_in > zcrit2, ec_case3, ec_case2))
        
        #print p1_evol, ec
        
        Phi = Phi0(L_in, z_in, params)
        return Phi * ec * power(10.0, params.Norm)
    

    else:
        if isinstance(L_in, float) and not isinstance(z_in, float):
            L_in = array([L_in]*len(z_in))
            z_in = array(z_in)
        elif isinstance(z_in, float) and isinstance(L_in, ndarray): 
            z_in = array([z_in]*len(L_in))
            L_in = array(L_in)    
                
        Lx10 = power(10.0, L_in)
        La110 = power(10.0, params.La1)
        La210 = power(10.0, params.La2)
        p1_evol = params.p1 + params.beta * (L_in - params.Lp)
        
        zc_1 = params.zc1
        zc_2 = params.zc2

        zc_case1 = params.zc1 * power( Lx10 / La110, params.a1 )
        zc_case2= params.zc2 * power( Lx10 / La210, params.a2 )
        
        
        zcrit1 = where(L_in >= params.La1, zc_1, zc_case1)
        zcrit2 = where(L_in >= params.La2, zc_2, zc_case2)
        
        ec_case1 = power( (1. + z_in), p1_evol )
        ec_case2 = power( (1. + zcrit1), p1_evol) * power( (1. + z_in) / (1. + zcrit1), params.p2)
        ec_case3 = power( (1. + zcrit1), p1_evol) * power( (1. + zcrit2) / (1. + zcrit1), params.p2) * power( (1. + z_in) / (1. + zcrit2), params.p3)
        
        
        ec = where(z_in <= zcrit1, ec_case1, where(z_in > zcrit2, ec_case3, ec_case2))
   
        #print p1_evol, ec  
        
        return Phi0(L_in, z_in, params) * ec * power(10.0, params.Norm)
    
    
    
#   Hasinger et al. 2005
def Hasinger( L_in, z_in, params):
    if isinstance(L_in, float) and isinstance(z_in, float):
        p1L = params.p1 + params.b1 * (L_in - 44.)
        p2L = params.p2 + params.b2 * (L_in - 44.)
        
        Lx10 = power(10.0, L_in)
        La10 = power(10.0, params.La)
        
        zc_case1 = params.zc
        zc_case2 = params.zc * power( Lx10 / La10, params.a )
        zcrit = where(L_in >= params.La, zc_case1, zc_case2)
        
        ec_case1 = power( (1. + z_in), p1L )
        ec_case2 = power( (1. + zcrit), params.p1) * power( (1. + z_in) / (1. + zcrit), p2L)
        ec = where(z_in <= zcrit, ec_case1, ec_case2)
        
        Phi = Phi0(L_in, z_in, params)
        return Phi * ec * power(10.,params.Norm)    
    else:    
        if isinstance(L_in, float):
            L_in = array([L_in]*len(z_in))
            z_in = array(z_in)
        elif isinstance(z_in, float): 
            z_in = array([z_in]*len(L_in))
            L_in = array(L_in)
        
        p1L = params.p1 + params.b1 * (L_in - 44.)
        p2L = params.p2 + params.b2 * (L_in - 44.)
        
        Lx10 = power(10.0, L_in)
        La10 = power(10.0, params.La)
        
        zc_case1 = params.zc
        zc_case2 = params.zc * power( Lx10 / La10, params.a )
        zcrit = where(L_in >= params.La, zc_case1, zc_case2)
        
        ec_case1 = power( (1. + z_in), p1L )
        ec_case2 = power( (1. + zcrit), params.p1) * power( (1. + z_in) / (1. + zcrit), p2L)
        ec = where(z_in <= zcrit, ec_case1, ec_case2)
        
        return Phi0(L_in, z_in, params) * ec * power(10.,params.Norm)
        
#   Barger et al. 2005, Yencho et al. 2009
def ILDE( L_in, z_in, params):
    if isinstance(L_in, float) and isinstance(z_in, float):
        eL = power((1. + z_in), params.p1)
        eD = power((1. + z_in), params.p2)     
        Phi = Phi0(L_in, z_in, params, eL)       
        return Phi * eD * power(10.0, params.Norm)
    else:
        if isinstance(L_in, float) and not isinstance(z_in, float):
            L_in = array([L_in]*len(z_in))
            z_in = array(z_in)
        elif isinstance(z_in, float) and not isinstance(L_in, float): 
            z_in = array([z_in]*len(L_in))
            L_in = array(L_in)            

        eL = power((1. + z_in), params.p1)
        eD = power((1. + z_in), params.p2)     
        return Phi0(L_in, z_in, params, eL) * eD * power(10.0, params.Norm)       
    

#   Barger et al. 2005, Yencho et al. 2009
def halted_ILDE( L_in, z_in, params):
    if isinstance(L_in, float) and isinstance(z_in, float):
        if z_in <= params.zc:
            eL = power((1. + z_in), params.p1)
            eD = power((1. + z_in), params.p2)     
            Phi = Phi0(L_in, z_in, params, eL)       
            return Phi * eD * power(10.0, params.Norm)
        else:
            eL = power((1. + params.zc), params.p1)
            eD = power((1. + params.zc), params.p2)     
            Phi = Phi0(L_in, params.zc, params, eL)       
            return Phi * eD * power(10.0, params.Norm)
    else:
        if isinstance(L_in, float):
            L_in = array([L_in]*len(z_in))
            z_in = array(z_in)
        elif isinstance(z_in, float): 
            z_in = array([z_in]*len(L_in))
            L_in = array(L_in)
        
        eL = where(z_in <= params.zc, power((1. + z_in), params.p1), power((1. + params.zc), params.p1))            
        eD = where(z_in <= params.zc, power((1. + z_in), params.p2), power((1. + params.zc), params.p2))
        return Phi0(L_in, z_in, params, eL) * eD * power(10.0, params.Norm)       
        
    
#   Aird et al. 2005
def LADE( L_in, z_in, params):
    if isinstance(L_in, float) and isinstance(z_in, float):
        zzc = (1. + params.zc) / (1. + z_in)
        #eL = (power(1+zc,p1)+power(1+zc,p2)) /(power(zzc,p1)+power(zzc,p2)) 
        eL = 1.0 / (power(zzc, params.p1) + power(zzc, params.p2)) 
        K = power(10., (params.d * (1. + z_in)) )
        Phi = Phi0(L_in, z_in, params, eL)       
        return Phi * K * power(10.0, params.Norm)
    else:
        if isinstance(L_in, float) and not isinstance(z_in, float):
            L_in = array([L_in]*len(z_in))
            z_in = array(z_in)
        elif isinstance(z_in, float) and not isinstance(L_in, float): 
            z_in = array([z_in]*len(L_in))
            L_in = array(L_in)
        
        zzc = (1. + params.zc)/(1. + z_in)
        #eL = (power(1+zc,p1)+power(1+zc,p2)) /(power(zzc,p1)+power(zzc,p2)) 
        eL = 1.0 / (power(zzc, params.p1) + power(zzc, params.p2)) 
        K = power(10., (params.d * (1. + z_in)) )
        
        return Phi0(L_in, z_in, params, eL) * K * power(10.0, params.Norm)       
    

#   Aird et al. 2005
def halted_LADE( L_in, z_in, params):
    print "Help!"
    if isinstance(L_in, float) and isinstance(z_in, float):
        zzc = (1. + params.zc) / (1. + z_in)
        #eL = (power(1+zc,p1)+power(1+zc,p2)) /(power(zzc,p1)+power(zzc,p2)) 
        eL = 1.0 / (power(zzc, params.p1) + power(zzc, params.p2)) 
        K = power(10., (params.d * (1. + z_in)) )
        Phi = Phi0(L_in, z_in, params, eL)       
        return Phi * K * power(10.0, params.Norm)
    else:
        if isinstance(L_in, float):
            L_in = array([L_in]*len(z_in))
            z_in = array(z_in)
        elif isinstance(z_in, float): 
            z_in = array([z_in]*len(L_in))
            L_in = array(L_in)

        zzc = (1. + params.zc)/(1. + z_in)
        #eL = (power(1+zc,p1)+power(1+zc,p2)) /(power(zzc,p1)+power(zzc,p2)) 
        eL = 1.0 / (power(zzc, params.p1) + power(zzc, params.p2)) 
        K = power(10., (params.d * (1. + z_in)) )
        return Phi0(L_in, z_in, params, eL) * K * power(10.0, params.Norm)       
   
#    LDDE: Fotopoulou et al. 2012
def Fotopoulou( L_in, z_in, params):

    if isinstance(L_in, float) and isinstance(z_in, float):                
        Lx10 = power(10.0, L_in)
        La10 = power(10.0, params.La)
        
        zc_case1 = params.zc
        zc_case2 = params.zc * power( Lx10 / La10, params.a )
        zcrit = where(L_in >= params.La, zc_case1, zc_case2)
        
        norm = power( (1.0 + zcrit), params.p1) + power( (1.0 + zcrit), params.p2)
        ez = norm/(power( (1. + z_in) / (1. + zcrit), -params.p1) + power( (1. + z_in) / (1. + zcrit), -params.p2) )
        
        Phi = Phi0(L_in, z_in, params)
        return Phi * ez * power(10.0, params.Norm)
    else:
        if isinstance(L_in, float) and not isinstance(z_in, float):
            L_in = array([L_in]*len(z_in))
            z_in = array(z_in)
        elif isinstance(z_in, float) and not isinstance(L_in, float): 
            z_in = array([z_in]*len(L_in))
            L_in = array(L_in)
        
        #print type(z_in), type(L_in)
        La10 = power(10.0, params.La)
        zc_case1 = params.zc
        zc_case2 = params.zc * power( power( 10.0, L_in) / La10, params.a )
        zcrit = where(L_in >= params.La, zc_case1, zc_case2)

        norm = power( (1.0 + zcrit), params.p1) + power( (1.0 + zcrit), params.p2)
        ez = norm/(power( (1. + z_in) / (1. + zcrit), -params.p1) + power( (1. + z_in) / (1. + zcrit), -params.p2) )

        return Phi0(L_in, z_in, params) * ez * power(10.0, params.Norm)
        
#    LDDE: Fotopoulou et al. 2012
def Fotopoulou2( L_in, z_in, params):

    if isinstance(L_in, float) and isinstance(z_in, float):                
        Lx10 = power(10.0, L_in)
        La10 = power(10.0, params.La)
        
        #zc_case1 = params.zc
        #zc_case2 = params.zc * power( Lx10 / La10, params.a )
        #zcrit = where(L_in >= params.La, zc_case1, zc_case2)
        zcrit = params.zc * power( Lx10 / La10, params.a )
        norm = power( (1.0 + zcrit), params.p1) + power( (1.0 + zcrit), params.p2)
        ez = norm/(power( (1. + z_in) / (1. + zcrit), -params.p1) + power( (1. + z_in) / (1. + zcrit), -params.p2) )
        
        Phi = Phi0(L_in, z_in, params)
        return Phi * ez * power(10.0, params.Norm)
    else:
        if isinstance(L_in, float) and not isinstance(z_in, float):
            L_in = array([L_in]*len(z_in))
            z_in = array(z_in)
        elif isinstance(z_in, float) and not isinstance(L_in, float): 
            z_in = array([z_in]*len(L_in))
            L_in = array(L_in)
        
        #print type(z_in), type(L_in)
        La10 = power(10.0, params.La)
        #zc_case1 = params.zc
        #zc_case2 = params.zc * power( power( 10.0, L_in) / La10, params.a )
        #zcrit = where(L_in >= params.La, zc_case1, zc_case2)
        zcrit = params.zc * power( power( 10.0, L_in) / La10, params.a )
        norm = power( (1.0 + zcrit), params.p1) + power( (1.0 + zcrit), params.p2)
        ez = norm/(power( (1. + z_in) / (1. + zcrit), -params.p1) + power( (1. + z_in) / (1. + zcrit), -params.p2) )

        return Phi0(L_in, z_in, params) * ez * power(10.0, params.Norm)

def Fotopoulou3( L_in, z_in, params):

    if isinstance(L_in, float) and isinstance(z_in, float):                
        Lx10 = power(10.0, L_in)
        L010 = power(10.0, params.L0)
        
        #zc_case1 = params.zc
        #zc_case2 = params.zc * power( Lx10 / La10, params.a )
        #zcrit = where(L_in >= params.La, zc_case1, zc_case2)
        zcrit = params.zc * power( Lx10 / L010, params.a )
        norm = power( (1.0 + zcrit), params.p1) + power( (1.0 + zcrit), params.p2)
        ez = norm/(power( (1. + z_in) / (1. + zcrit), -params.p1) + power( (1. + z_in) / (1. + zcrit), -params.p2) )
        
        Phi = Phi0(L_in, z_in, params)
        return Phi * ez * power(10.0, params.Norm)
    else:
        if isinstance(L_in, float) and not isinstance(z_in, float):
            L_in = array([L_in]*len(z_in))
            z_in = array(z_in)
        elif isinstance(z_in, float) and not isinstance(L_in, float): 
            z_in = array([z_in]*len(L_in))
            L_in = array(L_in)
        
        #print type(z_in), type(L_in)
        L010 = power(10.0, params.L0)
        #zc_case1 = params.zc
        #zc_case2 = params.zc * power( power( 10.0, L_in) / La10, params.a )
        #zcrit = where(L_in >= params.La, zc_case1, zc_case2)
        zcrit = params.zc * power( power( 10.0, L_in) / L010, params.a )
        norm = power( (1.0 + zcrit), params.p1) + power( (1.0 + zcrit), params.p2)
        ez = norm/(power( (1. + z_in) / (1. + zcrit), -params.p1) + power( (1. + z_in) / (1. + zcrit), -params.p2) )

        return Phi0(L_in, z_in, params) * ez * power(10.0, params.Norm)

def halted_Fotopoulou( L_in, z_in, params):
    
    if isinstance(L_in, float) and isinstance(z_in, float):                
        Lx10 = power(10.0, L_in)
        La10 = power(10.0, params.La)
        
        zc_case1 = params.zc
        zc_case2 = params.zc * power( Lx10 / La10, params.a )
        zcrit = where(L_in >= params.La, zc_case1, zc_case2)
        
        ec_case1 = power( (1. + z_in), params.p1 )
        ec_case2 = power( (1. + zcrit), params.p1) 
        ec = where(z_in <= zcrit, ec_case1, ec_case2)
        
        Phi = Phi0(L_in, z_in, params)
        return Phi * ec * power(10.0, params.Norm)
    else:
        if isinstance(L_in, float):
            L_in = array([L_in]*len(z_in))
            z_in = array(z_in)
        elif isinstance(z_in, float): 
            z_in = array([z_in]*len(L_in))
            L_in = array(L_in)    
        
        Lx10 = power(10.0, L_in)
        La10 = power(10.0, params.La)
        
        zc_case1 = params.zc
        zc_case2 = params.zc * power( Lx10/La10, params.a )
        zcrit = where(L_in >= params.La, zc_case1, zc_case2)
        
        ec_case1 = power( (1. + z_in), params.p1 )
        ec_case2 = power( (1. + zcrit), params.p1) 
        ec = where(z_in <= zcrit, ec_case1, ec_case2)
        
        return Phi0(L_in, z_in, params) * ec * power(10.0, params.Norm)
    
def zk(k,n, zmax=LF_config.zmax):
    z_poly = 0.5*(1.0 + cos((k+0.5)*pi/(n+1.0)))*log10(1.0+zmax)
    zk = power(10.0, z_poly) - 1
    return zk
    
def x_cheby(z,zmax=LF_config.zmax):    
    return 2.0*(log10(1.0+z)/log10(1.0+zmax))-1.0
    
def K_Cheby(n, K0, K1, z):
    zk0 = zk(0, n) # k, n
    xk0 = x_cheby(zk0)
    zk1 = zk(1, n) # k, n
    xk1 = x_cheby(zk1)
    zk2 = zk(2, n) # k, n
    xk2 = x_cheby(zk2)
    
    xz = x_cheby(z)
    logK = K0 + K1 +\
            + xk0*K0*xz + xk1*K1*xz +\
            + ( (2.0*xk0*xk0-1.0)*K0 + (2.0*xk1*xk1-1.0)*K1 ) * (2.0*xz*xz-1.0)
    return logK    
    
def L_Cheby(n, L0, L1, L2, z):
    zk0 = zk(0, n) # k, n
    xk0 = x_cheby(zk0)
    zk1 = zk(1, n) # k, n
    xk1 = x_cheby(zk1)
    zk2 = zk(2, n) # k, n
    xk2 = x_cheby(zk2)
    xz = x_cheby(z)
    logL = L0 + L1 + L2\
            + xk0*L0*xz + xk1*L1*xz + xk2*L2*xz +\
            + ( (2.0*xk0*xk0-1.0)*L0 + (2.0*xk1*xk1-1.0)*L1 + (2.0*xk2*xk2-1.0)*L2 ) * (2.0*xz*xz-1.0)
    return logL    
    
def g1_Cheby(n, g0, z):
    zk0 = zk(0, n) # k, n
    xk0 = x_cheby(zk0)
    xz = x_cheby(z)
    g = g0 +\
            + xk0*g0*xz +\
            + ( (2.0*xk0*xk0-1.0)*g0 ) * (2.0*xz*xz-1.0)
    return g    
     
    
def FDPL(L_in, z_in, params):
    Lx10 = power(10.0, L_in)
    K0 = params.K0
    K1 = params.K1
    K = K_Cheby(2, K0, K1, z_in)
    
    L0 = params.K0
    L1 = params.L1
    L2 = params.L2
    Lstar = L_Cheby(3, L0, L1, L2, z_in)
    
    g0 = params.g1
    g1 = g1_Cheby(1, g0, z_in)
    g2 = params.g2
    
    Phi = power(10.0,K) / ( (Lx10/power(10., Lstar))**g1+(Lx10/power(10., Lstar))**g2 )

    return Phi



def Schechter(L_in, z_in, params):
    Lratio = power(10.0, L_in)/power(10.0, params.Lx)
    a = params.a*(1+z_in)**params.b
    #Phi = power(10.0, params.A)*power(Lratio, -a)*exp(-Lratio)
    Phi = 0.4 * log(10) * power(10.0, params.A)*power(0.4*(params.Lx - L_in))
    print
    print params.A, params.Lx, params.a, params.b, Lratio, log10(Phi)
    return Phi
    
    
if __name__ == '__main__':
    print zk(1,2,6)
    print x_cheby(zk(0,2,6))

    for z in arange(0, 5.0):
        print K_Cheby(2, -6.19, -4.44, z)






def highz(L_in ,z_in, params):
    ed = power( ((1.0+z_in)/(1.0+3.0)), (params.q + params.b * (L_in - 44.)))
     
    return Phi0(L_in, z_in, params) * ed * power(10.,params.Norm)













 