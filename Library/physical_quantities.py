#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains analytical calculations for all physical quantities used throughout project

CONTAINS
    FUNCTION : calc_T_kep
    FUNCTION : calc_tau_infall
    FUNCTION : calc_Veff_gr
    FUNCTION : calc_Veff_n
    FUNCTION : calc_Veff_extrema
    FUNCTION : calc_precession_analyt
"""

import numpy as np


def calc_T_kep(a, R_s=1):
    '''Calculates the period of a keplerian orbit (Newtonian physics) of semi-major axis a 
    around an object of SC radius R_s
    
    INPUT
        a : Semi-major axis
        R_s : Schwarzchild radius. Set to 1 as default
        
    OUTPUT
        Orbital period'''
    
    return 2*np.pi*(2*a**3/R_s)**.5



def calc_tau_infall(r0, R_s=1):
    '''Calculate the analytical expression for the time to fall radially, from rest, to the event horison
    
    INPUT
        r0 : Initial radius 
        R_s : Schwarzchild radius. Set to 1 as default
        
    OUTPUT
        Radial infall time'''
        
    return r0**1.5/R_s**.5 * (np.pi/2 + (R_s/r0*(1-R_s/r0))**.5 + np.arctan(-(R_s/(r0-R_s))**.5))



def calc_Veff_gr(r,j, R_s=1):
    '''Calculates the effective potential for a relativistic orbit of specific angular momentum j at radius r
    
    INPUT
        r : Radius
        j : Specific angular momentum
        R_s : Schwarzchild radius. Set to 1 as default
        
    OUTPUT
        Effective potential'''
    
    return j**2/(2*r**2) * (1-R_s/r) - R_s/(2*r)



def calc_Veff_n(r,j, R_s=1):
    '''Calculates the effective potential for a newtonian orbit of specific angular momentum j at radius r
    
    INPUT
        r : Radius
        j : Specific angular momentum
        R_s : Schwarzchild radius. Set to 1 as default
        
    OUTPUT
        Effective potential'''
    
    return j**2/(2*r**2) - R_s/(2*r)



def calc_Veff_extrema(j, R_s=1, Veff_type='both'):
    '''Calculates the minima of the newtonian and relativistic effective potential of specific angular momentum j
    and the maxima of the relativistic potential.
    
    INPUT
        j : Specific angular momentum
        R_s : Schwarzchild radius. Set to 1 as default
        Veff_type : Set to 'gr' for only relativistic extrema, 'n' for only newtonian extrema
            and 'both' for both. Set to 'both' as default
             
    OUTPUT
        Which of the following are outputed depends on Veff_type
            r_uo_gr : Radius of unstable circular orbit (Relativistic)
            r_so_gr : Radius of stable circular orbit (Relativistic)
            r_so_n : Radius of stable circular orbit (Newtonian)'''
             
    
    #Keys for type of V_eff
    keys = {'gr':0, 'n':1, 'both':2}
    Veff_type = keys[Veff_type]
    
    #Newtonian potential
    if Veff_type == 1 or Veff_type == 2:
        r_so_n = 2*j**2/R_s
      
        #Return only newtonian values
        if Veff_type == 1:
            return r_so_n
    
    #General relativity potential
    if Veff_type == 0 or Veff_type == 2:
        if j <= 3**.5:
            r_uo_gr = np.nan
            r_so_gr = np.nan
            
            print('j is less than j_min - no real solution. Returned 0 for both rs.')
        
        elif j > 3**.5:
            #Unstable circular orbit
            r_uo_gr = j**2/R_s * (1 - (1-3*R_s**2/j**2)**.5)
            
            #Stable circular orbit
            r_so_gr = j**2/R_s * (1 + (1-3*R_s**2/j**2)**.5)
        
        #Return only GR values
        if Veff_type == 0:
            return r_uo_gr, r_so_gr
    
    #Reuturn all
    if Veff_type == 2:  
        return r_uo_gr, r_so_gr, r_so_n
    
    
    
def calc_precession_analyt(j, R_s=1):
    '''Calculates the approximate analytical expression for precession of the perihelion
    
    INPUT
        j : Specific angular momentum
        R_s : Schwarzchild radius. Default set to 1
        
    OUTPUT
        Analytical procession of phi'''
    
    return 3*np.pi/2 * (R_s/j)**2

