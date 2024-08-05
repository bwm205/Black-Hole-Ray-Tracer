#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script compares a numerical calculation for the value of the perhelion procession to an approximate
    analytical expression. 
The user inputs a given specific angular momentum and perturbs the orbit (to produce precession).
The geodesic is calculated and by numerically finding successive minima of the radius, the value for the
    precession is calculated
This is then compared to the analytical expression with an error, and the orbit is plotted.

CONTAINS
    FUNCTION : find_extrema
    FUNCTION : calc_precession_numeric
"""


import numpy as np
import time
import os

from scipy.interpolate import CubicSpline

from Library.plotting import plot_geodesic, plot_validation
from Library.physical_quantities import calc_T_kep, calc_Veff_extrema, calc_precession_analyt
from Geodesic_Simulations.simulation import sim_geodesic



def find_extrema(x,y, which='both'):
    '''Finds the extrema of an array of data using cubic spline interpolation. The extrema are found
    as the roots of the first derivative
    
    INPUT
        x : x coordinate
        y : y coordinate
        which : Set to 'min' to return minima, 'max' to return maxima and 'both' to return both.
        Set to 'both' as default
        
    OUTPUT
        Eiher minima, maxima or both depending on 'which' input'''
    
    #Calcualte cubic spline
    spl = CubicSpline(x, y)
    
    #Find extrema as roots of first derivative
    roots = spl.derivative(1).roots(extrapolate=False)
    
    #Check if first ind is a min or max
    root1 = spl(roots[0])
    root2 = spl(roots[1])
    
    if root1 > root2:
        root1_type = 0
        
    elif root1 < root2:
        root1_type = 1

    #Return minima, maxima or all
    keys = {'min':0, 'max':1, 'both':2}
    which = keys[which]
    
    if which == 0:
        return roots[root1_type+1::2]
    
    elif which == 1:
        return roots[root1_type::2]
    
    elif which == 2:
        return roots
    
    

def calc_precession_numeric(phi,r):
    '''Numerically calculates the precession of the perihelion for a given orbit.
    The data is interpolated as a cubic spline and its minima (perihelion) are found as the as 
    the roots of the first derivative. Their difference is then calculated
    
    INPUT
        phi : Phi coordinates
        r : r coordinates
        
    OUTPUT
        delta_phi : The difference bettween successive minima, averaged for all orbits'''
    
    #Find minima
    phi_mins = find_extrema(phi,r, which='min')
    
    previous = 0
    
    #Apply periodic conditions
    for i,phi_min in enumerate(phi_mins):
        
        while phi_min >= 2*np.pi and phi_min-2*np.pi > previous:  #Ensure never bigger than 2pi
            phi_min -= 2*np.pi 
            
        phi_mins[i] = phi_min
        previous = phi_min
        
    #Hence calculate precession
    delta_phi = np.diff(phi_mins)
    
    return np.mean(delta_phi) #Return average



def main(j, perturb_factor, num_period, validate, image_filename=None):
    #Find circular orbit
    r_uo_gr, r_so_gr = calc_Veff_extrema(j, R_s=1, Veff_type='gr')
    r0 = r_so_gr
    
    r0 *= 1 + perturb_factor/100
    
    tau_final = calc_T_kep(r0, R_s=1) * num_period #Estimate orbial period using Newtonian mechanics and calculate roughly 5 periods
    step = tau_final/100000
    
    #Initial conditions
    U_r0 = 0
    U_phi0 = j/r0**2 #Convert to angular velocity
    phi0 = 0
    
    ## Simulaion ##
    
    start_time = time.perf_counter()
    
    print('Simulating... \n')
    geodesic = sim_geodesic(tau_final, step, U_r0,U_phi0,r0,phi0, photon=0, R_s=1, validate=validate, n_save=100000)
    end_time = time.perf_counter() 
    print('Simulation time = {:.2f} s \n'.format(end_time-start_time))
    
    #Numerical solution
    rs,phis = geodesic.r_history, geodesic.phi_history
    precession_numeric = calc_precession_numeric(phis,rs)
    
    #Analytical solution
    precession_analyt = calc_precession_analyt(j, R_s=1)
    
    #Error
    err_precession = (precession_numeric - precession_analyt)/precession_analyt
    
    print('Numerical precession = {:.5f} \nAnalytical precession = {:.5f} \nError = {:.2e} \n'.format(precession_numeric, precession_analyt, err_precession))
    
    if image_filename != None:
        if validate == 0:
            plot_geodesic(geodesic=geodesic, image_filename=image_filename+'.png')
            
        elif validate == 1:
            plot_geodesic(geodesic=geodesic, image_filename=image_filename+'_geodesic.png')
            plot_validation(geodesic=geodesic, taus=np.linspace(0,tau_final,100000), image_filename=image_filename+'_validation.png')
        
    elif image_filename == None:
        plot_geodesic(geodesic=geodesic, image_filename=None)
        
        if validate == 1:
            plot_validation(geodesic=geodesic, taus=np.linspace(0,tau_final,100000), image_filename=None)
    
    
        
if __name__ == "__main__":
    
    ## User Input ##
    
    #Initialise potential
    j = float(input('Provide specific angular momentum j : '))
        
    while j <= 3**.5:
        j = float(input('Value bellow j_min = 3**.5. Please input again : '))
    
    #Perturb to view precession
    perturb_factor = float(input('Provide percentage by which to perturb r0 : '))
    
    #Length of simulation
    num_period = float(input('Input how many orbital periods (keplerian) to integrate : '))
    
    #Choose to validate
    validate = int(input('Validate? \n1 = yes, 0 = no : '))
    
    #Choose to save plot
    plot = int(input('Save plots? \n1 = yes, 0 = no : '))
    
    if plot == 0:
        image_filename = None
    
    if plot == 1:
        image_filename = input('Provide filename for saving plot : ')
        image_filename = os.path.abspath(os.path.dirname(__file__)) + '/' + image_filename
        print('Plots will be saved to : {} '.format(image_filename))
    
    print('')
    
    
    ## Main Code ##
    
    main(j, perturb_factor, num_period, validate, image_filename=image_filename)
        
        
        