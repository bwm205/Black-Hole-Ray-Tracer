#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script demonstrates the simulated geodesics recovering a Keplerian orbit (Newtonian gravity) at high orbital radii.
    via the orbial period
The comparison bewtween Keplerian predictions and relativity can be made in through a single or muliple orbits.
For the single orbit, an initial radius is inputted and the corresponding keplerian angular velocity for a circualr orbit is calculated
    as an initial condion. The geodesic is then simulated an plotted and the Numerical and Keplerian orbital periods are calculated and printed
For multiple orbits, the user inputs a maximum radius and then orbital periods are calculated in the same way for a range of r0s. The Keplerian
    and relativistic periods are then plotted along with their error. The error should decrease with increasing initial radius, 
    showing the integrator recovering Newtonian gravity.
    
CONTAINS
    FUNCTION : calc_T_gr
    FUNCTION : plot_Ts
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import os

from tqdm import tqdm

from Library.classes import Geodesic
from Library.coordinate_transforms import pol2cart
from Library.plotting import plot_geodesic, plot_validation
from Library.physical_quantities import calc_T_kep
from Geodesic_Simulations.perihelion_precession import find_extrema



def calc_T_gr(r, phi, t):
    '''Calculates the period of an orbit (Relativistic physics) using numerically simulated data.
    The period is found as the distance between successive maxima
    
    INPUT
        r : Radial coordinate
        phi : Azimuthal coordinate
        t : Time for calculating period
        
    OUTPUT
        Numerically calculated orbital period'''
    
    x,y = pol2cart(r,phi)
    
    extrema = find_extrema(t,y, which='max')
    
    T_gr = np.diff(extrema)
    
    return np.mean(T_gr)


def plot_Ts(r0s, T_grs, T_keps, err_Ts, figsize=(10,10), image_filename=None):
    '''Plots the rate of orbital period as a function of the initial starting radius r0
    for both relativistic and Keplerian values, along with the respective error in numerical value
    
    INPUT
        r0s : Initial radii
        T_grs : Numerical, relativistic periods
        T_keps : Analytical keplerian periods
        err_Ts : Error between relativistic and Keplerian values
        figsize : Figure size. Default set to (10,10)
        image_filename : Filename to save plot. Ensure to inculde path, but not filetype.
            Default doesn't save plot'''
    
    
    #Plot periods
    fig = plt.figure(figsize=figsize)
    ax = fig.subplots()
    
    #Plot Ts
    ax.plot(r0s, T_grs, label='General Relativity')
    ax.plot(r0s, T_keps, label='Keplerian Prediction')
    
    
    #Configure plot
    ax.set(xlim=(np.min(r0s),np.max(r0s)), ylim=(0,np.max(T_keps)))
    ax.set_xlabel('$r_0$', fontsize=max(figsize)*2)
    ax.set_ylabel('$T$', fontsize=max(figsize)*2)
    ax.legend(fontsize=max(figsize)*1.5)
    
    #If requested, save plot
    if image_filename != None:
        plt.tight_layout()
        plt.savefig(image_filename + '_periods.png', dpi=360)

        print('Saved plotted periods to {}'.format(image_filename + '_periods.png'))
        
    plt.show()
    
    #Plot error
    fig = plt.figure(figsize=(10,10))
    ax = fig.subplots()
    
    #Plot taus
    ax.plot(r0s, err_Ts) 
    
    #Configure plot
    ax.set(xlim=(np.min(r0s),np.max(r0s)), ylim=(np.min(err_Ts), 0))
    ax.set_xlabel('$r_0$', fontsize=max(figsize)*2)
    ax.set_ylabel(r'$\Delta T$',  fontsize=max(figsize)*2)

    if image_filename != None:
        plt.tight_layout()
        plt.savefig(image_filename + '_errors.png', dpi=360)

        print('Saved plotted errors to {}'.format(image_filename + '_errors.png'))
    
    plt.show()
    


def main_individual_r(r0,image_filename=None, validate=0):
    
    #Keplerian period
    T_kep = calc_T_kep(r0, R_s=1)
    
    #Initial conditions
    U_r0 = 0.
    U_phi0 = r0**(-3/2) / 2**.5
    phi0 = 0.
    
    taus = np.arange(0, T_kep*5, T_kep*5/10000)  #Simulate 2 orbital periods
    
    geodesic = Geodesic(U_r0,U_phi0,r0,phi0)
    
    #Simulate
    for k,tau in enumerate(taus[1:]):
        geodesic.step(tau)
        
        #If crossed event horison, exit system
        if geodesic.status != 1:
            print('Trajectory crossed event horison. Please input a wider orbit')
            
            sys.exit()
        
    r,phi = geodesic.r_history, geodesic.phi_history
    
    #Relativistic period
    T_gr = calc_T_gr(r,phi,taus)
    
    print('Predicted orbital period = {:.2f} \nAnalytical orbital period = {:.2f} \nError = {:.2e} \n'.format(T_gr, T_kep, (T_gr-T_kep)/T_kep))
    
    #Plot and save if requested
    if image_filename != None:
        if validate == 0:
            plot_geodesic(geodesic=geodesic, image_filename=image_filename+'.png')
        
        elif validate == 1:
            plot_geodesic(geodesic=geodesic, image_filename=image_filename+'_geodesic.png')
            plot_validation(geodesic=geodesic, taus=taus, image_filename=image_filename+'_validation.png')
        
    elif image_filename == None:
        plot_geodesic(geodesic=geodesic, image_filename=None)

        if validate == 1:
            plot_validation(geodesic=geodesic, taus=taus, image_filename=None)
            
            
    
def main_multiple_rs(r0_max, image_filename=None):
    
    r0s = np.linspace(1,r0_max,100)
    
    T_keps = calc_T_kep(r0s, R_s=1)
    T_grs = np.zeros(len(r0s))
    
    for i,r0 in tqdm(enumerate(r0s)):
        
        #Initial conditions
        phi0 = np.pi/2.
        U_r0 = 0.
        U_phi0 = r0**(-3/2) / 2**.5
    
        geodesic = Geodesic(U_r0,U_phi0, r0,phi0)
        
        taus = np.arange(0, T_keps[i]*2, T_keps[i]*2/10000)  #Simulate 2 orbital periods
        
        #Simulate
        for k,tau in enumerate(taus[1:]):
            geodesic.step(tau)
            
            if geodesic.status != 1:
                break
            
        if geodesic.status == 1:
            r,phi = geodesic.r_history, geodesic.phi_history
            T_grs[i] = calc_T_gr(r,phi,taus)
            
            
        elif geodesic.status != 1:
            T_grs[i] = 0
            
    err_Ts = (T_grs - T_keps) / T_keps
        
    #Plot
    plot_Ts(r0s, T_grs, T_keps, err_Ts, figsize=(10,10), image_filename=image_filename)
        
    


if __name__ == "__main__":
    
    many_rs = int(input('Produce commparison plot for multiple radii? \nInput 1 for yes or 0 for no : '))

    
    if many_rs == 0:
        r0 = float(input('Provide individual initial radius : '))
        
        #Choose to validate
        validate = int(input('Validate? \n1 = yes, 0 = no : '))
        
        #Choose to save plot
        plot = int(input('Save geodesic plot? \n1 = yes, 0 = no : '))
        
        if plot == 0:
            image_filename = None
        
        elif plot == 1:
            image_filename = input('Provide filename for saving plot : ')
            image_filename = os.path.abspath(os.path.dirname(__file__)) + '/' + image_filename
            print('Plots will be saved to : {} '.format(image_filename))
            print('')
    
        main_individual_r(r0, image_filename=image_filename, validate=validate)
        
    
    if many_rs == 1:
        r0_max = float(input('Provide maximum initial radius : '))
        
        #Choose to save plot
        plot = int(input('Save plots? \n1 = yes, 0 = no : '))
        
        if plot == 0:
            image_filename = None
        
        elif plot == 1:
            image_filename = input('Provide filename for saving plot : ')
            image_filename = os.path.abspath(os.path.dirname(__file__)) + '/' + image_filename
            print('Plots will be saved to : {} '.format(image_filename))
        
        print('')
        
        main_multiple_rs(r0_max, image_filename)