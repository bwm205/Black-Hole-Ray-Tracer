#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script calculate the time to fall radially to the event horison by simulation and analytically
One may choose to calculate for a single radius r, which outputs a value and an error,
    or one may choose to calculate for many rs, which outputs a plot of values and errors
    
CONTAINS
    FUNCION : plot_tau_infall
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from tqdm import tqdm

from Library.classes import Geodesic
from Library.physical_quantities import calc_tau_infall



def plot_tau_infall(r0s, taus, tau_analyts, err_taus, figsize=(10,10), image_filename=None):
    '''Plots the rate of radial infall as a function of the initial starting radius r0
    for both numerical and analytical values, along with the respective error in numerical value
    
    INPUT
        r0s : Initial radii
        taus : Numerical radial infall times
        taus_analyt : Analytical radial infall times
        err_taus : Error between numerical and analytical values
        figsize : Figure size. Default set to (10,10)
        image_filename : Filename to save plot. Ensure to inculde path, but not filetype.
            Default doesn't save plot'''
    
    #Plot infall
    fig = plt.figure(figsize=figsize)
    ax = fig.subplots()
    
    #Plot taus
    ax.plot(r0s, taus, label='Numerical Estimate')
    ax.plot(r0s, tau_analyts, label='Analytical Solution')
    
    #Configure plot
    ax.set(xlim=(1,np.max(r0s)), ylim=(0,np.max(taus)))
    ax.set_xlabel('$r_0$', fontsize=max(figsize)*2)
    ax.set_ylabel(r'$\tau_{infall}$', fontsize=max(figsize)*2)
    ax.legend(fontsize=max(figsize)*1.5)
    
    #If requested, save plot
    if image_filename != None:
        plt.tight_layout()
        plt.savefig(image_filename + '_taus.png', dpi=360)

        print('Saved plotted taus to {}'.format(image_filename + '_taus.png'))
        
    plt.show()
    
    #Plot error
    fig = plt.figure(figsize=(10,10))
    ax = fig.subplots()
    
    #Plot taus
    ax.plot(r0s, err_taus) 
    
    #Configure plot
    ax.set(xlim=(1,np.max(r0s)), ylim=(np.min(err_taus), 0))
    ax.set_xlabel('$r_0$', fontsize=max(figsize)*2)
    ax.set_ylabel(r'$\Delta\tau_{infall}$',  fontsize=max(figsize)*2)

    if image_filename != None:
        plt.tight_layout()
        plt.savefig(image_filename + '_errors.png', dpi=360)

        print('Saved plotted errors to {}'.format(image_filename + '_errors.png'))
    
    plt.show()



def main_individual_r(r0):
    
    #Initial conditions
    U_r0 = 0
    U_phi0 = 0
    phi0 = 0
    
    geodesic = Geodesic(U_r0,U_phi0,r0,phi0)
    
    tau = 0
    step = 0.001
    
    print('Simulating...\n')
        
    #Integrate
    while geodesic.status == 1:
        tau += step
        
        geodesic.step(tau)
            
    #Analutical solution
    tau_analyt = calc_tau_infall(r0)
    
    print('Predicted infall time = {:.2f} \nAnalytical infall time = {:.2f} \nError = {:.2e}'.format(geodesic.integrator.t, tau_analyt, (geodesic.integrator.t-tau_analyt)/tau_analyt))



def main_multiple_rs(r0_max, image_filename=None):
    
    #Range of taus
    r0s = np.linspace(1 + 1e-3,r0_max,500)
    taus = np.zeros(len(r0s))
    step = 0.01
    
    #Initial conditions
    U_r0 = 0
    U_phi0 = 0
    phi0 = 0
    
    print('Simulating...\n')
    
    for i, r0 in tqdm(enumerate(r0s)):
        geodesic = Geodesic(U_r0,U_phi0,r0,phi0)
        
        tau = 0
        
        #Integrate
        while geodesic.status == 1:
            tau += step
            
            geodesic.step(tau)

        taus[i] = geodesic.integrator.t  #Integrator may take more steps hence take its current tau
        
    #Analutical solution
    tau_analyts = calc_tau_infall(r0s)
    
    #Error in nummerical solution
    err_taus = (taus - tau_analyts)/tau_analyts
    
    plot_tau_infall(r0s, taus, tau_analyts, err_taus, image_filename=image_filename)
    



if __name__ == "__main__":
    
    many_rs = int(input('Produce commparison plot for multiple radii? \nInput 1 for yes or 0 for no : '))
    
    
    ## Individual r0 ##
    
    if many_rs == 0:
        r0 = float(input('Provide individual initial radius : '))
        print('')
    
        main_individual_r(r0)
        
        
    ## Many r0s ##
        
    elif many_rs == 1:
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
    
        
        
        
        
        
    
