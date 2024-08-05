#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script allows the user to input orbits determined by the relativistic and newtonian effective potentials.
The user produces an effective potential of given specific angular momentum j.
One may then either simulate a circular orbit, a perturbed cirbular orbit or input their chosen initial radius.
One should input the number of orbital periods, approximated as the keplerian orbital period.
The initial radius is shown on a plot of the effective potential and the geodesic is plotted

CONTAINS
    FUNCTION : plot_Veff
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os

from Library.plotting import plot_geodesic, plot_validation
from Library.physical_quantities import calc_T_kep, calc_Veff_extrema, calc_Veff_gr, calc_Veff_n
from Geodesic_Simulations.simulation import sim_geodesic



def plot_Veff(j, R_s=1, Veff_type='both' ,figsize=(6,6), vline=None, image_filename=None):
    '''Plots the relativistic and newtonian effective potentials for specific angular momentum j.
    User may mark on a vertical line vline to represent initial conditions
    
    INPUT
        j : Specific angular momentum
        R_s : Schwarzchild radius. Set to 1 as default
        Veff_type : Set to 'gr' for only relativistic plot, 'n' for only newtonian plot
            and 'both' for both. Set to 'both' as default
        figsize : Figure size. Default set to (6,6) 
        vline : r coordinate to plot vertical line. If bigger than current range, plot is rescaled
        image_filename : Filename to save plot. Ensure to inculde path and filetype.
            Default doesn't save plot
            '''
    
    r_uo_gr, r_so_gr, r_so_n = calc_Veff_extrema(j, R_s=1, Veff_type='both')
        
    
    #Keys for type of V_eff
    keys = {'gr':0, 'n':1, 'both':2}
    Veff_type = keys[Veff_type]
    
    #Range of r
    if Veff_type == 0 or Veff_type == 2:  #If GR iclude, it scales the plott
        max_r = r_so_gr*2
    
    elif Veff_type == 1:   #Else scale by newtonian
        max_r = r_so_n*3
        
    if vline != None and vline >= max_r:  #Extend plot if we have a vline furtther out
        max_r = vline* 1.1
    
    r = np.linspace(1e-10,max_r,5000)
        
    #Calculate potenials
    Veff_gr = calc_Veff_gr(r,j, R_s=R_s)
    Veff_n = calc_Veff_n(r,j, R_s=R_s)
    
    #Initialise figure
    fig = plt.figure(figsize=figsize)
    ax = fig.subplots()
    
    #Plot requested potentials
    if Veff_type == 0 or Veff_type == 2:
        ax.plot(r,Veff_gr, label='General Relativity', color='tab:blue')
        
    if Veff_type == 1 or Veff_type == 2:
        ax.plot(r,Veff_n, label='Newtonian', color='tab:orange')
        
    if vline != None:
        plt.axvline(vline, linestyle='--', color='tab:red')
        
    #For y scaling
    max_V, min_V = calc_Veff_gr(r_uo_gr,j), calc_Veff_gr(r_so_gr,j)
    y_range = max_V - min_V    
        
    #Configure plot   
    ax.set(xlim=(0,max_r), ylim=(min_V - y_range*0.1, max_V + y_range*0.1))
    ax.set_xlabel('r', fontsize=max(figsize)*2)
    ax.set_ylabel('$V_{eff}$', fontsize=max(figsize)*2)
    ax.legend(fontsize=max(figsize)*2)
    
    #Save image if requested
    if image_filename != None:
        plt.tight_layout()
        plt.savefig(image_filename, dpi=360)
        
        print('Saved plotted effective potential to {}'.format(image_filename))
        
    #Show image
    plt.show()
    
    

def main(j,r0,num_period, potential, validate, data_filename, image_filename):
    #For compatability with plotting function
    keys = {0:'gr', 1:'n'}
    potential = keys[potential] 
        
    #Plot effective potential with r0
    if image_filename == None:
        plot_Veff(j, Veff_type=potential, vline=r0, image_filename=None)
        
    elif image_filename != None:
        plot_Veff(j, Veff_type=potential, vline=r0, image_filename=image_filename + '_v_eff.png')
    
    
    tau_final = calc_T_kep(r0, R_s=1) * num_period #Estimate orbial period using Newtonian mechanics and calculate roughly 5 periods
    step = tau_final/100000
    
    #Initial conditions
    U_r0 = 0
    U_phi0 = j/r0**2 #Convert to angular velocity
    phi0 = 0
    
    start_time = time.perf_counter()
    
    print('Simulating... \n')
    sim_geodesic(tau_final, step, U_r0,U_phi0,r0,phi0, photon=0, R_s=1, validate=validate, data_filename=data_filename, n_save=200000)
    
    end_time = time.perf_counter() 
    print('Simulation time = {:.2f} s \n'.format(end_time-start_time))
    
    #Plot and save if requested
    if image_filename == None:
        print('Plotting...')
        
        plot_geodesic(data_filename=data_filename, image_filename=None)
    
        if validate == 1:
            plot_validation(data_filename=data_filename, image_filename=None)
            
    elif image_filename != None:
        plot_geodesic(data_filename=data_filename, image_filename=image_filename + '_geodesic.png')
    
        if validate == 1:
            plot_validation(data_filename=data_filename, image_filename=image_filename + '_validation.png')
    
    
    
if __name__ == "__main__":
    ## Initialise Effective Potential ##
    
    j = float(input('Provide specific angular momentum j : '))
    
    while j <= 3**.5:
        j = float(input('Value bellow j_min = 3**.5 . Please input again : '))
    print('')
    
    #Plot V_eff
    r_uo_gr, r_so_gr, r_so_n = calc_Veff_extrema(j, R_s=1, Veff_type='both')
    print('Unstable circular orbit radius (GR) = {:.2f} \nStable circular orbit radius (GR) = {:.2f} \nKeplerian circular orbit radius = {:.2f}' \
          .format(r_uo_gr, r_so_gr, r_so_n))
    
    plot_Veff(j)
    
    potential = int(input('Chose which effective potential to use. \nInput 0 for relativistic and 1 for newtonian : '))  
    
    ## Initialise Simulation ##
    
    circular = int(input('Stable circular orbit? \nInput 1 for yes and 0 for no : '))
    
    if circular == 1:
        
        if potential == 0:
            r0 = r_so_gr
            
        else:
            r0 = r_so_n
          
        #Option to perturb orbit
        perturb = int(input('Perturb circular orbit? \nInput 1 for yes and 0 for no : '))
        
        if perturb == 1:
            perturb_factor = float(input('Provide percentage by which to perturb r0 : '))
            r0 *= 1 + perturb_factor/100
            
        
    elif circular == 0:
        r0 = float(input('Input starting orbital radius : '))
        print(r0)
    
    #Length of simulation
    num_period = float(input('Input how many orbital periods (keplerian) to integrate : '))
    
    #Provide filename. Saves data to same location as script
    data_filename = input('Provide filename for saving data : ')
    data_filename = os.path.abspath(os.path.dirname(__file__)) + '/' + data_filename + '.csv'
    print('Geodesic data will be save to : {} '.format(data_filename))
    
    #Choose to validate
    validate = int(input('Validate data? \n1 = yes, 0 = no : '))
    
    ## Initialise Plotting ##
    
    #Choose whether to save plots
    plot = int(input('Save plots? \n1 = yes, 0 = no : '))
    
    if plot == 0:
        image_filename = None
        
    if plot == 1:
        image_filename = input('Provide filename for saving plot : ')
        image_filename = os.path.abspath(os.path.dirname(__file__)) + '/' + image_filename
        print('Plots will be saved to : {} '.format(image_filename))
        
    print('')


    ## Simulaion ##
    
    main(j,r0,num_period, potential, validate, data_filename, image_filename)
    
    
    
    


