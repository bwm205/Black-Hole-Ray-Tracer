#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Holds function sim_geodesic which simulates a geodesic within the Schwarzchild metric with given initial conditions
It either outputs the Geodesic class containing coordinate's history (use for shorter geodesics)
    or it allows the user to save data periodically to a specified file (use for longer geodesics where memory becomes a constraint)
Function may be imported to be run in seperate routine or simply run the script as the main function, 
    providing initial conditions for 2D geodesic

CONTAINS
    FUNCTION : init_file
    FUNCTION : save
    FUNCTION : sim_geodesic
"""

import numpy as np
import time
import os

from Library.classes import Geodesic
from Library.user_input import select_file
from Library.plotting import plot_geodesic, plot_validation

def init_file(filename, geodesic, validate=0):
    '''Initialise file to save geodesic data
    
    INPUT
        filename : Name of file. Ensure to include path and file type
        geodesic : Initialised Geodesic object
        validate : Set to 1 to add validation variables. Set to 0 as default'''
    
    #Produce file header
    header = 'tau,' + ','.join(geodesic.track_names)

    if validate == 1:
        header += ',V,E,j'
    
    #Initialise file
    np.savetxt(filename, np.array([]), delimiter=',', header=header)
    
    

def save(filename, geodesic, taus, validate=0):
    '''Saves geodesic data for a range of times 'taus' to a given file
    
    INPUT
        filename : Name of file. Ensure to include path and file type
        geodesic : Geodesic object containing data
        taus : Tau values for data contained in geodesic
        validate : Set to 1 to add validation variables. Set to 0 as default'''
    
    #Configure data
    saved_data = geodesic.dump(validate=validate)
    saved_data = np.hstack((taus,saved_data))
    
    #Append to given file
    with open(filename, 'a') as f:
        np.savetxt(f, saved_data, delimiter=',')
        


def sim_geodesic(tau_final,step, *coords, photon=0, R_s=1, validate=0, data_filename=None, n_save=None, track_names=None):
    '''Simulates a geodesic in the Schwarzchild metric for a given set of initial conditions up to a 
    specified tau with a given step. The user may choose to save the final data or output directly
    along with the option of specifying which variables to track. 
        
    INPUT
        tau_final : Final tau to integrate up to
        step : Step size. Note, the integrator uses an adapdable step size and so this only gives the 
            output step size. There were likely intemediate steps.
        *coords : Initial conditions for the velocity and position.
                Input as U_r,U_phi, r,phi for 3D and U_r,U_theta,U_phi, r,theta,phi for 4D.
        photon : Set to 1 for a  null geodesic. 0 as default
        R_s : Black hole Schwartzchild radius. Set to 1 as default
        validate : Set to 1 to track validation variables. Set to 0 as default.
        data_filename : Filename to save output data. Ensure to inculde path and filetype.
            Default outputs data directly
        n_save : Number of steps after which to save data and wipe history. 
            Default integrates entire path at once
        track_names : List of variable names to track. Default tracks all variables
        
    OUTPUT
        Returns Geodesic when filename not specified. If given, saves data to file
    '''

    #Initialise Geodesic
    geodesic = Geodesic(*coords, photon=photon)
    
    #If track names are given
    if track_names != None:
        geodesic.track(track_names)
        
        
    #When looping over many short geodesics, helpful to return the class
    if data_filename == None:
        taus = np.arange(0,tau_final,step)
        
        #Integrate
        for i,tau in enumerate(taus[1:]):
            geodesic.step(tau)
             
            #Check for integrator errors
            if geodesic.status != 1:
                if geodesic.status == 0:
                    print('Integration stopped : trajectory crossed event horison')
                
                elif geodesic.status == 2:
                    print('Integration stopped : integration failed at theta=0')
                
                break
                
        return geodesic
    
    
    #For longer geodesics, save to file
    elif data_filename != None:
        i_old = 0
        init_file(data_filename, geodesic, validate=validate)
    
        #Timepoints to integrate over
        taus = np.arange(0,tau_final,step)
        i_final = len(taus)
        
        #Integrate
        for i,tau in enumerate(taus[1:]):
            geodesic.step(tau)
            
            #Save to file after n_save
            if (i+1) % n_save == 0:
                save(data_filename, geodesic, taus[i_old:i+2,None], validate=validate)   #None expands dimensions
                print('Saved data to {}'.format(data_filename))
                i_old = i+2
              
            #Check for integrator errors
            if geodesic.status != 1:
                if geodesic.status == 0:
                    print('Integration stopped : trajectory crossed event horison')
                
                elif geodesic.status == 2:
                    print('Integration stopped : integration failed at theta=0')
            
                i_final = i+2
            
                break
            
        save(data_filename, geodesic, taus[i_old:i_final,None], validate=validate)
        print('Saved data to {} \n'.format(data_filename))
        
        return None
    
    

if __name__ == "__main__":
    ## User Input ##
    
    #Download initial conditions
    print('Select initial conditions')
    conditions_filename = select_file('CSV')
    print('Dowloaded initial conditions from : {} '.format(conditions_filename))
    
    #Provide filename. Saves data to same location as initial conditions
    data_filename = input('Provide filename for saving data : ')
    data_filename = os.path.abspath(os.path.dirname(conditions_filename)) + '/' + data_filename + '.csv'
    print('Geodesic data will be save to : {} '.format(data_filename))
    
    d_type = [('U_r0', np.float16), ('U_phi0', np.float16), ('r0', np.float16), ('phi0', np.float16), ('tau_final', np.float16), ('step', np.float16), ('photon', np.int32), ('R_s', np.float16)]
    con = np.genfromtxt(conditions_filename, delimiter=',', dtype=d_type)
    
    #Choose to validate
    validate = int(input('Validate data? \n1 = yes, 0 = no : '))
    
    #Choose to plot
    plot = int(input('Save plot? \n1 = yes, 0 = no : '))
    
    if plot == 1:
        image_filename = input('Provide filename for saving plot : ')
        image_filename = os.path.abspath(os.path.dirname(conditions_filename)) + '/' + image_filename
        print('Plotted geodesic will be saved to : {} '.format(image_filename))
        
    
    ## Main Code ##
    
    #Simulate
    start_time = time.perf_counter()
    
    print('Simulating... \n')
    sim_geodesic(con['tau_final'], con['step'], con['U_r0'],con['U_phi0'],con['r0'],con['phi0'], photon=con['photon'], R_s=con['R_s'], validate=validate, data_filename=data_filename, n_save=100000, track_names=None)
    
    end_time = time.perf_counter() 
    print('Simulation time = {:.2f} s \n'.format(end_time-start_time))
    
    #Plot
    if plot == 0:
        plot_geodesic(data_filename=data_filename, image_filename=None)
            
        if validate == 1:
            plot_validation(data_filename=data_filename, image_filename=None)
    
    #If requested, saveplot
    if plot == 1:
        if validate == 0:
            plot_geodesic(data_filename=data_filename, image_filename=image_filename + '.png')
            
        elif validate == 1:
            plot_geodesic(data_filename=data_filename, image_filename=image_filename + '_geodesic.png')
            plot_validation(data_filename=data_filename, image_filename=image_filename + '_validation.png')
        
        
