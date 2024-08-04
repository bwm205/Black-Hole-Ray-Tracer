#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Holds plotting functions for use in project

CONTAINS
    FUNCTION : plot_geodesic
    FUNCTION : plot_validation

"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle

from Library.coordinate_transforms import pol2cart, pol2cart_vector, rotate_plane


def plot_geodesic(data_filename=None, geodesic=None, min_point=0, max_point=None, R_s=1, figsize=(10,10), image_filename=None):
    '''Produces 2D matplotlib plot of geodesic. Either provide data as a file or directly input geodesic class containing history.
    3D geodesics are rotated to 2D plane using their initial angular velocity
    
    INPUT
        data_filename : Filename containing geodesic data. Ensure to inculde path and filetype
        geodesic : Geodesic class containing simulated data within history
        min_point & max_point : Range of datapoints to plot. Default plots all data
        R_s : Black hole Schwarzchild radius. Default = 1.
        figsize : Figure size. Default set to (10,10)
        image_filename : Filename to save plot. Ensure to inculde path and filetype.
            Default doesn't save plot
    '''
    
    #If filename given, download from file
    if data_filename != None:
        #Download data
        data = np.genfromtxt(data_filename, delimiter=',', names=True)
        
        #If max_ind given, plot all data
        if max_point == None:
            max_point = len(data)
        
        r, phi = data['r'][min_point:max_point], data['phi'][min_point:max_point]
        
        rmax= 1.1*np.max(data['r'][min_point:max_point])  #For use in plot limits
        
        #Test if 2D or 3D
        try:
            theta = data['theta'][min_point:max_point]
            x,y,z = pol2cart(r,theta,phi)  #Convert to cartesians
            
            U_r0,U_theta0,U_phi0 = data['U_r'][0], data['U_theta'][0], data['U_phi'][0] #For use in rotation
            
            dim = 3
        
        except: 
            x,y = pol2cart(r,phi)  #Convert to cartesians
            
            dim = 2
            
            
        
    #If geodesic given, use history data
    elif geodesic != None:
        #If max_ind given, plot all data
        if max_point == None:
            max_point = len(geodesic.history)
            
        r, phi = geodesic.r_history[min_point:max_point], geodesic.phi_history[min_point:max_point]
        
        rmax= 1.1*np.max(geodesic.r_history[min_point:max_point])  #For use in plot limits
        
        #Test if 2D or 3D
        try:
            theta = geodesic.theta_history[min_point:max_point]
            x,y,z = pol2cart(U_r0,U_theta0,U_phi0, r,theta,phi)  #Convert to cartesians
            
            U_r0,U_theta0,U_phi0 = geodesic.U_r_history[0],geodesic.U_theta_history[0],geodesic.U_phi_history[0] #For use in rotation
            
            dim = 3
        
        except:
            x,y = pol2cart(r,phi)  #Convert to cartesians
            
            dim = 2    
        
    #Else exit function
    else:
        print('No data given. Please provide either filename or geodesic object.')
        return None
    
    
    #If 3D geodesic, rotate to 2D plane 
    if dim == 3:
        #Find planar unit normal vectors
        v2 = np.array([0,0,1])   #2D plane normal vector
        
        r0 = np.array([x[0],y[0],z[0]])
        U0 = pol2cart_vector(U_r0,U_theta0,U_phi0, x[0],y[0],z[0])
        v1 = np.cross(r0,U0) #Find normal vector through angular velocity
        v1 /= np.linalg.norm(v1,axis=-1) #3D plane normal vector
        
        #Rotate to 2D
        x,y,z = rotate_plane(np.array([x,y,z]),v1,v2) 
        

    #Plot
    fig = plt.figure(figsize=figsize)
    ax = fig.subplots()
    ax.plot(x,y)
    
    #For BH circle
    center = (0,0)
    radius = R_s
    circle = Circle(center, radius, color='black')
    ax.add_patch(circle)
    
    #Configure plot
    ax.set(xlim=(-rmax,rmax),ylim=(-rmax,rmax))
    ax.set_xlabel('x', fontsize=max(figsize)*2)
    ax.set_ylabel('y', fontsize=max(figsize)*2)
    
    #Save image if requested
    if image_filename != None:
        plt.tight_layout()
        plt.savefig(image_filename, dpi=360)
        
        print('Saved plotted geodesic to {} \n'.format(image_filename))

    #Show image
    plt.show()
    
    
    
def plot_validation(data_filename=None, geodesic=None, taus=None, min_point=0, max_point=None, figsize=(10,10), image_filename=None):
    '''Plots the validation variables V,E & j for a given geodesic. 
    Either provide data as a file or directly input geodesic class containing history.
    
    INPUT
        data_filename : Filename containing geodesic data. Ensure to inculde path and filetype
        geodesic : Geodesic class containing simulated data within history
        taus : Values of affine parameter integrated. Only input if using geodesic class
        min_point & max_point : Range of datapoints to plot. Default plots all data
        R_s : Black hole Schwarzchild radius. Default = 1.
        figsize : Figure size. Default set to (10,10)
        image_filename : Filename to save plot. Ensure to inculde path and filetype.
            Default doesn't save plot'''
    
    
    #If filename given, download from file
    if data_filename != None:
        #Download data
        data = np.genfromtxt(data_filename, delimiter=',', names=True)
        
        #If max_ind given, plot all data
        if max_point == None:
            max_point = len(data)
        
        variables = [data['V'][min_point:max_point], data['E'][min_point:max_point], data['j'][min_point:max_point]]
        taus = data['tau'][min_point:max_point]
    
    #If geodesic given, use history data
    elif geodesic != None and taus.any() != None:
        #If max_ind given, plot all data
        if max_point == None:
            max_point = len(geodesic.history)
            
        variables = [geodesic.V_history[min_point:max_point], geodesic.E_history[min_point:max_point], geodesic.j_history[min_point:max_point]]
        
        
    #Else exit function
    else:
        print('No data given. Please provide either filename or geodesic object.')
        return None

    #Plot
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=.1)
    
    names = {0:'V', 1:'E', 2:'j'} #For labelling
    
    for i,ax in enumerate(axes):
        var = variables[i] #Extract variable
        
        ax.plot(taus,var)
        
        #For use in scaling
        y_max = np.max(var)
        y_min = np.min(var)
        y_range = y_max - y_min

        #Configure plot
        ax.set_xlim(0,np.max(taus))
        ax.set_ylim(y_min-y_range*.1,y_max+y_range*.1)
        ax.set_ylabel(names[i], fontsize=max(figsize)*1.5)

    axes[-1].set_xlabel('tau', fontsize=figsize[0]*1.5) #Add tau label at bottom
        
    #Save image if requested
    if image_filename != None:
        plt.tight_layout()
        plt.savefig(image_filename, dpi=360)

        print('Saved plotted validation to {} \n'.format(image_filename))

    #Show image
    plt.show()  
        
        