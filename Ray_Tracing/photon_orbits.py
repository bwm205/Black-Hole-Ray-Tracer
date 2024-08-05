#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script produces plot of photon orbits for a given image size
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
from matplotlib.patches import Circle

from Library.classes import Geodesic
from Library.coordinate_transforms import pol2cart, cart2pol, cart2pol_vector

if __name__ == "__main__":
    
    #User input plot sze
    plot_width = float(input('Input plot width : '))
    max_r = 2**.419*plot_width
    
    #Choose whether to save plots
    plot = int(input('Save plots? \n1 = yes, 0 = no : '))
        
    if plot == 1:
        image_filename = input('Provide filename for saving plot : ')
        image_filename = os.path.abspath(os.path.dirname(__file__)) + '/' + image_filename + '.png'
        print('Plots will be saved to : {} '.format(image_filename))
        
    print('')
    
    #Initial Conditions
    x0 = -plot_width
    y0s = np.linspace(0,max_r,30)
    
    U_x0 = 1.
    U_y0 = 0.
    
    r0s, phi0s = cart2pol(x0,y0s)
    U_r0s, U_phi0s = cart2pol_vector(U_x0,U_y0, x0,y0s, velocity=1)
    
    #Lists for holding variables
    r_histories = []
    phi_histories = []
    
    #Simualate
    for U_r0, U_phi0, r0, phi0 in tqdm(zip(U_r0s, U_phi0s, r0s, phi0s)):
        geodesic = Geodesic(U_r0,U_phi0,r0,phi0, photon=0)
        
        tau = 0
        step = .1
        
        #Integrate
        while geodesic.r <= max_r*5:
            tau += step
            
            geodesic.step(tau)
            
            #If crossed event horison, exit system
            if geodesic.status != 1:
                break
        
        #Store path
        r_histories.append(geodesic.r_history)
        phi_histories.append(geodesic.phi_history)
            
            
    #Plotting
    figsize=(10,10)
    fig = plt.figure(figsize=figsize)
    ax = fig.subplots()
    
    for r,phi in tqdm(zip(r_histories,phi_histories)):
        x,y = pol2cart(r,phi)
        
        ax.plot(x,y, color='goldenrod')
        
    #Configure plot
    ax.set(xlim=(-plot_width,plot_width),ylim=(-plot_width,plot_width))
    ax.set_xlabel('x', fontsize=max(figsize)*2)
    ax.set_ylabel('y', fontsize=max(figsize)*2)
        
    #For BH circle
    center = (0,0)
    radius = 1
    circle = Circle(center, radius, color='black')
    ax.add_patch(circle)
    
    #For photon sphere
    center = (0,0)
    radius = 1.5
    circle = Circle(center, radius, facecolor='none', edgecolor='tab:red', linewidth=2, zorder=40)
    ax.add_patch(circle)
    
    #If requested, save figure
    if plot == 1:
        plt.tight_layout()
        plt.savefig(image_filename, dpi=360)

        print('Saved plotted validation to {}'.format(image_filename))