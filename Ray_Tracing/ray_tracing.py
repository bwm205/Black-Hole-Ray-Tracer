#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs ray tracing within the Schwarzchild metric to produce a jpg image of a black hole.
All 3D geodesics are rotated to a 2D plane for simulation.
Geodesics are simulated in parallel using python multiprocessing
Image is then shown and saved to the desired file

FILE LOCATION INPUTS
    background_filename : Path to background all sky image
    storage_filename : Path to store image. Should be stored in same location as background image

INITIAL CONDITIONS FILE FORMAT
    R : background radius
    projection : 2D projection for all-sky image.
        Choose either "hammer' or 'aitoff' 
    fov : Image field of view
    aspect : Image aspect ratio
    num_pix : Height of image in pixels
    cam_x, cam_y, cam_z : Camera viewpoint position
    alpha, beta : horisontal and vertical tilt to apply to camera
"""

import numpy as np
import multiprocessing as mp
import time
import cv2
import os

from functools import partial

from Library.classes import Geodesic, Background, Camera
from Library.coordinate_transforms import cart2pol, cart2pol_vector, pol2cart_vector, rotate_plane
from Library.user_input import select_file



def sim_ray(y0, v2, background):
    '''Simulates a null geodesic in the Schwarzschild metric until the background radius is reached, 
    returning the background pixel values at the final longditude and latitude. The path is simulated in 2D
    and the final point is rotated back to its original 3D plane with normal vector v2
    
    INPUT
        y0 : Initial conditions for 2D geodesic. Rotate to x-y plane before simulating
        v2 : Unit normal vector to ray's original 3D plane
        background : Object containing background image. 
            When using multiprocessing, fix this argument in a partial function to share between processes
            
    OUTPUT
        pix_val : Array of background RGB pixel values at final longditude and latitude'''
    
    U_r0,U_phi0, r0,phi0 = y0
    
    tau = 0.1  #Time to integrate up to
    
    geodesic = Geodesic(U_r0,U_phi0, r0,phi0, photon=1)
    
    stop_sim = 0     #Setting to 1 when conditions met stops simulation
    pix_val = np.zeros(3, dtype='int')   #Storing pixel RGB vals
    
    while stop_sim == 0:
        geodesic.step(tau)
        
        tau += 0.1
        
        if geodesic.status == 0:
            #Store black for BH pixel value
            pix_val = np.array([0,0,0])
            
            stop_sim = 1
            
            
        elif geodesic.r >= background.R:   #When asymptoting
            #Tilt plane to 3D
            x,y = pol2cart_vector(geodesic.U_r,geodesic.U_phi,geodesic.r,geodesic.phi)   #Take vector so no need to asymptote
            x,y,z = rotate_plane(np.array([x,y,0]),np.array([0,0,1]),v2)
            r,theta,phi = cart2pol(x,y,z)
            
            #Convert from theta/phi to long/lat
            long = phi - np.pi     
            lat = theta - np.pi/2
            
            #Store background pixel value
            pix_val = background.get_pix(long,lat)
            
            stop_sim = 1
         
    return pix_val    
    
    
    
def main(background_filename,R,projection, fov,aspect,num_pix, cam_x,cam_y,cam_z, alpha,beta, storage_filename):
        
    start_time = time.perf_counter()

    #Initialise background
    background = Background(background_filename, R, projection=projection, crop=1)
    print('Background downloaded and formatted')

    #Initialise camera
    camera = Camera(fov,aspect,num_pix, cam_x,cam_y,cam_z, alpha=alpha, beta=beta)

    #Initial condition
    y0 = np.zeros((len(camera.image_store),4))
    
    #Find planar unit normal vectors
    v1 = np.array([0,0,1])   #2D plane normal vector
    
    v2 = np.cross(camera.view_r,camera.pix_vectors.T) #Find normal vector through angular momentum
    v2 /= np.expand_dims(np.linalg.norm(v2,axis=-1),axis=-1)  #3D plane normal vector
    
    #Rotate planes
    U_x0,U_y0,U_z0 = rotate_plane(camera.pix_vectors,v2,v1)   
    x0_temp,y0_temp,z0_temp = rotate_plane(camera.view_r,v2,v1)
    print('Orbital planes rotated')
    
    # Convert to sphericals
    y0[:,2], y0[:,3] = cart2pol(x0_temp,y0_temp)
    y0[:,0], y0[:,1] = cart2pol_vector(U_x0,U_y0,x0_temp,y0_temp, velocity=1)
    print('Coordinates converted to sphericals \n')

    #Partial function shares background in multiprocessing
    partial_function = partial(sim_ray, background=background)

    #Use all but one core
    workers = mp.cpu_count() - 1
    print('Running using {} cores'.format(workers))

    #Run ray tracing
    with mp.Pool(workers) as p:
        pix_vals = np.array(list(p.starmap(partial_function, zip(y0,v2))))

    #Asign image to camera
    camera.image_store = pix_vals

    end_time = time.perf_counter()
    print('Simulation time = {:.2f} s \n'.format(end_time-start_time))
    
    #Save image
    cv2.imwrite(storage_filename, camera.image)
    print('Saved ray-traced image to : {}'.format(storage_filename))
    
    #Show image
    camera.show()



if __name__ == "__main__":
    ## User Input ##
    
    #Download image file
    print('Select all sky image')
    background_filename = select_file('image')
    print('Dowloaded all-sky image from : {} \n'.format(background_filename))

    #Download initial conditions
    print('Select initial conditions')
    conditions_filename = select_file('CSV')
    print('Dowloaded initial conditions from : {} '.format(conditions_filename))
    
    d_type = [('R', np.float16), ('fov', np.float16), ('num_pix', np.int32), ('aspect', np.float16), ('cam_x', np.float16), ('cam_y', np.float16), ('cam_z', np.float16), ('alpha', np.float16), ('beta', np.float16), ('projection', 'U6')]
    con = np.genfromtxt(conditions_filename, delimiter=',', dtype=d_type)
    
    #Provide filename. Saves image to same location as background image
    storage_filename = input('Provide filename for saving : ')
    storage_filename = os.path.abspath(os.path.dirname(background_filename)) + '/' + storage_filename + '.jpg'
    print('Ray-traced image will save to : {} \n'.format(storage_filename))
    
    
    ## Main Code ##
    
    #Run ray tracing
    main(background_filename,con['R'], str(con['projection']), con['fov']*np.pi/180., con['aspect'], con['num_pix'], con['cam_x'], con['cam_y'], con['cam_z'], con['alpha']*np.pi/180 ,con['beta']*np.pi/180, storage_filename)
    
    
    