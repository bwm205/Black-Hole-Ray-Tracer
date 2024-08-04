#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Holds all coordinate transform functions for use in project

CONTAINS
    Function : pol2cart
    Function : cart2pol
    Function : cart2pol_vector
    Function : pol2cart_vector
    Function : rotate
    Function : rotate_plane

"""

import numpy as np

def pol2cart(*coords):
    '''Converts from polar coordinates to cartesians
    
    INPUT
        *coords : Polar coordinates to be transformed.
            Input as r,phi for 2D and r,theta,phi for 3D
            
    OUTPUT
        coords: Transformed cartesian coordinates.
            Returns x,y for 2D and x,y,z for 3D'''
    
    dim = len(coords)
    
    #2D transform
    if dim == 2:
        r,phi = coords
        
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        
        return x, y
    
    #3D transform
    elif dim == 3:
        r,theta,phi = coords
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        return x, y, z



def cart2pol(*coords):
    '''Converts from cartesians to polar coordinates
    
    INPUT
        *coords : Cartesian coordinates to be transformed.
            Input as x,y for 2D and x,y,z for 3D
            
    OUTPUT
        coords: Transformed polar coordinates.
            Returns r,phi for 2D and r,theta,phi for 3D'''
    
    dim = len(coords)
    
    #2D transform
    if dim == 2:
        x,y = coords
        
        r = (x**2 + y**2)**.5
        phi = np.arctan2(y,x)
        
        return r, phi
    
    #3D transform
    elif dim == 3:
        x,y,z = coords
        
        r = (x**2 + y**2 + z**2)**.5
        theta = np.arccos(z/r)
        phi = np.arctan2(y,x)
        
        return r, theta, phi
    
    

def cart2pol_vector(*coords, velocity=0):
    '''Converts a cartesian vector to polar coordinates
    
    INPUT
        *coords : Vector components and position in cartesians. First enter vector and then position.
            Input as vx,vy,x,y for 2D and vx,vy,vz,x,y,z for 3D
        velocity : Set to 1 if vector is a velocity (ie : time derivative).
            Set to 0 as default
            
            
    OUTPUT
        components : Transformed vector components in polar coordinates.
            Returns vr,vphi for 2D and vr,vtheta,vphi for 3D'''
    
    dim = int(len(coords)/2)
    
    #2D transform
    if dim == 2:
        vx,vy,x,y = coords
        
        #Calculating spherical coords
        r, phi = cart2pol(x,y)
        
        N = phi.shape
        
        #Producing transformation matrix
        M = np.zeros((2,2,*N))
        
        s = np.sin(phi)    #For speed
        c = np.cos(phi)
        
        M[0,0] = M[1,1] = c
        M[1,0] = -s
        M[0,1] = s
        
        #Transform coordinates
        if len(N) == 0:
            vr, vphi = M @ np.array([vx,vy])

        
        elif len(N) == 1:
            vr, vphi = (M.transpose(2,0,1) @ np.expand_dims(np.array([vx,vy]).T,axis=-1)).squeeze().T
            
        #Since velocity is the time derivative of a vector   
        if velocity == 0:
            return vr, vphi
        
        elif velocity == 1:
            return vr, vphi/r
    
      
    #3D transform
    elif dim == 3:
        vx,vy,vz,x,y,z = coords
        
        #Calculating spherical coords
        r, theta, phi = cart2pol(x,y,z)
        
        N = phi.shape
        
        #Producing transformation matrix
        M = np.zeros((3,3,*N))
        
        sp = np.sin(phi)    #For speed
        cp = np.cos(phi)
        st = np.sin(theta)
        ct = np.cos(theta)
        
        M[0,0] = st*cp
        M[1,0] = ct*cp
        M[2,0] = -sp
        
        M[0,1] = st*sp
        M[1,1] = ct*sp
        M[2,1] = cp
        
        M[0,2] = ct
        M[1,2] = -st

        #Transform coordinates
        if len(N) == 0:
            vr, vtheta, vphi = M @ np.array([vx,vy,vz])
        
        elif len(N) == 1:
            vr, vtheta, vphi = (M.transpose(2,0,1) @ np.expand_dims(np.array([vx,vy,vz]).T,axis=-1)).squeeze().T
        
        #Since velocity is the time derivative of a vector 
        if velocity == 0:
            return vr, vtheta, vphi
        
        elif velocity == 1:
            return vr, vtheta/r, vphi/(st*r)
        
    
    
    
def pol2cart_vector(*coords, velocity=0):
    '''Converts a polar vector to cartesian coordinates
    
    INPUT
        *coords : Vector components and position in polar coordinates. First enter vector and then position.
            Input as vr,vphi,r,phi for 2D and vr,vtheta,vphi,r,theta,phi for 3D
        velocity : Set to 1 if vector is a velocity (ie : time derivative).
            Set to 0 as default
            
            
    OUTPUT
        components : Transformed vector components in cartesian coordinates.
            Returns vx,vy for 2D and vx,vy,vz for 3D'''
    
    dim = int(len(coords)/2)
    
    #2D transform
    if dim == 2:
        vr,vphi,r,phi = coords
        
        #Since velocity is the time derivative of a vector 
        if velocity == 1:
            vphi *= r
        
        N = phi.shape
        
        #Producing transformation matrix
        M = np.zeros((2,2,*N))
        
        s = np.sin(phi)    #For speed
        c = np.cos(phi)
        
        M[0,0] = M[1,1] = c
        M[1,0] = s
        M[0,1] = -s
        
        #Transform coordinates
        if len(N) == 0:
            vx, vy = M @ np.array([vr,vphi])
            
            return vx, vy
        
        elif len(N) == 1:
            vx, vy = (M.transpose(2,0,1) @ np.expand_dims(np.array([vr, vphi]).T,axis=-1)).squeeze().T
        
            return vx, vy
        
      
    #3D transform
    elif dim == 3:
        vr,vtheta,vphi,r,theta,phi = coords
        
        #Since velocity is the time derivative of a vector 
        if velocity == 1:
            vtheta *= r
            vphi *= np.sin(theta)*r
        
        N = phi.shape
        
        #Producing transformation matrix
        M = np.zeros((3,3,*N))
        
        sp = np.sin(phi)    #For speed
        cp = np.cos(phi)
        st = np.sin(theta)
        ct = np.cos(theta)
        
        M[0,0] = st*cp
        M[1,0] = st*sp
        M[2,0] = ct
        
        M[0,1] = ct*cp
        M[1,1] = ct*sp
        M[2,1] = -st
        
        M[0,2] = -sp
        M[1,2] = cp

        #Transform coordinates
        if len(N) == 0:
            vx, vy, vz = M @ np.array([vr,vtheta,vphi])
        
            return vx, vy, vz
        
        elif len(N) == 1:
            vx, vy, vz = (M.transpose(2,0,1) @ np.expand_dims(np.array([vr, vtheta, vphi]).T,axis=-1)).squeeze().T
        
            return vx, vy, vz
        
    

def rotate(r, theta, axis):
        '''Rotates a vector r by an angle theta about a given axis
        
        INPUT
            r : Vector to be rotated in cartesians
            theta : Angle by which to rotate vector in radians
            axis : Coordinate axis around which to rotate. Input 0 for x, 1 for y and 2 for z
            
        OUTPUT
            r : Rotated vector  in cartesians'''

        M = np.zeros((3,3))
        
        s = np.sin(theta)
        c = np.cos(theta)

        #Calculate rotation matrix
        if axis == 0:    #about x-axis
            M[0,0] = 1
            M[1,1] = c
            M[1,2] = -s
            M[2,1] = s
            M[2,2] = c

        elif axis == 1:  #about y-axis
            M[0,0] = c
            M[0,2] = s
            M[1,1] = 1
            M[2,0] = -s
            M[2,2] = c

        elif axis == 2:  #about z-axis
            M[0,0] = c
            M[0,1] = -s
            M[1,0] = s
            M[1,1] = c
            M[2,2] = 1   
          
        #Apply rotation
        r = M @ r

        return r        
        
    

def rotate_plane(r,v1,v2):
    '''Rotates points r on a plane with normal vector v1 onto a plane with normal vector v2. 
    Ensure normal vectors are normalised
    
    INPUT
        r : Vector(s) to rotate
        v1 : Unit normal vector for plane 1 containing points r
        v2 : Unit normal vector for plane 2 to rotate r into
        
    OUTPUT
        r : Rotated vectors in plane 2'''

    #Compute axis to rotate around by angle theta
    theta = np.arccos(np.dot(v1,v2))
    axis = np.cross(v1,v2)
    axis /= np.expand_dims(np.linalg.norm(axis,axis=-1),axis=-1)

    x,y,z = axis.T  
    N = x.shape     #Number of points to rotate
    
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1-c
    
    #Initialise rotation matrix
    Q = np.zeros((3,3,*N))
    
    Q[0,0] = x**2*C + c
    Q[0,1] = x*y*C - z*s
    Q[0,2] = x*z*C + y*s
    
    Q[1,0] = y*x*C + z*s
    Q[1,1] = y**2*C + c
    Q[1,2] = y*z*C - x*s
    
    Q[2,0] = z*x*C - y*s
    Q[2,1] = z*y*C + x*s
    Q[2,2] = z**2*C + c
    
    #Compute matrix multiplication
    if len(N) == 0: 
        r = Q @ r
        
        return r
        
    elif len(N) ==1:  #Many points requires rearangement
        r = Q.transpose(2,0,1) @ np.expand_dims(r.T,axis=-1)
    
        return r.squeeze().T
        
    
    