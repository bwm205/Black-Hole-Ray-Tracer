#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Holds classes for use in project

CONTAINS
    CLASS : Geodesic
    CLASS : Background
    CLASS : Camera
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.integrate import ode
from PyAstronomy import pyasl

from Library.coordinate_transforms import cart2pol, rotate


class Geodesic():
    '''The Geodesic class allows one to integrate the path of massive particles or photons along a geodesic within the Schwarzschild metric.
    
    Initialisation:
        The user should specify the initial velocity and position in sphericals. 
        The initialisation will calculate the time velocity. Initial time is set to 0.
        
    Integration:
        By using self.step(tau), the user performs a step of integration up to proper time tau. 
        When successful, the self.status = 1. 
        If failed : status = 0 when crossed the event horison. Status = 2 when integrator failed at theta=0 singularity (only problem in 4D).
        A 'dopri5' fifth order Runge-Kutta integrator with an adaptive timestep is used for 3D.
        A 'lsoda' integrator based on backward differentiation formulas is used for 4D (since stiff problem).   
        
    Tracking Data
        Data temporarily stored in self.history. For long paths, this should be dumped frequently to avoid clogging memory and saved to a file.
        Initialise track using self.track() with chosen variable names. self.dump() will dump all variables if a track not initialised.
        Properties can return either the current state of variables as self.variableName or their history as self.variableName_history. 
        If history wiped, no data returnable.
        '''
    
    
    def __init__(self, *coords, photon=0, R_s=1):
        '''Initialises Geodesic
        
        INPUT
            *coords : Initial conditions for the velocity and position.
                Input as U_r,U_phi, r,phi for 3D and U_r,U_theta,U_phi, r,theta,phi for 4D.
            photon : Set to 1 for a  null geodesic. 0 as default
            R_s : Black hole Schwartzchild radius. Set to 1 as default'''
            
        super().__init__()
        
        self.R_s = R_s   #BH radius
        self.status = 1  # Outside R_s, status = 1. Inside R_s, status = 0. At theta=0, status = 2.
        self.dim = int((len(coords))/2) + 1   #Dimension
        
        #Initialisation depends on dimension
        if self.dim == 3:
            U_r,U_phi, r,phi = coords
            t=0.
        
            #Initialise U_t
            if photon == 0:
                U_t = ((1 + U_r**2/(1-R_s/r) + (r*U_phi)**2)/(1-R_s/r))**0.5
                
            elif photon == 1:
                U_t = ((U_r**2/(1-R_s/r) + (r*U_phi)**2)/(1-R_s/r))**0.5
            
            #Compile initial conditions
            self.y = np.array([U_t,U_r,U_phi, t,r,phi])
            
            #Set up integrator
            self.integrator = ode(self.ODEs_3D)
            self.integrator.set_integrator('dopri5')
        
        elif self.dim == 4:
            U_r,U_theta,U_phi, r,theta,phi = coords
            t=0.
            
            #Initialise U_t
            if photon == 0:
                U_t = ((1 + U_r**2/(1-R_s/r) + (r*U_theta)**2 + (r*np.sin(theta)*U_phi)**2)/(1-R_s/r))**0.5
            
            elif photon == 1:
                U_t = ((U_r**2/(1-R_s/r) + (r*U_theta)**2 + (r*np.sin(theta)*U_phi)**2)/(1-R_s/r))**0.5
                
            #Compile initial conditions
            self.y = np.array([U_t,U_r,U_phi, t,r,phi, U_theta,theta])
            
            #Set up integrator
            self.integrator = ode(self.ODEs_4D)
            self.integrator.set_integrator('lsoda')    #Since problem stiff at theta=0
            print('WARNING : 4D ODEs are stiff equations. lsoda integrator has no adaptive step.\n Consider rotating orbit to 2D plane or lower step size for higher accuracy.')
        
        #Set initial conditions
        self.integrator.set_initial_value(self.y)
        
        #Tracking
        self._history = [self.y]   #Track variable history
        self.keys = {'U_t':0,'U_r':1,'U_phi':2, 't':3,'r':4,'phi':5, 'U_theta':6,'theta':7}   #Keys for tracking indicies
        
        #If no track declared, track all variables
        if self.dim == 3:
            self.track_names = ['U_t','U_r','U_phi', 't','r','phi']
            self.track_inds = [0,1,2,3,4,5]
            
        elif self.dim == 4:
            self.track_names = ['U_t','U_r','U_phi', 't','r','phi', 'U_theta','theta']
            self.track_inds = [0,1,2,3,4,5,6,7]
    
    
    ## History Properties ##
    
    @property
    def history(self):
        '''Returns history as a numpy array, not a list'''
        return np.array(self._history)
        
    
    ## Validation Properties ##
    
    @property
    def V(self):
        '''Gives the magnitude of the particle four velocity through space-time, along its worldline.
        Should be 1 for a massive particle and 0 for a photon.'''
        y = self.y
        
        V = self.calc_V(y)
        
        return V
        
    @property
    def V_history(self):
        '''Gives the history of the magnitude of the particle four velocity through space-time, along its worldline.
        Should be 1 for a massive particle and 0 for a photon.'''
        history = self.history.T
        
        V = self.calc_V(history)
             
        return V
    
    def calc_V(self, coords):
        '''Calculates the magnitude of the particle four velocity through space-time, along its worldline.
        Should be 1 for a massive particle and 0 for a photon.
        
        INPUT
            coords : Array containing four velocity and position.
                Input as U_t,U_r,U_phi,t,r,phi for 3D and U_t,U_r,U_phi,t,r,phi,U_theta,theta for 4D.
            
        OUTPUT
            V : The magnitude of the four velocity'''
        
        if self.dim == 3:
            U_t,U_r,U_phi, t,r,phi = coords
            
            V = (1-self.R_s/r)*U_t**2 - U_r**2/(1-self.R_s/r) - (r*U_phi)**2
            
        elif self.dim == 4:
            U_t,U_r,U_phi, t,r,phi, U_theta,theta = coords
            
            V = (1-self.R_s/r)*U_t**2 - U_r**2/(1-self.R_s/r) - (r*U_theta)**2 - (r*np.sin(theta)*U_phi)**2
             
        return V
    
    
    @property
    def E(self):
        '''Gives energy for use in validation. Should be a conserved quantity'''
        
        E = (1-self.R_s/self.r) * self.U_t
        
        return E
        
    @property
    def E_history(self):
        '''Gives energy for use in validation. Should be a conserved quantity'''
        
        E = (1-self.R_s/self.r_history) * self.U_t_history 
        
        return E
    
    
    @property
    def j(self):
        '''Gives the specific angular momentum of the orbit. Should be a conserved quantity'''
        
        if self.dim == 3:
            j = self.r**2 * self.U_phi
            
        elif self.dim == 4:
            j = self.r**2 * np.sin(self.theta) * self.U_phi
            
        return j
    
    @property
    def j_history(self):
        '''Gives the history of the specific angular momentum of the orbit. Should be a conserved quantity'''
        
        if self.dim == 3:
            j = self.r_history**2 * self.U_phi_history
            
        elif self.dim == 4:
            j = self.r_history**2 * np.sin(self.theta_history) * self.U_phi_history
            
        return j
        
        
    ## Current Coordinates ##
    
    @property
    def U_t(self):
        '''Time coordinate of the four velocity'''
        return self.y[0]
    
    @property
    def U_r(self):
        '''Radial coordinate of the four velocity'''
        return self.y[1]
    
    @property
    def U_phi(self):
        '''Azimuthal coordinate of the four velocity'''
        return self.y[2]
    
    @property
    def t(self):
        '''Time coordinate in spacetime'''
        return self.y[3]
    
    @property
    def r(self):
        '''Radial coordinate in spacetime'''
        return self.y[4]
    
    @property
    def phi(self):
        '''Azimuthal coordinate in spacetime'''
        return self.y[5]
    
    @property
    def U_theta(self):
        '''Polar coordinate of the four velocity'''
        return self.y[6]
    
    @property
    def theta(self):
        '''Polar coordinate in spacetime'''
        return self.y[7]
    
    
    ## History of Variables ##
    
    @property
    def U_t_history(self):
        '''Time coordinate of the four velocity history'''
        return self.history[:,0]
    
    @property
    def U_r_history(self):
        '''Radial coordinate of the four velocity history'''
        return self.history[:,1]
    
    @property
    def U_phi_history(self):
        '''Azimuthal coordinate of the four velocity history'''
        return self.history[:,2]
    
    @property
    def t_history(self):
        '''Time coordinate in spacetime history'''
        return self.history[:,3]
    
    @property
    def r_history(self):
        '''Radial coordinate in spacetime history'''
        return self.history[:,4]
    
    @property
    def phi_history(self):
        '''Azimuthal coordinate in spacetime history'''
        return self.history[:,5]
    
    @property
    def U_theta_history(self):
        '''Polar coordinate of the four velocity history'''
        return self.history[:,6]
    
    @property
    def theta_history(self):
        '''Polar coordinate in spacetime history'''
        return self.history[:,7]
      
    
    ## Integration Functions ##
    
    def ODEs_3D(self,tau, y):
        '''Differential equations for a 3D geodesic in the Schwartzschild metric
        
        INPUT
            tau : Integration parameter
            y : Numpy vector for integrator current state. Contains four velocity and position.
            
        OUTPUT:
            Vector countaining calculated derivatives for use in integration'''
        
        U_t,U_r,U_phi, t,r,phi = y

        w = 1-self.R_s/r  #Speeds up calculation
        
        #Calculate ODEs
        dU_t = -self.R_s/(r**2*w) * U_r*U_t
        dU_r = w*(r*U_phi**2-self.R_s/2*(U_t/r)**2) + self.R_s*U_r**2/(2*r**2*w)
        dU_phi = -2*U_r*U_phi/r

        #Return as array
        return np.array([dU_t,dU_r,dU_phi, U_t,U_r,U_phi])
    
    
    def ODEs_4D(self,tau, y):
        '''Differential equations for a 4D geodesic in the Schwartzschild metric
        
        INPUT
            tau : Integration parameter
            y : Numpy vector for integrator current state. Contains four velocity and position.
            
        OUTPUT:
            Vector countaining calculated derivatives for use in integration'''
        
        U_t,U_r,U_phi, t,r,phi, U_theta,theta = y

        w = 1-self.R_s/r   #Speeds up calculation

        #Calculate ODEs
        dU_t = -self.R_s/(r**2*w) * U_r*U_t
        dU_r = w*(r*U_theta**2 + r*(np.sin(theta)*U_phi)**2 - self.R_s/2*(U_t/r)**2) + self.R_s*U_r**2/(2*r**2*w)
        dU_theta = np.sin(theta)*np.cos(theta)*U_phi**2 - 2*U_r*U_theta/r
        dU_phi = -2*np.cos(theta)/np.sin(theta)*U_theta*U_phi - 2*U_r*U_phi/r

        #Return as array
        return np.array([dU_t,dU_r,dU_phi, U_t,U_r,U_phi, dU_theta,U_theta])
    
    def step(self, tau):
        '''Performs a step of integration up to affine parameter tau. 
        Integrators with adaptable timestep may have steps between taus - these are not returned.
        
        INPUT
            tau : Integration parameter'''
        
        #Integrate
        self.integrator.integrate(tau)
        
        #Check if integrator fails
        if self.integrator.successful() == False:
            
            if self.integrator.y[4]-self.R_s < 1e-8:  #Check if inside R_s (1e-8 is a tolerance)
                self.status = 0
                
            else:
                self.status = 2 #Else must have failed at theta=0 
            
        
        #Upadate coords
        self.y = (self.integrator.y)
        self._history.append(self.integrator.y)  
    
     
    ## Tracking Functions ##
    
    def track(self, track_names):
        '''Initialising track ensures only variables in track_names are dumped for saving
        
        INPUT
            track_names : List containing variable names to be tracked. See self.keys for naming convention'''
        
        self.track_names = track_names
        self.track_inds = [self.keys[key] for key in track_names]  #Indicies to index from history
        
        
    def dump(self, validate=0, wipe=1):
        '''Dumps history of tracked data as a numpy array.
        
        INPUT
            validate : Set to 1 to dump validation variables. Set to 0 as default.
            wipe : Set to 1 to wipe history after dumping data. Set to 1 as default.
            
        OUTPUT
            history : Numpy array containing the history of tracked variables.'''
        
        history = self.history
        history = history[:,self.track_inds]
            
        if validate == 1:
            history = np.hstack((history,self.V_history[:,None],self.E_history[:,None],self.j_history[:,None]))
            
        if wipe == 1:
            self._history = []
            
        return history
    
    
 
class Background():
    '''The Background class takes a given 2D all all-sky iamge and maps it onto a sphere of given radius
    
    Initialisation:
        The user provides a file direction to their chosen 2D all-sky image along with its respective projection method.
        The image is downloaded and formatted to remove black border.
        
    Function:
        self.get_pix() allows one to index RGB pixel values for a given longditude and latitude
        self.show() allows one show the image using matplotlib.imshow'''
    
    
    def __init__(self, filename, R, projection='hammer', crop=1):     #Default theta as 2D case
        '''Initialises Background
    
        INPUT
            filename : File from which to download image. Enure to include file location and type (EG:.jpg).
            R : Radius at which to place background image.
            projection : 2D projection to use. Default uses hammer projection.
            crop : Set to 1 to remove black image besels. Set to 1 as default.'''
        
        super().__init__()
        
        self.R = R       #Radius of background image
        self.image = cv2.imread(filename)   #Download image
        self.projection = projection        #2D image projection
        
        #Crop image
        if crop == 1:
            self.crop_image()
            
            
    @property
    def projection(self):
        '''Chosen projection in for 3D background onto a 2D plane'''
        
        return self._projection

    @projection.setter
    def projection(self, value):
        '''Projection setter converts name to a number for faster computation'''
        
        projection_types = {'hammer':0, 'aitoff':1}    #Possible projections
        self._projection = projection_types[value]
            
    
    def crop_image(self):
        '''Crops black border from image to more eaily obtain pixels using given projection.'''
        
        #Pixel brightness
        image_brightness = np.sum(self.image, axis=-1)
        
        #y-indicies
        column_brightness = np.mean(image_brightness, axis=1)
        good_columns = np.where(column_brightness>1)  #Only take column when total brightness is high enough
        min_y_ind = good_columns[0][0] 
        max_y_ind = good_columns[0][-1]+1   #+1 accounts for indexing
        
        #x-indicies
        row_brightness = np.mean(image_brightness, axis=0)
        good_rows = np.where(row_brightness>1)        
        min_x_ind = good_rows[0][0]
        max_x_ind = good_rows[0][-1]+1
        
        #index image
        self.image = self.image[min_y_ind:max_y_ind,min_x_ind:max_x_ind,:]
        
        
    def get_pix(self, long,lat):
        '''Returns pixel values for background at a given longitude and latitude, using the specified image projection.

        INPUT
            long : longditude coordinate
            lat : latitude coordinate
        
        OUTPUT
            Array of pixel RGB pixel values for given long and lat'''
    
        #Apply periodic conditions
        long,lat = self.periodic_conditions(long,lat)
        
    
        if self.projection == 0:    #Hammer projection
            #Calculate 2D coords
            x = np.cos(lat)*np.sin(long/2) / (1+np.cos(lat)*np.cos(long/2))**.5
            y = np.sin(lat) / (1+np.cos(lat)*np.cos(long/2))**.5

            #Scale to image inds
            x_ind = ((x+1)/2 * (self.image.shape[1]-1)).astype(int)
            y_ind = ((y+1)/2 * (self.image.shape[0]-1)).astype(int)


        elif self.projection == 1:    #Aitoff projection
            #Calculate 2D coords
            long = long*180/np.pi
            lat = lat*180/np.pi
            
            x,y = pyasl.aitoff(long, lat)

            #Scale to image inds
            if type(x) == float:
                x_ind = int((x+180)/360 * (self.image.shape[1]-1))
                y_ind = int((y+90)/180 * (self.image.shape[0]-1))

            elif type(x) == np.ndarray:   #array  requires more formatting
                x_ind = ((x+180)/360 * (self.image.shape[1]-1)).astype(int)
                y_ind = ((y+90)/180 * (self.image.shape[0]-1)).astype(int)

            
        return self.image[y_ind,x_ind] 
    
    
    def periodic_conditions(self, long,lat):
        '''Applies periodic boundary conditions to long and lat coordinate. 
        Ensures -pi < long < pi and -pi/2 < lat < pi/2.
        
        INPUT
            long : longditude coordinate
            lat : latitude coordinate
            
        OUTPUT
            long : adjusted longditude coordinate
            lat : adjusted latitude coordinate'''
        
        #Adjust longditude
        while long < -np.pi:
            long += 2*np.pi

        while long > np.pi:
            long -= 2*np.pi

        #Adust latitude
        while lat < np.pi/2:
            lat += np.pi

        while lat > np.pi/2:
            lat -= np.pi

        return long,lat
    
    
    def show(self):
        '''Shows background image as a 2D plot using matplotlib.imshow'''
        
        ind_shuffle = np.array([2,1,0])  #RGB channels differ for matplotlib and cv2
        image = self.image[:,:,ind_shuffle]
        
        plt.imshow(image)
    


class Camera():
    '''The Camera class allows one to produce initial conditions for a ray traced image and store it
    
    Initialisation:
        The user provides the appropriate camera position, orientation and image specifications at initialisation.
        The camera calculates pixel coordinates at the image plane in the camera's local space (viewpoint at origin)
        This plane is tilted to face the black hole in BH space and then any inputed tilt is applied.
        The camera coords are then translated to BH space using the position viewpoint.
        Initial ray conditions are calculated as the normalised vectors from viewpoint to each pixel.
        
    Function:
        Properties ensure whenever extra tilt or translation are applied, 
            camera atributes are uptated so that it still points at the black hole with the desired tilt
        To simplify loops, initial conditions and image_store are kept as a vector.
        However when accessed as self.image, the output is formatted as a 2D array with the appropriate aspect ratio.
        '''
    
    def __init__(self, fov, aspect, num_pix, x,y,z=0, focal=.5, alpha=0, beta=0):
        '''Initialises Camera
        
        INPUT
            fov : Field of view in radians
            aspect : Image aspect ratio as width:height. EG - For 2:1 input the integer 2
            num_pix : Height of image in pixels
            x,y,z : Position of camera viewpoint in BH coords. Default 2D with z=0
            focal : Focal length is the distance from camera viewpoint to centre of image plane. Set to 0.5 as default
            alpha : Horisontal tilt in radians. Set to 0 as default
            beta : Vertical tilt in radians. Set to 0 as default'''
        
        super().__init__()
        
        self.focal = focal   #Focal length to image plane. Preset to R_s/2 (R_s=1)
        self.fov = fov       #Field of view
        self.aspect = aspect    #Aspect ratio
        self.num_pix = num_pix  #Number of pixels
        self._alpha = alpha    #Horizontal tilt
        self._beta = beta      #Vertical tilt
        
        #Image plane dimensions
        self.plane_height = 2*self.focal * np.tan(fov/2)     #Image plane height
        self.plane_width = self.plane_height * aspect             #Image plane width determined by aspect ratio
        
        #Positions of image pixels in camera's coords
        xx = np.linspace(-self.plane_width/2, self.plane_width/2, int(aspect*num_pix))
        zz = np.linspace(-self.plane_height/2, self.plane_height/2, num_pix)
        pix_x_local, pix_z_local = np.meshgrid(xx,zz)

        pix_x_local = pix_x_local.flatten()   #Reshape for compatability with matmul
        pix_z_local = pix_z_local.flatten()
        pix_y_local = np.zeros(pix_x_local.shape) + focal
        
        self.pix_r_local = np.array([pix_x_local, pix_y_local, pix_z_local])   #Compile into r_vector
        
        #Blank array for holding image
        self.image_store = np.zeros((*pix_x_local.shape,3), dtype=int)  
        
        #Position of viewpoint in BH coords
        self.view_r = np.array([x,y,z])
        

    ## Properties update camera attributes when tilted/moved ##

    @property
    def view_r(self):
        '''Position of viewpoint in BH coords'''
        return self._view_r

    @view_r.setter
    def view_r(self, value):
        '''Setter updates pixel coordinates when called.'''
        self._view_r = value

        #Update pixel coords
        self.translate_pix_coords()
        
        
    @property
    def alpha(self):
        '''Horisontal camera tilt'''
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        '''Setter updates pixel coordinates when called.'''
        self._alpha = value

        #Update pixel coords
        self.translate_pix_coords()
        
        
    @property
    def beta(self):
        '''Horisontal camera tilt'''
        return self._beta

    @beta.setter
    def beta(self, value):
        '''Setter updates pixel coordinates when called.'''
        self._beta = value

        #Update pixel coords
        self.translate_pix_coords()
        
      
    @property
    def pix_r(self):
        '''Coordinates of pixels of the image plane in BH coords'''
        return self._pix_r

    @pix_r.setter
    def pix_r(self, value):
        '''Setter calculaes initial photon vectors from viewpoint to camera pixel for use in ray tracing'''
        self._pix_r = value

        #Initial conditions for ray tracing
        pix_vectors = self.pix_r - np.expand_dims(self.view_r,axis=1)
        self.pix_vectors = pix_vectors / np.linalg.norm(pix_vectors,axis=0)  #normalise
        
    
    @property
    def image(self):
        '''Image stored in a flat vector so property returns as a reshaped 2D array for display'''
        return self.image_store.reshape(self.num_pix, int(self.aspect*self.num_pix), 3)
    
    
    ## Functions ##
    
    def translate_pix_coords(self):
        '''Translates pixels from camera to black hole coordinates. 
        Called within properties whenever attributes are updated'''
        
        x,y,z = self.view_r
        
        #Rotate to face BH
        r,theta,phi = cart2pol(x,y,z)
        
        #Rotate camera - alpha and beta account for any tilt
        pix_r = rotate(self.pix_r_local, theta - np.pi/2 + self.beta, 0)    #about x
        pix_r = rotate(pix_r, phi + np.pi/2 + self.alpha, 2)                #about z
        
        #Translate to viewpoint
        self.pix_r = pix_r + np.expand_dims(self.view_r,axis=1)
        
    
    def show(self):
        '''Shows camera image as a 2D plot using matplotlib.imshow'''
        
        ind_shuffle = np.array([2,1,0])   #RGB channels differ for matplotlib and cv2
        image = self.image[:,:,ind_shuffle]
        
        plt.imshow(image)
    
    
    