#!/usr/bin/env python

'''
    Module for rendering simple optical outputs using raytracing.
'''

# System imports
import os
import sys
import pdb

# Scipy and related imports
from functools import lru_cache

import numpy as np
import scipy.linalg as lin
import scipy.ndimage as ndim
import matplotlib.pyplot as plt

def propagate_rays(components, rays, lmb=525e-9):
    '''
        Function to propagate rays through a set of components

        Inputs:
            components: List of optical components
            rays: List of 3-tuple of rays with x-coord, y-coord and angle
            lmb: Wavelength of rays in m. Default if 525nm

        Outputs:
            ray_bundles: For N components, this is a list of 3x(N+1) coordinates
                of rays propagated through the components.
    '''
    # Create output ray first
    ncomponents = len(components)
    nrays = len(rays)
    ray_bundles = []

    for idx in range(nrays):
        ray_bundles.append(np.zeros((3, ncomponents+1)))
        ray_bundles[idx][:, 0] = rays[idx]

    # Now propagate each ray through each component
    for r_idx in range(nrays):
        for c_idx in range(1, ncomponents+1):
            ray_bundles[r_idx][:, c_idx] = components[c_idx-1].propagate(
                                            ray_bundles[r_idx][:, c_idx-1],
                                            lmb).reshape(3,)

    # Done, return
    return ray_bundles

def angle_wrap(angle):
    '''
        Wrap angle between -180 to 180

        Inputs:
            angle: In radians

        Outputs:
            wrapped_angle: In radians
    '''
    if angle > np.pi:
        angle -= 2*np.pi
    elif angle < -np.pi:
        angle += 2*np.pi

    return angle

class OpticalObject(object):
    '''
        Generic class definition for optical object. This object is inherited
        to create optical objects such as lenses, mirrors and gratings

        Common properties:
            1. Position of the object
            2. Orientation of the object w.r.t global Y axis
            3. Aperture: Diameter of the object

        Specific properties:
            Lens/mirror: Focal length
            Grating (Transmissive only): Number of groves per mm
            DMD: Deflection angle

        Note: Unless you definitely know what you are doing, do not create
              any object with this class. Look at the objects which inherit
              this class.
    '''
    def __init__(self, aperture, pos, theta, name=None):
        '''
            Generic constructor for optical objects.

            Inputs:
                aperture: Aperture size
                pos: Position of lens in 2D cartesian grid
                theta: Inclination of lens w.r.t Y axis
                name: Name (string) of the optical component. Name will be
                    used for labelling the ocmponents in drawing

            Outputs:
                None.
        '''
        self.theta = theta
        self.pos = pos
        self.aperture = aperture
        self.name = name

        # Create coordinate transformation matrix
        self.H = self.create_xform()
        self.Hinv = lin.inv(self.H)

    def get_intersection(self, orig, theta):
        '''
            Method to get interesection of optical object plane and ray

            Inputs:
                orig: Origin of ray
                theta: Orientation of ray w.r.t x-axis

            Outputs:
                dest: Destination of ray
        '''

        # If theta is nan, it means the ray terminated
        if np.isnan(theta):
            return np.ones((3,1))*float('nan')

        # Transform ray origin to new coordinates
        p = np.array([[orig[0]], [orig[1]], [1]], dtype=np.float64)

        p_new = self.H.dot(p)

        # Similarly find the angle in new coordinate system
        theta_new = theta + self.theta

        # Now compute the intersection
        x_new_tf = 0
        y_new_tf = p_new[1][0] - p_new[0][0]*np.tan(theta_new)

        # Sanity check to see if the interesection lies within the aperture
        eps = 1e-6
        if y_new_tf > self.aperture/2.0 + eps or y_new_tf < -self.aperture/2.0 - eps:
           flag = float('nan')
        else:
            flag = 1.0

        p_tf = np.array([[x_new_tf], [y_new_tf], [1]])

        # Go back to original system and return result
        p_final = self.Hinv.dot(p_tf)
        p_final[2] = flag

        return p_final

    def propagate(self, point, lmb=None):
        '''
            Function to propagate a ray through the object. Requires an extra
            definition for computing angles

            Inputs:
                point: 3x1 vector with x-coordinate, y-coordinate and angle (rad)
                lmb: Wavelength. Only required for grating
            Outputs:
                dest: 3x1 vector with x-coordinate, y-coordinate and angle (rad)
        '''
        # First get intersection of the point with object plane
        dest = self.get_intersection(point[:2], point[2])

        # Then compute angle
        if np.isnan(dest[2]):
            return dest

        dest[2] = self._get_angle(point, lmb, dest)

        return dest

    def create_xform(self):
        '''
            Function to create transformation matrix for coordinate change

            Inputs:
                None

            Outputs:
                H: 3D transformation matrix
        '''
        R = np.zeros((3, 3))
        T = np.zeros((3, 3))

        # Rotation
        R[0, 0] = np.cos(self.theta)
        R[0, 1] = -np.sin(self.theta)
        R[1, 0] = np.sin(self.theta)
        R[1, 1] = np.cos(self.theta)
        R[2, 2] = 1

        # Translation
        T[0, 0] = 1
        T[0, 2] = -self.pos[0]
        T[1, 1] = 1
        T[1, 2] = -self.pos[1]
        T[2, 2] = 1

        return R.dot(T)


class Sensor(OpticalObject):
    ''' Class definition for a sensor object'''
    def __init__(self, aperture, pos, theta, name='Sensor'):
        '''
            Constructor for sensor object. Sensor is simply a plan which blocks
            all rays

            Inputs:
                aperture: Size of sensor
                pos: Position of sensor in 2D cartesian grid
                theta: Inclination of sensor w.r.t Y axis
                name: Name of the optical component. If Empty string, generic
                    name is assigned. If None, no name is printed.

            Outputs:
                None.
        '''
        # Initialize parent optical object parameters
        OpticalObject.__init__(self, aperture, pos, theta, name)

        # Extra parameters
        self.type = 'sensor'


    def _get_angle(self, point, lmb, dest):
        '''
            Angle after propagation. Since this is a sensor, return NaN to flag
            end of ray

            Inputs:
                point: 3-tuple point with x, y, angle
                lmb: Wavelength of ray. Only needed for grating
                dest: 2D coordinate of interesection of ray with plane

            Outputs:
                theta: NaN, since this is a sensor
        '''
        return float('NaN')


class LightGuide(OpticalObject):
    ''' Class definition for a LightGuide object.

    Most waveguides, such as optical fibers, only accept light that falls both within an aperture
    and within an acceptance angle. This class allows to define an aperture that only accepts rays
    up to an angle given by the NA.
    '''

    def __init__(self, aperture, pos, theta, name='Lightguide', NA=0.4):
        '''
            Constructor for sensor object. Sensor is simply a plan which blocks
            all rays

            Inputs:
                aperture: Size of sensor
                pos: Position of sensor in 2D cartesian grid
                theta: Inclination of sensor w.r.t Y axis [radians]
                name: Name of the optical component. If Empty string, generic
                    name is assigned. If None, no name is printed.

            Outputs:
                None.
        '''
        # Initialize parent optical object parameters
        OpticalObject.__init__(self, aperture, pos, theta, name)

        # Extra parameters
        self.type = 'Lightguide'
        self.NA = NA


    def _get_angle(self, point, lmb, dest):
        '''
            Angle after propagation. Since this is a sensor, return NaN to flag
            end of ray

            Inputs:
                point: 3-tuple point with x, y, angle
                lmb: Wavelength of ray. Only needed for grating
                dest: 2D coordinate of interesection of ray with plane

            Outputs:
                theta: NaN, since this is a sensor
        '''
        max_angle = np.arcsin(self.NA)

        incidence_angle = abs(point[-1]+self.theta)
        NA = np.sin(incidence_angle)
        if abs(NA) < self.NA:
            return -self.theta
        incidence_angle2 = abs(point[-1] + self.theta + np.pi)
        if  incidence_angle < max_angle or incidence_angle2 < max_angle:
            return - self.theta
        else:
            print(f'Incidence angle: {incidence_angle}, corresponding to NA {np.sin(incidence_angle)} is not guided')
        return float('NaN')

class LinearFilter(OpticalObject):
    """ Class for a linear spectral filter. This is a bandpass filter where the transmitted wavelengths depend on the y position.

    At the moment only works for theta==0, but this could easily be changed.

    """
    def __init__(self, aperture, pos, theta, name='Lin. Filter',
                 wl_min_nm=400, wl_max_nm=700, window_width_nm=20, upsidedown=True):
        '''Constructor for lens object.

            Inputs:
                aperture: Aperture size
                pos: Position of lens in 2D cartesian grid
                theta: Inclination of lens w.r.t Y axis. Must be 0.
                name: Name of the optical component. If Empty string, generic
                    name is assigned. If None, no name is printed.
                wl_min_nm: Wavelength that is transmitted at the lower end of the filter
                wl_max_nm: Wavelength that is transmitted at the higher end of the filter
                window_width_nm: Window width in nanometer.

            Outputs:
                None.
        '''

        assert theta == 0, 'Theta != 0 is not implemented at the moment'
        OpticalObject.__init__(self, aperture, pos, theta, name)
        self.type = 'Linear Filter'
        self.wl_min = wl_min_nm * 1e-9
        self.wl_max = wl_max_nm * 1e-9
        self.window_width_nm = window_width_nm
        self.upsidedown = upsidedown
        # print('Min max wavelengths:')
        # print(self._calc_center_wavelength(self.pos[1]-self.aperture))
        # print(self._calc_center_wavelength(self.pos[1]+self.aperture))

    def _get_angle(self, point, lmb, dest):

        """
                point: 3-tuple point with x, y, angle
                lmb: Wavelenght of ray. Only needed for grating
                dest: 2D coordinate of interesection of ray with plane

        """
        # coordinates of the origin of the beam
        x, y, angle = point
        # coordinates at the interface with our surface
        x,y,angle_ = tuple(dest.flatten())

        angle = point[-1]
        # calculate the central wavelength at this point
        center_wavelength = self._calc_center_wavelength(y)
        # check if we're within the central window, if so, transmit, if not, don't.

        d_wl = abs(lmb-center_wavelength)*1e9
        if d_wl < self.window_width_nm:
            return angle
        else:
            return np.nan

    @lru_cache
    def _calc_center_wavelength(self, y):
        """Calculate the central wavelength at position y"""
        y = np.array(y)
        y = y - self.pos[1]
        y = y / self.aperture # from -0.5 to 0.5
        y = y + 0.5 # 0 to 1
        if self.upsidedown:
            y = 1 - y # 1 to 0

        central_wavelength_at_position = self.wl_min + y * (self.wl_max - self.wl_min)
        # print(y, central_wavelength_at_position*1e9)
        return np.clip(central_wavelength_at_position, self.wl_min, self.wl_max)

    def plot_transmittance(self):
        # plot the transmittance
        import matplotlib.pyplot as plt
        positions = np.linspace(-self.aperture, self.aperture, 50)
        center_wavelengths = self._calc_center_wavelength(tuple(list(positions))) * 1e9
        plt.plot(positions, center_wavelengths)
        plt.fill_between(positions, center_wavelengths-self.window_width_nm, center_wavelengths+self.window_width_nm)
        plt.xlabel('Position [mm]')
        plt.ylabel('Central wavelength [nm]')
        plt.show()



class Lens(OpticalObject):
    ''' Class definition for lens object'''
    def __init__(self, f, aperture, pos, theta, name=""):
        '''
            Constructor for lens object.

            Inputs:
                f: Focal length of lens
                aperture: Aperture size
                pos: Position of lens in 2D cartesian grid
                theta: Inclination of lens w.r.t Y axis
                name: Name of the optical component. If Empty string, generic
                    name is assigned. If None, no name is printed.

            Outputs:
                None.
        '''
        if name == "":
            name = f'f, A = {f:.1f},{aperture:.1f}'

        # Initialize parent optical object parameters
        OpticalObject.__init__(self, aperture, pos, theta, name)

        # Extra parameters
        self.f = f
        self.type = 'lens'

    def _get_angle(self, point, lmb, dest):
        '''
            Angle after propagation. This function is used by propagate function
            of master class. Do not use it by itself.

            Inputs:
                point: 3-tuple point with x, y, angle
                lmb: Wavelenght of ray. Only needed for grating
                dest: 2D coordinate of interesection of ray with plane

            Outputs:
                theta: Angle after propagation.
        '''
        # Find the point on focal plane where all parallel rays meet
        focal_dest = self.Hinv.dot(np.array([[self.f],
                                             [self.f*np.tan(point[2]+self.theta)],
                                             [1]]))
        # Now find the angle
        theta = np.arctan2(focal_dest[1]-dest[1],focal_dest[0]-dest[0])

        # For concave lens, add 180 degrees
        if self.f < 0:
            theta += np.pi

        return theta

class SphericalMirror(OpticalObject):
    ''' Class definition for spherical mirror object'''
    def __init__(self, f, aperture, pos, theta, name=""):
        '''
            Constructor for spherical mirror object.

            Inputs:
                f: Focal length of mirror -- positive for concave and negative 
                    for convex
                aperture: Aperture size
                pos: Position of mirror in 2D cartesian grid
                theta: Inclination of mirror w.r.t Y axis
                name: Name of the optical component. If Empty string, generic
                    name is assigned. If None, no name is printed.

            Outputs:
                None.
        '''
        if name == "":
            name = 'f = %d'%f

        # Initialize parent optical object parameters
        OpticalObject.__init__(self, aperture, pos, theta, name)

        # Extra parameters
        self.f = f
        self.type = 'spherical_mirror'

    def _get_angle(self, point, lmb, dest):
        '''
            Angle after propagation. This function is used by propagate function
            of master class. Do not use it by itself.

            Inputs:
                point: 3-tuple point with x, y, angle
                lmb: Wavelenght of ray. Only needed for grating
                dest: 2D coordinate of interesection of ray with plane

            Outputs:
                theta: Angle after propagation.
        '''
        # Find the point on focal plane where all parallel rays meet
        focal_dest = self.Hinv.dot(np.array([[-self.f],
                                             [self.f*np.tan(point[2]+self.theta)],
                                             [1]]))
        # Now find the angle
        theta = np.arctan2(focal_dest[1]-dest[1], focal_dest[0]-dest[0])

        # For convex lens, add 180 degrees
        if self.f < 0:
            theta += np.pi

        return theta
    
class Grating(OpticalObject):
    ''' Class definition for a diffraction grating'''
    def __init__(self, ngroves, aperture, pos, theta, m=1, transmissive=True,
                 name=''):
        '''
            Constructor for Grating object.

            Inputs:
                ngroves: Number of groves per mm.
                aperture: Size of diffraction grating
                pos: Position of diffraction grating
                theta: Inclination of theta w.r.t Y axis
                m: Order of diffraction. If you want light to diffract on the
                   other side, use m=-1
                transmissive: If True, diffraction grating is treated as being
                    transmissive, else is treated as reflective
                name: Name of the optical component. If Empty string, generic
                    name is assigned. If None, no name is printed.

            Outputs:
                None.
        '''

        # Initialize parent optical object parameters
        OpticalObject.__init__(self, aperture, pos, theta)

        # Extra parameters
        self.ngroves = ngroves
        self.m = m
        self.type = 'grating'
        if name == '':
            self.name = f'G: {self.ngroves} l/mm'
        else:
            self.name = name


    def _get_angle(self, point, lmb, dest):
        '''
            Function to compute angle after propagation. This is used by master
            class, do not use it by itself.

            Inputs:
                point: 3-tuple of x-coordinate, y-coordinate and angle (radian)
                lmb: Wavelength of the ray.
                dest: 2D coordinate of interesection of ray with plane

            Outputs:
                lmb: Angle after propagation
        '''
        # Compute destination first
        dest = self.get_intersection(point[:2], point[2])

        # Next find output angle
        incident_theta = point[2] + self.theta

        a = 1e-3/self.ngroves
        refracted_theta = np.arcsin(np.sin(incident_theta) - self.m*lmb/a)

        return angle_wrap(refracted_theta - self.theta)

class Mirror(OpticalObject):
    ''' Class definition for Mirror object'''
    def __init__(self, aperture, pos, theta, name='Mirror'):
        '''
            Constructor for Mirror object.

            Inputs:
                aperture: Size of diffraction grating
                pos: Position of diffraction grating
                theta: Inclination of theta w.r.t Y axis
                name: Name of the optical component. If Empty string, generic
                    name is assigned. If None, no name is printed.

            Outputs:
                None.
        '''

        # Initialize parent optical object parameters
        OpticalObject.__init__(self, aperture, pos, theta, name)

        # Extra parameters
        self.type = 'mirror'

    def _get_angle(self, point, lmb, dest):
        '''
            Function to compute angle after propagation through mirror.

            Inputs:
                point: 3-tuple of x-coordinate, y-coordinate and angle (radians)
                lmb: Wavelength of ray, only needed for grating

            Outputs:
                theta: Angle after propagation
        '''

        # Next compute deflection angle
        theta_dest = np.pi - point[2] - 2*self.theta

        return angle_wrap(theta_dest)

class DMD(OpticalObject):
    ''' Class definition for DMD object'''
    def __init__(self, deflection, aperture, pos, theta, name='DMD'):
        '''
            Constructor for DMD object.

            Inputs:
                deflection: Deflection angle of DMD
                aperture: Size of diffraction grating
                pos: Position of diffraction grating
                theta: Inclination of theta w.r.t Y axis
                name: Name of the optical component. If Empty string, generic
                    name is assigned. If None, no name is printed.

            Outputs:
                None.
        '''

        # Initialize parent optical object parameters
        OpticalObject.__init__(self, aperture, pos, theta, name)

        # Extra parameters
        self.deflection = deflection
        self.type = 'dmd'

    def _get_angle(self, point, lmb=None, dest=None):
        '''
            Function to compute angle after propagation

            Inputs:
                point: 3-tuple of x-coordinate, y-coordinate, angle (radian)
                lmb: Wavelength of ray. Not relevant for DMD

            Outputs:
                theta: Angle after propagation
        '''
        return angle_wrap(np.pi - point[2] - 2*(self.theta + self.deflection))
        # return (np.pi - point[2] - 2*(self.theta + self.deflection))

class Aperture(OpticalObject):
    ''' Class definition for aperture object'''
    def __init__(self, aperture, pos, theta, name='Aperture'):
        '''
            Function to initialize the aperture object

            Inputs:
                aperture: Size of the aperture
                pos: Position of the aperture
                theta: Clockwise rotation of aperture w.r.t y-axis in radians
                name: Name of the Optical object. Default is 'Aperture'
        '''

        # Initialize parent optical object parameters
        OpticalObject.__init__(self, aperture, pos, theta, name)

        # Extra parameters
        self.type = 'aperture'

    def _get_angle(self, point, lmb=None, dest=None):
        '''
            Function to compute output angle after propagation

            Inputs:
                point: 3-tuple of x-coordinate, y-coordinate, angle (radians)
                lmb: Wavelength of ray. Not relevant.

            Outputs:
                theta: Angle after propagation
        '''
        # Simply return the angle
        return point[2]
