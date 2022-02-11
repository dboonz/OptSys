#!/usr/bin/env python3

import os, sys



import numpy as np
import matplotlib.pyplot as plt

from OptSys import raytracing as rt
from OptSys import visualize as vis
from OptSys import ray_utilities

if __name__ == '__main__':
    # Create a spectrometer using a simple 4f system and diffraction grating
    f = 50  # Focal length of first lens
    aperture = 25.4  # Size of lenses
    npoints = 2  # Number of light source points
    nrays = 30# Number of light rays per point
    ymax = -.5  # Limit of source plane. Controls spectral resolution
    ymin = .5
    ngroves = 600  # Grove density of diffraction grating

    # Simulate system for these wavelengths
    lmb = list(np.linspace(300,600, 20) * 1e-9)

    components = []
    rays = []
    image_plane = -200


    # Create three scene points
    scene = np.zeros((2, npoints))
    scene[1, :] = np.linspace(ymin, ymax, npoints)

    # # Place a collimation lens
    components.append(rt.Lens(f=f,
                              aperture=aperture,
                              pos=[f, 0],
                              theta=0))

    # components.append(rt.Lens(f=f,
    #                           aperture=aperture,
    #                           pos=[3*f, 0],
    #                           theta=0))

    # Place a linear filter
    components.append(rt.LinearFilter(aperture=10.01,
                                 pos=[2*f, 0],
                                 theta=0))

    components[-1].plot_transmittance()


    # Get the initial rays
    [rays, ptdict, colors] = ray_utilities.initial_rays(scene,
                                                        components[0],
                                                        nrays)
    # Create rainbow colors
    colors = vis.get_colors(len(lmb), nrays * npoints, cmap='rainbow')

    # Create a new canvas
    canvas = vis.Canvas([-5, 3*f], [-20,20])

    # Draw the components
    canvas.draw_components(components)

    # Draw the rays for each wavelength
    transmittance = np.zeros_like(lmb)
    for idx in range(len(lmb)):
        solution = rt.propagate_rays(components, rays,
                                           lmb=lmb[idx])
        canvas.draw_rays(solution, colors[idx],
                         linewidth=0.2)

        transmittance[idx] = ray_utilities.throughput(solution)

    plt.figure(2)
    plt.plot(np.array(lmb)*1e9, 1e2*transmittance)
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Transmittance [%]')
    # Show the system
    canvas.show()

    # Save a copy
    canvas.save('grating.png')
    plt.show()
