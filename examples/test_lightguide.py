#!/usr/bin/env python3

import os, sys

sys.path.append('../OptSys')

import numpy as np
import matplotlib.pyplot as plt

import raytracing as rt
import visualize as vis
import ray_utilities

if __name__ == '__main__':
    # Create a spectrometer using a simple 4f system and diffraction grating
    f = 50  # Focal length of all lenses
    aperture = 25.4  # Size of lenses
    npoints = 3  # Number of light source points
    nrays = 50 # number of direction per light point
    ymax = -2  # Limit of source plane. Controls spectral resolution
    ymin = 2

    # Simulate system for these wavelengths
    lmb = 400e-9#list(np.linspace(400, 700, 11) * 1e-9)

    components = []
    rays = []
    image_plane = -200


    # Create three scene points
    scene = np.zeros((2, npoints))
    scene[1, :] = np.linspace(ymin, ymax, npoints)

    # Place a lightguide
    components.append(rt.LightGuide(
                              aperture=aperture,
                              pos=[f, 0],
                              theta=np.deg2rad(10),
                              NA=0.1))

    # Get the initial rays
    [rays, ptdict, colors] = ray_utilities.initial_rays(scene,
                                                        components[0],
                                                        nrays)
    # colour rays by direction instead of origin
    colors = ['#ff6660',] * nrays
    colors += ['#00ff00',] * nrays
    colors += ['#ff1100', ] * nrays

                                                        # nrays)
    # # Create rainbow colors
    # # colors = 'rgb' * nrays
    colors = vis.get_colors(nrays, 1, cmap='rainbow', flatten=True)
    colors = np.vstack([colors,] * npoints)

    # colors = colors * npoints

    # Create a new canvas
    canvas = vis.Canvas([-5, 4.1 * f], [-2 * aperture, 2 * aperture])

    # Draw the components
    canvas.draw_components(components)

    # Draw the rays
    canvas.draw_rays(rt.propagate_rays(components, rays,
                                           lmb=lmb), colors,
                         linewidth=0.2)

    # Show the system
    canvas.show()

    # Save a copy
    canvas.save('lightguide.png')
    plt.show()
