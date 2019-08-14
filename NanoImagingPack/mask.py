#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:23:24 2017

@author: root

should provide masks:
    
    2D and 3D
    
    Sine Shaped
    Square
    Rect
    circular

"""

import numpy as np
from .coordinates import xx, yy, zz, VolumeList, freq_ramp


def sin2D(mysize = (256,256), angle=0, period = 10, init_phase =0):
    """
        create a sinusoidal pattern (2Dimensional) with sizes size_x and size_y in pixel

        at given angle in degree

        given period in pixels

        and initial phase in units of pi

    """
    if type(mysize) == np.ndarray:
        mysize = mysize.shape
    kx =1/period*np.sin(angle*np.pi/180)
    ky =1/period*np.cos(angle*np.pi/180)
    grat =( np.sin((xx(mysize)*kx+yy(mysize)*ky+init_phase)*np.pi))**2
    return(grat)

def spherical_cap(mysize=  (256,256,32), maskpos = (0,0,0), R_sphere = 100, R_cap = 30, thickness = 'fill'):
    """
    Create a binary mask of a spherical cap (only shell or whole volume in a 3D volume with the cap pointing in z -direction
    :param mysize: Shape of the volume
    :param maskpos:  Center position of the ground surface (pixels),
                    int (than same mask position for all axes);
                    tuple or list (give the mask positions)
                    'center'
    :param R_sphere: Radius of the sphere (outer Radius in case non filled sphere)
    :param R_cap: Radius of the ground surface of the cap
    :param thickness: Thickness in pixels (than it makes a shell), or 'fill', than it fills the volume
    :return:

    Example:
    import NanoImagingPack as nip
    out = nip.spherical_cap(mysize=  (256,256,32), maskpos = (0,0,0), R_sphere = 100, R_cap = 30, thickness = 'fill')
    """

 
    if (type(maskpos) == list or type(maskpos) == tuple):
            if len(maskpos) > len(mysize):
                print('Waringing: Too many mask positions. Only taking the first '+str(len(maskpos)))
                maskpos = maskpos[:len(mysize)]
            if len(maskpos) < len(mysize):
                maskpos += tuple(np.ones(len(mysize)-len(maskpos)).astype(int)*0)
    elif (type(maskpos) == int or type(maskpos) == float):
        maskpos = tuple(np.ones(len(mysize)).astype(int)*maskpos)
    elif maskpos == 'center':
        maskpos = tuple(x//2+(np.mod(x,2)-1) for x in mysize)
    else:
          try: 
              if np.issubdtype(maskpos.dtype, np.number):
                  maskpos = tuple(np.ones(len(mysize)).astype(int)*maskpos)
          except AttributeError:
              pass;
    if R_cap > R_sphere:
        print('Error: Cap radius cannot be larger then sphere radius!')
    else:
        P0 = (maskpos[0],maskpos[1], maskpos[2]- np.sqrt(R_sphere**2-R_cap**2))
        Sphere = VolumeList(mysize, MyCenter = P0, polar_axes = 'all', return_axes =0)
        Z = zz(mysize, mode = 'positive')
        if thickness == 'fill':
            return(((Sphere<=R_sphere)*(Z>=maskpos[2]))*1)
        else:
            return((((Sphere<=R_sphere)*1-(Sphere<=(R_sphere-thickness))*1)*(Z>=maskpos[2]))*1)


def spherical_mask(mysize = (256,256,32), maskpos = (0,0,0), R_sphere = 100, thickness = 'fill'):
    """
    SPHERICAL_MASK creates a spherical mask
    Create a binary mask of a spherical cap (only shell or whole volume in a 3D volume with the cap pointing in z -direction)

    :param mysize: Shape of the volume
    :param maskpos: Center position of the sphere (pixels)
                    int or float (than same mask position for all axes);
                    tuple or list (give the mask positions)
                    'center'
    :param R_sphere: Radius of the sphere (outer Radius in case non filled sphere)
    :param thickness: Thickness in pixels (than it makes a shell), or 'fill', than it fills the volume
    :return: a binary 3-D array of spherical cap pointing in z -direction

    Example:
    import NanoImagingPack as nip
    out = nip.spherical_mask(mysize = (256,256,32), maskpos = (0,0,0), R_sphere = 100, thickness = 'fill')

    """
    Sphere = VolumeList(mysize, MyCenter = maskpos, polar_axes = 'all', return_axes =0)
    if thickness == 'fill':
        return((Sphere<=R_sphere)*1)
    else:
        return((Sphere<=R_sphere)*1-(Sphere<=(R_sphere-thickness))*1)


def create_circle_mask(mysize =(256,256),maskpos = (0,0) ,radius=100, zero = 'center'):
    """
    CREATE_CIRCLE_MASK creates a circle mask of given radius around the maskpos coordinates
    :param mysize: tuple of sizes ()
    :param maskpos: center of circle
    :param radius: circle radius -> can be list or tuple if the radii in two directions are different (i.e. elliptical mask)
    :param zero: 'center': cooridnate origin at center
                'image': like image coordinates (zero is in the upper left corner)
    :return: a 2-D mask

    Example:
    import NanoImagingPack as nip
    from NanoImagingPack import v5
    out = nip.create_circle_mask(mysize =(256,256),maskpos = (0,0) ,radius=100, zero = 'center')
    v5(out)

    """
    import numbers
    if isinstance(radius, numbers.Number):
        radius = [radius, radius]
    if zero == 'center':
        xr = xx(mysize)
        yr = yy(mysize)
    elif zero == 'image':
        xr = xx(mysize, placement = 'positive')
        yr = yy(mysize, placement = 'positive')
    mask = ((xr-maskpos[0])**2/radius[0]**2+(yr-maskpos[1])**2/radius[1]**2<1)*1
    return(mask)


def otf_support(im, pixelsize, l, NA):
    """
    OTF_SUPPORT Creates a mask for a (2D) image, which is 1 for k<k_0 and 0 for k>k_0 (k_0 is the abbe frequency)
    :param im: Input image
    :param pixelsize: physical dimension of the pixel
    :param l: wavelength
    :param NA: Numerical Aperture
    :return: a 2-D mask
    Note: Make sure l and pixelsize have the same unit

    Example:
    import NanoImagingPack as nip
    from NanoImagingPack import v5
    im = nip.readim()
    out = nip.otf_support(im,(16,16),500,1.4)
    v5(out)
    """

    import numbers
    if isinstance(pixelsize, numbers.Number):
        pixelsize = [pixelsize,pixelsize]

    fx = freq_ramp(im, pixelsize[0], axis =0)
    fy = freq_ramp(im, pixelsize[1], axis =1)
    k0 = 2*NA/l
    return((np.sqrt(fx**2+fy**2)<k0)*1)