# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:47:22 2019

@author: pi96doc and others (from Internet)
This is for contributed and downloaded code. The web-links are given before each code.
"""

import numpy as np
import NanoImagingPack as nip

# some code from https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile
# modified by R.H.
def radial_profile(data, center=None, binsize=1.0):
    '''
    calculates the radial profile of an image
    the original code is from https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile

    Parameters
    ----------
    data: data to calcuate the profile form
    center (default=None): center from where to calculate the radius. None means the (Fourier-) middle of the image is used.
    binsize (default=1.0): size of each bin

    Returns
    -------
    One-D radial profile

    See also
    -------
    

    Example
    -------
    '''
    if center==None:
        center=data.mid()
        
    y,x = np.indices((data.shape)) # first determine radii of all pixels
    r = np.sqrt((x-center[1])**2+(y-center[0])**2)    

    # radius of the image.
    r_max = np.max(r)  

    bins=(r_max*binsize).astype('int32')
    ring_brightness, radius = np.histogram(r, weights=data, bins=bins)
    return nip.image(ring_brightness)

