# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 17:45:44 2017

@author: ckarras
"""

import numpy as np
from .util import get_type;
from .image import image;

def poisson(im, NPhot = None, seed=None, dtype=None):
    '''
        Puts poisson noise on an image.
        Input:  image matrix im
                Photoncount of the brigthest pixel in im (default=None, meaning use the given scaling)
        
        Be carefull: the return data type is a float!
        
    '''
    if NPhot!=None:
        im = im.astype(float)/np.max(im)*NPhot
    if seed != None:
        np.random.seed(seed)
    res=image(np.random.poisson(im))
    if dtype != None:
        res=res.astype(dtype)
    return res
