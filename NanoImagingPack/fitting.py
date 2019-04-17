#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 17:40:05 2017

@author: root
"""
import numpy as np
from .functions import gauss2D as gaussian
import scipy as scp
from .util import max_coord
from .view import view


def fit_gauss2D(dat, DBG = False, title = '2D Gauss fit'):
    """
        Fit a 2D gaussian on a given data set

        Optimization method is scipy.optimize.leastsq
        Returns:
                1) parameters of gaussian fit p:
                    amplitude, center_x, center_y, sigma_x, sigma_y, rotation, offset
                2) optimized fit function g with the same dimensions as dat

        DBG: Debugging option (show guess or other stuff)
     """

    def remove_nan(arr):
        return(arr[~np.isnan(arr)])

    def check_angle_bonds(dat,p):
        if (p[5] <0) or (p[5]>180):
            return(1e6)
        else:
            return remove_nan(np.ravel(gaussian(*p)(*np.indices(dat.shape))-dat))

    m1,m2 =max_coord(dat)
    x,y = np.meshgrid(np.arange(0,np.size(dat,axis =0),1),np.arange(0,np.size(dat,axis =1),1))
    guess =np.asarray([np.max(remove_nan(dat)),m1,m2,np.size(dat,axis=0)/2,np.size(dat,axis=0)/2,90,np.min(remove_nan(dat))])

    deviation = lambda p: remove_nan(np.ravel(gaussian(*p)(np.indices(dat.shape)[1],np.indices(dat.shape)[0])-dat))
    p, success = scp.optimize.leastsq(deviation, guess)
    if DBG:
        print('Initial guess parameters:')
        print(guess)
        print('Result parameters:')
        print(p)
        print('Successful?')
        print(success)
        view(gaussian(*p)(x,y), title= title)

    '''
     The Parameter interchange is a auxilary solution! For some reason center_x and center_y are mixed up in the fitted result!
   
    a=p[1];
    p[1] = p[2];
    p[2] = a;
     '''
    return(p, gaussian(*p)(x,y))