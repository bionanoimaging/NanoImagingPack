#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 17:40:05 2017

@author: root
"""
import numpy as np
from .functions import gauss2D as gaussian
import scipy as scp
from scipy import optimize
from .util import max_coord
from .view import view
from .view5d import v5
from .image import catE

def fit_gauss2D(dat, DBG = False, title = '2D Gauss fit', startPos=None, startSigma=None):
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
        return arr[~np.isnan(arr)]

    def check_angle_bonds(dat,p):
        if (p[5] <0) or (p[5]>180):
            return 1e6
        else:
            return remove_nan(np.ravel(gaussian(*p)(*np.indices(dat.shape))-dat))

    if startPos is None:
        m1, m2 = max_coord(dat)
    else:
        (m1, m2) = startPos

    if startSigma is None:
        s1 = np.size(dat, axis=-1)/2
        s2 = np.size(dat, axis=-2)/2
    else:
        (s1, s2) = startSigma

    x,y = np.meshgrid(np.arange(0,np.size(dat, axis =-1),1),np.arange(0,np.size(dat, axis =-2),1))
    guess =np.asarray([np.max(remove_nan(dat)), m1, m2, s1, s2, 90, np.min(remove_nan(dat))])

    deviation = lambda p: remove_nan(np.ravel(gaussian(*p)(np.indices(dat.shape)[1], np.indices(dat.shape)[0])-dat))
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
    return p, gaussian(*p)(x,y)


def fit_FWHM(dat, Verbose = True, startPos=None, startSigma=None):
    """
        Fit a 2D gaussian on a given data set

        Optimization method is scipy.optimize.leastsq
        Returns:
        FWHM_x, FWHM_y:  in dat.pixelsize
        amplitude

        Verbose: Verbose option (print results and plot data and fit), default: True
    """
    if Verbose:
        print("Starting fit to determine the FWHMs")

    params, fitted = fit_gauss2D(dat, DBG=Verbose, startPos=startPos, startSigma=startSigma)
    (amplitude, center_x, center_y, sigma_x, sigma_y, rotation, offset) = params

    FWHM_x = 2*(sigma_x * np.sqrt(-np.log(0.5)*2))
    FWHM_y = 2*(sigma_y * np.sqrt(-np.log(0.5)*2))

    if Verbose:
        print("FWHM x: " + str(np.abs(FWHM_x)) + " pixels")
        print("FWHM y: " + str(np.abs(FWHM_y)) + " pixels")
    FWHM_x *= dat.pixelsize[-1]
    FWHM_y *= dat.pixelsize[-2]
    if Verbose:
        print("FWHM x: " + str(np.abs(FWHM_x)) + " " + dat.unit)
        print("FWHM y: " + str(np.abs(FWHM_y)) + " " + dat.unit)
        v5(catE(dat, fitted))
    return FWHM_x, FWHM_y, amplitude
