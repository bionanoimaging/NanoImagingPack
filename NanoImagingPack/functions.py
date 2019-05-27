# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 14:38:16 2018

@author: Christian Karras, Rainer Heintzmann

implement some functions here, to be used in the package

make sure, x is always the first argument

"""
import numpy as np
from . import util
from . import coordinates

def sigmaFromFWHM(fwhm):
    return fwhm/(2*np.sqrt(2 * np.log(2)))

def FWHMFromSigma(fwhm):
    return (2*np.sqrt(2 * np.log(2))) * fwhm

def gauss(x, sigma = None, fwhm=10, maxVal=1.0, x0=0.0, offsetVal=0.0):
    """
        1-Dimensional gaussian function
        maxval    maximal value (not including the offsetVal below)
        sigma     if stated, this dominates over the fwhm, otherwise it is calculated form the FWHM
        fwhm      FWHM
        x0        x-shift
        offsetVal        y-shift
    """
    if sigma is None:
        sigma = sigmaFromFWHM(fwhm)
    return maxVal * np.exp(- (x - x0) ** 2 / sigma ** 2 / 2.0) + offsetVal
#    return a * np.exp(-4 * np.log(2) * (x - x0) ** 2 / fwhm ** 2) + y0


def cossqr(x, length=10, x0=0):
    """
        cos^2 
                x0          - point of one maximum;
                lenght      - distance from first maximum to first minimum (0)
    """
    return (np.cos((x - x0) / length * np.pi / 2)) ** 2


def coshalf(x, length=10, x0=0):
    """
        cos(x) from 1.0 to zero
                x0          - point of one maximum;
                lenght      - distance from first maximum to first minimum (0)
    """
    return np.cos((x - x0) / length * np.pi / 2)


def linear(x, x0=0):
    """
        (1-x) from 1.0 to zero
                x0          - point of one maximum;
    """
    return 1.0 - (x - x0)


def gauss_norm(x, sigma=1.0, x0=0, y0=0, a=1):
    """
        1 Dimensional gaussian function   form normal distribution
        A         Amplitude
        tau       sigma
        x0        x-shift
        y0        y-shift
    """
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2) + y0)


def gauss2D(amplitude=1, center_x=0, center_y=0, sigma_x=1, sigma_y=1, rotation=0, offset=0):
    """
        2 Dimensional gaussian function
        Returns a gaussian function with the given parameters
    
        Rotation in Degree
    """
    center_x = float(center_x)
    center_y = float(center_y)
    sigma_x = float(sigma_x)
    sigma_y = float(sigma_y)
    rotation = np.deg2rad(rotation)
    a = (np.cos(rotation) ** 2) / (2 * sigma_x ** 2) + (np.sin(rotation) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * rotation)) / (4 * sigma_x ** 2) + (np.sin(2 * rotation)) / (4 * sigma_y ** 2)
    c = (np.sin(rotation) ** 2) / (2 * sigma_x ** 2) + (np.cos(rotation) ** 2) / (2 * sigma_y ** 2)
    return lambda x, y: offset + amplitude * np.exp(
        - (a * ((x - center_x) ** 2) + 2 * b * (x - center_x) * (y - center_y) + c * ((y - center_y) ** 2)))


def gaussian(myshape, sigma, placement='center'):
    """
        n-dimensional gaussian function
    """
    assert isinstance(sigma, object)
    sigma = np.array(util.repToList(sigma, len(myshape)))
    myNorm = 1.0
    for d in range(len(sigma)):
        if myshape[d]>1 and sigma[d] > 0.0:
            myNorm *= 1.0/np.sqrt(2*np.pi*sigma[d]**2)

    sigma[sigma <= 0.0] = 1e-10
    return myNorm * np.exp(- coordinates.rr2(myshape, scale=tuple(1 / (np.sqrt(2) * np.array(sigma))), placement=placement))


