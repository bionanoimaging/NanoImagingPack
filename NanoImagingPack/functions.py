# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 14:38:16 2018

@author: ckarras

implement some functions here, to be used in the package

make sure, x is always the first argument

"""
import numpy as np
import numbers


def gauss(x, tau=1, a=1, x0=0, y0=0):
    """
        1 Dimensional gaussian function
        A         Amplitude
        tau       FWHM
        x0        x-shift
        y0        y-shift
    """
    return a * np.exp(-4 * np.log(2) * (x - x0) ** 2 / tau ** 2) + y0


def gaussian(myshape, sigma):
    """
        n-dimensional gaussian function
    """
    from .coordinates import rr2  # Why does this need to be local?
    assert isinstance(sigma, object)
    if isinstance(sigma, numbers.Number):
        sigma = len(myshape) * [sigma]
    return np.exp(-rr2(myshape, scale=tuple(1 / (np.sqrt(2) * np.array(sigma)))))


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
