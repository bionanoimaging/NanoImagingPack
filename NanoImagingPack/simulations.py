# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 14:38:16 2018

@author: Rainer Heintzmann

handy functions for microscopy image simulations

"""

from . import image, microscopy, noise, coordinates


def simulateImage(obj=None, PSFParam=None, pixelsize=None,  MaxPhotons=1000, ReadNoise=None, NumSimulations=1):
    """
        simulates a microscopy image
    :param obj: Can be a string with "2D" or "3D" or a filename (with  path) or an array with object data
    :param pixelsize: vector of pixel sizes
    :param PSFParam: see PSF for details. By default a widefield PSF is simulated.
    :param MaxPhotons:  Maximal number of expected emitted photons. The random number generator will always start with seed=0. If MaxPhotons is None, no photon noise is applied.
    :param ReadNoise:  If specified, Gaussian readnoise will be added (in addition to Poisson noise, if specified). The value states the Std.Dev. of the readnoise.
    :param NumSimulations: If specified, multiple simulations with different noise instances will be generated. seed=0 is applied only to the first simulation
    :return: simulated image(s), psf, object and perfect image
    """

    if obj is None:
        obj = image.readim()
    else:
        if type(obj) == str:
            if obj == "2D":
                obj = image.readim()
            elif obj == "3D":
                obj = image.readim("obj3d")
            else:
                obj = image.readim(obj)

    if pixelsize is None:
        pixelsize = [100, 50, 50]
    else:
        obj.pixelsize = pixelsize

    if PSFParam is None:
        PSFParam = microscopy.PSF_PARAMS

    if obj.pixelsize is None:
        obj.pixelsize = pixelsize

    psf = microscopy.psf(obj, PSFParam)

    rotf = microscopy.PSF2ROTF(psf)
    pimg = microscopy.convROTF(obj,rotf)
    pimg[pimg<0.0]=0.0

    if MaxPhotons is not None:
        nimg = noise.poisson(pimg,NPhot=MaxPhotons, seed=0, dtype='float32')  # Christian: default should be seed=0
    else:
        nimg = pimg

    if ReadNoise is not None:
        nimg = noise.gaussian(nimg, ReadNoise)

    if NumSimulations > 1:
        allSims = [nimg]
        for n in range(NumSimulations-1):
            if MaxPhotons is not None:
                nimg = noise.poisson(pimg, NPhot=MaxPhotons, seed=None, dtype='float32')  # Christian: default should be seed=0
            else:
                nimg = pimg
            if ReadNoise is not None:
                nimg = noise.gaussian(nimg, ReadNoise, seed=None)
            allSims = allSims.append(nimg)
        nimg = coordinates.catE(allSims)

    return nimg, psf, obj, pimg

