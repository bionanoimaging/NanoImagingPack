#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:22:51 2017

@author: Christian Karras, Rainer Heintzmann

This Package should contain funcitons in order to create psfs and otfs both 2-Dimensional and 3-Dimensional
 
Also SIM stuff
"""

import numpy as np
from .util import zernike, expanddim, midValAsg, zeros, shapevec, nollIdx2NM, abssqr, RWLSPoisson
from .transformations import rft,irft,ft2d
from .coordinates import rr, phiphi, px_freq_step, ramp1D, cosSinTheta
from .config import PSF_PARAMS, __DEFAULTS__
import numbers
import warnings
from scipy.special import j1
from .image import image, gaussf
import scipy.ndimage
import scipy.signal
# from .view5d import v5 # for debugging
import matplotlib.pyplot as plt
import matplotlib
import pathlib
import io
import ruamel.yaml


def getDefaultPSF_PARAMS(psf_params):
    if psf_params is None:
        return PSF_PARAMS()
    else:
        return psf_params

def __setPol__(im, psf_params= None):
    """
    Create the polarization maps (x and y), returned as tuple of 2 2D images based on the polarization parameters (pol, pol_xy_phase_shift, pol_lin_angle) of the psf parameter structure

    :param im: input image
    :param psf_params: psf parameter structure
    :return: returns the polarization map (tuple of 2 images with x and y polarization components)
    """
    psf_params = getDefaultPSF_PARAMS(psf_params)
    pol = [image(np.zeros(im.shape[-2:], dtype=np.complex128)),image(np.zeros(im.shape[-2:], dtype=np.complex128))]
    if psf_params.pol == 'elliptic':
        pol[0]+=1/np.sqrt(2)
        pol[1]+=1/np.sqrt(2)*np.exp(1j*psf_params.pol_xy_phase_shift)
    elif psf_params.pol == 'circular':
        pol[0] += 1/np.sqrt(2)
        pol[1] += 1/np.sqrt(2) * np.exp(1j * np.pi/2)
    elif psf_params.pol == 'lin':
        pol[0]+=np.cos(psf_params.pol_lin_angle)
        pol[1]+=np.sin(psf_params.pol_lin_angle)
    elif psf_params.pol == 'lin_x':
        pol[0]+=1
    elif psf_params.pol == 'lin_y':
        pol[1]+=1
    elif psf_params.pol == 'radial':
         phi = phiphi(im.shape)
         pol[0]+=  np.cos(phi)
         pol[1]+=  -np.sin(phi)
    elif psf_params.pol == 'azimuthal':
         phi = phiphi(im.shape)
         pol[0] += np.sin(phi)
         pol[1] += np.cos(phi)
    elif isinstance(psf_params.pol, list) or isinstance(psf_params.pol, tuple):
        if isinstance(psf_params.pol[0], np.ndarray) and isinstance(psf_params.pol[1], np.ndarray):
            if psf_params.pol[0].shape == im.shape[-2:] and psf_params.pol[1].shape == im.shape[-2:]:
                pol = psf_params.pol
            else:
                ValueError('Wrong Polarization setting -> give list or tuple of two arrays (x,y polarization maps) of xy shape like image ')
        else:
            ValueError('Wrong Polarization setting -> give list or tuple of two arrays (x,y polarization maps) of xy shape like image ')
    else:
        raise ValueError('Wrong Polarization: "lin", "lin_x", "lin_y", "azimuthal", "radial", "circular", "elliptic" or tuple or list of polarization maps (polx,poly)')
    return(tuple(pol))

def zernikeStack(im, psf_params=None, nzernikes=15):
    """
    ZERNIKESTACK creates a stack of Zernike base aberrations including the scalar pupil
    :param im: Input image
    :param psf_params: default PSF parameter
    :param nzernikes: number of stacks
    :return: a stack of Zernike base aberrations

    Example:
    import NanoImagingPack as nip
    from NanoImagingPack import v5
    im = nip.readim()
    out = nip.zernikeStack(im,nzernikes=3)
    v5(out)
    """

    psf_params = getDefaultPSF_PARAMS(psf_params)
    r = pupilRadius(im, psf_params)
    myzernikes = zeros((nzernikes, im.shape[-2], im.shape[-1]))  # + 1j * nip.zeros((nzernikes, Po.shape[-2], Po.shape[-1]))
    for noll in range(0, nzernikes):
        n, m = nollIdx2NM(noll + 1)
        myzernikes[noll, :, :] = zernike(r, m, n) * np.pi
    myzernikes.name = "Zernike Polynomials"
    return myzernikes

def aberrationMap(im, psf_params= None):
    """
    ABERRATIONMAP creates an aberration phase map (based on Zernike polynomials) for PSF generation
    :param im: Input Image
    :param psf_params: PSF paramter
    :return:
    """
    """
    

    uses:
        PSF_PARAMS().aberration_strength = None;
        PSF_PARAMS().aberration_types = None;
        PSF_PARAMS().aperture_transmission = None;
        returns the phase map (not the complex exponential!)

    strength:           strength of the aberration as multiples of 2pi in the phase (polynomial reach from -1 to 1)
    aberration_types:   can be
                            tuple (m, n) describing the Z^m_n polynomial
                        or phasemap

                        or string
                                piston     -> (Z00)
                                tiltY      -> (Z-11)
                                tiltX      -> (Z11)
                                astigm     -> (Z-22)
                                defoc      -> (Z02)
                                vastig     -> (Z22)
                                vtrefoil   -> (Z-33)
                                vcoma      -> (Z-13)
                                hcoma      -> (Z13)
                                obtrefoil  -> (Z33)
                                obquadfoil -> (Z-44)
                                asti2nd    -> (Z-24)
                                spheric    -> (Z04)
                                vasti2nd   -> (Z24)
                                vquadfoil  -> (Z44)

                    can be also lists or tuples -> Then everything will summed up
    """
    psf_params = getDefaultPSF_PARAMS(psf_params)
    zernike_para = {'piston': (0,0),'tiltY': (-1,1),'tiltX': (1,1),'astigm': (-2,2),'defoc': (0,2),'vastig': (2,2),'vtrefoil': (-3,3),'vcoma': (-1,3),
                    'hcoma': (1,3),'obtrefoil': (3,3),'obquadfoil': (-4,4),'asti2nd': (-2,4),'spheric': (0,4),'vasti2nd': (2,4),'vquadfoil': (4,4)}
    shape = shapevec(im)[-2:]   # works also for tensorflow objects
    aberration_map = zeros(shape)  # phase in radiants
    strength = psf_params.aberration_strength
    aberration = psf_params.aberration_types
    # transmission = psf_params.aperture_transmission
    #
    # if transmission is not None:
    #     if isinstance(transmission, np.ndarray):
    #         if transmission.shape == im.shape[-2:]:
    #             if np.min(transmission)<0:
    #                 warnings.warn('Transmission mask is negative at some values')
    #             if np.max(transmission)>1:
    #                 warnings.warn('Transmission mask is larger than one at some values')
    #             aberration_map*= transmission
    #         else:
    #             raise ValueError('Wrong dimension of transmission matrix')
    #     else:
    #         raise ValueError('Wrong transmission -> must be 2D image or array of the same xy-dimension as the image')
    if strength is not None and aberration is not None:
        # if isinstance(im, image):
        #     pxs=im.px_freq_step()[-2:]
        # else:
        #     pxs = __DEFAULTS__['IMG_PIXELSIZES']
        #     pxs = px_freq_step(im, pxs)[-2:]
        # r = rr(shape, scale=pxs) * psf_params.wavelength / psf_params.NA  # BAD !
        r = pupilRadius(im, psf_params)
        if isinstance(strength, numbers.Real):
            strength = [strength]
        if type(aberration) == str or isinstance(aberration,np.ndarray):
            aberration = [aberration]
        elif type(aberration) == list or type(aberration) == tuple:
            if isinstance(aberration[0], numbers.Integral):
                aberration= [aberration]
        for s, ab in zip(strength, aberration):
            if type(ab) == str:
                m = zernike_para[ab][0]
                n = zernike_para[ab][1]
                aberration_map += s*zernike(r, m, n)*np.pi
            elif isinstance(ab,np.ndarray):
                if ab.ndim > 2:
                    raise ValueError('Array map in aberration type needs to be 2D.')
                aberration_map += ab
            else:
                m = ab[0]
                n = ab[1]
                aberration_map += s*zernike(r, m, n)*np.pi
    aberration_map.name = "aberrated pupil phase"
    return aberration_map

def propagatePupil(pupil, sizeZ, distZ=None, psf_params = None, mode = 'Fourier', doDampPupil=False):
    """
    Propagates a pupil plane to a 3D stack of pupils.
    :param pupil: pupil amplitude to propagate
    :param sizeZ: Number of slices to propagate
    :param distZ: Distance between the slices (in units of the XY pixel information of the pupil)
    :param psf_params: Set of PSF parameters. Default: PSF_PARAMS()
    :param mode: mode of calculaton
    :param doDampPupil: should the damping around the edges be applied to the propagation?
    :return:

    Example: 100nm XY pixelsize and 1000nm spacing
    im = nip.readim()
    im.set_pixelsize((100,100))
    pupil = nip.jincAperture(im)
    out = nip.propagatePupil(pupil,10,1000.0)
    amp3d = nip.ift2d(out)

    see also jincAperture, pupilAperture
    """

    return pupil * propagationStack(pupil, sizeZ, distZ, psf_params, mode, doDampPupil)

def propagationStack(pupil, sizeZ, distZ=None, psf_params = None, mode = 'Fourier', doDampPupil=False):
    """
    generates a 3D stack in Fourier space containing the propagators. The pixelsize needs to be defined for the pupil (in real space pixels).
    :param pupil:    Fourier plane to propagate
    :param sizeZ:    Size to propagate along Z. XY sizes are inferred from the pupil size
    :param distZ:     pixelsize along Z for propagation
    :param psf_params:  structure of PSF parameters
    :param mode:
    :param doDampPupil:  defines wheter a damped pupil shall be used for the propagation
    :return:

    Example:
    import NanoImagingPack as nip
    from NanoImagingPack import v5
    im = nip.readim()
    im.set_pixelsize((100,100))
    nip.pupilAperture(im)
    pupil = nip.jincAperture(im)

    """
    psf_params = getDefaultPSF_PARAMS(psf_params)
    myshape = (sizeZ,) + shapevec(pupil)[-2:]
    return __make_propagator__(pupil, psf_params, doDampPupil, shape=myshape, distZ=distZ)

def __make_propagator__(im, psf_params = None, doDampPupil=False, shape=None, distZ=None):
    """
    Compute the field propagation matrix

    :param im:              input image
    :param psf_params:      psf params structure

                            TODO: Include chirp Z transform method
    :return:                Returns the phase propagation matrix (exp^i*delta_phi)
    """
    psf_params = getDefaultPSF_PARAMS(psf_params)

    pxs = im.pixelsize
    if shape is None:
        shape = shapevec(im)

    if pxs is None:
        raise ValueError("To propagate the pixelsize needs to be defined! Found: None")

    if len(shape) > 2:
        if (len(pxs)<3):
            if distZ is not None:
                axial_pxs = distZ
            else:
                raise ValueError("To propagate the pixelsize along Z needs to be defined or distZ needs to be used!")
                axial_pxs = __DEFAULTS__['IMG_PIXELSIZES'][0]
                warnings.warn('makePropagator: Only 2 Dimensional image given -> using default pixezsize[-3] = ('+str(axial_pxs)+')')
        else:
            axial_pxs = pxs[-3]
    else:
        axial_pxs = None

#    if axial_pxs is None:
#        raise ValueError("For propagation an axial pixelsize is needed. Use input.set_pixelsize().")

    if len(shape)>2:
        if axial_pxs is None:
            raise ValueError("For propagation an axial pixelsize is needed. Use input.set_pixelsize() or the parameter distZ.")
        cos_alpha, sin_alpha = cosSinAlpha(im, psf_params)
        defocus = axial_pxs * ramp1D(shape[-3], -3) # a series of defocus factors
        PhaseMap = defocusPhase(cos_alpha, defocus, psf_params)
        PhaseMap.pixelsize[-3] = axial_pxs
        if doDampPupil:
            return dampPupilForRealSpaceCut(PhaseMap) * np.exp(1j * PhaseMap)
        else:
            return np.exp(1j * PhaseMap)
    else:
        return 1.0

def pupilRadius(im, psf_params = None):
    """
        returns a map of the radius to the center of the pupil with respect to the pupil aperture edgre as defined by lambda, N, NA and the image size.

    :param im: image in Fourier space, defining the coordinates. This is given in any image with a set pixelsize
    :param psf_params: a structure of point spread function parameters. See SimLens for details
    :return: map of raltive distance from center to the edge of the pupil
    """
    psf_params = getDefaultPSF_PARAMS(psf_params)
    NA = psf_params.NA
    wl = psf_params.wavelength
    if isinstance(im, image):
        ft_pxs = im.px_freq_step()
    else:
        pxs = __DEFAULTS__['IMG_PIXELSIZES']
        ft_pxs = px_freq_step(im, pxs)
    if ft_pxs is not None:
        ft_pxs = ft_pxs[-2:]

    shape = shapevec(im)[-2:]   # works also for tensorflow objects

    res = rr(shape, scale=ft_pxs) * wl / NA  # [:2]
    res.__array_finalize__(im)
    return res

def pupilAperture(im, psf_params = None):
    """
    calulaltes a hard-edge pupil aperture by using pupilRadius<1.0
    :param im: image in Fourier space, defining the coordinates. This is given in any image with a set pixelsize
    :param psf_params: a structure of point spread function parameters. See SimLens for details
    :return: boolean aperture

    Example:
    import NanoImagingPack as nip
    im = nip.readim()
    ft_im = nip.ft2d(im)
    nip.pupilAperture(ft_im)

    See also: jincAperture(), psf_params()

    """
    psf_params = getDefaultPSF_PARAMS(psf_params)
    return pupilRadius(im,psf_params) < 1.0


def jincAperture(im, psf_params = None):
    """
    calulaltes a soft-edge pupil aperture by Fourier-transforming the real-space jinc solution of the problem.
    :param im: image in Fourier space, defining the coordinates. This is given in any image with a set pixelsize
    :param psf_params: a structure of point spread function parameters. See SimLens for details
    :return: boolean aperture

    Example:
    import NanoImagingPack as nip
    im = nip.readim()
    ft_im = nip.ft2d(im)
    nip.jincAperture(ft_im)

    See also: psf_params(), pupilAperture()
    """
    if im.pixelsize is None:
        raise ValueError("jincAperture needs the pixelsize to be present in the image. Please first set the pixelsize via img.set_pixelsize(value).")
    psf_params = getDefaultPSF_PARAMS(psf_params)
    shape = shapevec(im)[-2:]   # works also for tensorflow objects
    NA = psf_params.NA
    wl = psf_params.wavelength
    np.seterr(divide='ignore', invalid='ignore')
    arg = rr(shape, scale=np.array(im.pixelsize[-2:]) * 2 * np.pi * NA / wl)  # RH 3.2.19 was 2*mp.pi*np.sqrt((xx(shape)*self.pixelsize[0])**2+(yy(shape)*self.pixelsize[1])**2)*self.NA/self.wavelength;
    arg.pixelsize = im.pixelsize[-2:]
    ret = 2 * j1(arg) / arg  # that again creates the dip!
    ret[shape[-2] // 2, shape[-1] // 2] = 1
    ret = ret / np.sum(np.abs(ret))
    plane = ft2d(ret)
    plane.pixelsize = im.pixelsize
    np.seterr(divide='warn', invalid='warn')
    plane.name = "jincAperture"
    return plane

def cosSinAlpha(im, psf_params = None):
    """
    calculates the cos and sin of the angles (alpha) of beams (in real space) to the optical axis in the pupil plane
    :param im: image in Fourier space, defining the coordinates. This can be obtained by Fourier-transforming a real space image with a set pixelsize
    :param psf_params: a structure of point spread function parameters. See SimLens for details
    :return: a tuple of cos(alpha) and sin(alpha) images in the pupil plane

    Example:
    import NanoImagingPack as nip
    im = nip.readim()
    ft_im = nip.ft2d(im)
    nip.cosSinAlpha(ft_im)
    """
    psf_params = getDefaultPSF_PARAMS(psf_params)
    NA = psf_params.NA
    n = psf_params.n
    s = NA / n
    r = pupilRadius(im, psf_params)
    sin_alpha = s*r  # angle map sin
    sin_alpha[sin_alpha >= 1.0] = (1.0 - 1e-9)
    cos_alpha = np.sqrt(1-(sin_alpha)**2)  # angle map cos
    return cos_alpha, sin_alpha

def FresnelCoefficients(cos_alpha, sin_alpha, n1, n2):
    """
    returns a tuple of Fresnel coefficient maps (Ep/E0) and (Es/E0) when the light is transiting from refractive index n1 to n2.
    :param cos_alpha: the cosine of the angle of the incident beam towards the normal
    :param n1: refractive index of the medium of the incident beam
    :param n2: refractive index of the medium of the outgoing beam
    :return: ((Ep/E0p),(Es/E0s), cos_beta, sin_beta)  with E0p being the incident amplitude for the parallel polarisation
            and Ep being the outgoing amplitude. beta is the angle made by transmitted beam with normal

    Example:
    import NanoImagingPack as nip
    nip.FresnelCoefficiens(cos_alpha=1, sin_alpha=0, n1=1, n2=1.5)
    """
    sin_beta = n1 * sin_alpha / n2
    cos_beta = np.sqrt(1 - sin_beta**2)    # n1 sin_alpha = n2  sin_beta.
    numerator = 2.0*n1*cos_alpha
    F_s = numerator / (n1 * cos_alpha + n2 * cos_beta)
    F_p = numerator / (n2 * cos_alpha + n1 * cos_beta)
    return (F_s, F_p, cos_beta, sin_beta)

def dampPupilForRealSpaceCut(PhaseMap):
    """
    Considers the limited field-of-view effect in real space by dimming the higher frequencies in the pupil plane.
    :param PhaseMap:
    :return: an amplitude strength modification factor for the pupil and given defocus value(s)
    """
    # figure out phi: the relative phase change between neighboring pixels, estimated from cos_alpha
    dphiX = (np.roll(PhaseMap,1,axis=-1) - np.roll(PhaseMap,-1,axis=-1))/2.0
    dphiY = (np.roll(PhaseMap,1,axis=-2) - np.roll(PhaseMap,-1,axis=-2))/2.0
    damp = np.sinc(dphiX/2.0/np.pi)*np.sinc(dphiY/2.0/np.pi) # sin(pi x)/(pi x)
    return damp

def defocusPhase(cos_alpha, defocus=None, psf_params = None):
    """
    Calculates the complex-valued defocus phase propagation pattern for a given focus position. The calculation is accurate for high NA. However, it does not consider Fourier-space undersampling effects.
    :param cos_alpha: a map of cos(alpha) values in the pupil plane. Use "cosSinAlpha(im,psf_param)" to optain such a map
    :param defocus: real-space defocus value. if None is supplied, psf_params.off_focal_distance is used instead
    :param psf_params: a structure of point spread function parameters. See SimLens for details
    :return: a phase map (in rad) of phases for the stated defocus

    Example:
    import NanoImagingPack as nip
    im = nip.readim()
    cos_alpha, sin_alpha = nip.cosSinAlpha(im)
    nip.defocusPhase(cos_alpha) TODO : kindly check the result - MG
    """
    psf_params = getDefaultPSF_PARAMS(psf_params)
    if defocus is None:
        defocus = psf_params.off_focal_distance
    defocusPhase = psf_params.n / psf_params.wavelength * cos_alpha * defocus
    return 2.0 * np.pi * defocusPhase

def aplanaticFactor(cos_alpha, aplanar = 'emission'):
    """
    calculated the aplanatic factor according to the argument aplanar

    :param cos_alpha: A 2D image of cos(alpha). This is obtained by cosSinAlpha(image)
    :param aplanar:
        "emission": widefield emission in terms of irradiance (fluorpophore emission with random orientation)
        "excitation": laser scanning excitation in terms of irradiance (fluorophore excitation with random orientation)
        "emission2": widefield emission in terms of intensity (as emitted by a lambertian emitter oriented in XY)
        "excitation2": laser scanning excitation in terms of intensity (flux through a unit XY areaa at the object)
    :return:
    """
    # Apply aplanatic factor:
    CEps = 1e-4
    if aplanar is None:
        return 1.0
    elif aplanar == 'excitation':
        return np.sqrt(cos_alpha)
    elif aplanar == 'emission':
        apl = 1.0/np.sqrt(cos_alpha)
        apl[cos_alpha< CEps] = 1.0
        return apl
    elif aplanar == 'excitation2':
        return cos_alpha
    elif aplanar == 'emission2':
        apl = 1.0 / cos_alpha
        apl[cos_alpha < CEps] = 1.0
        return apl
    else:
        raise ValueError('Wrong aplanatic factor in PSF_PARAM structure:  "excitation", "emission","excitation2","emission2" or None, 2 means squared aplanatic factor for flux measurement')

def aberratedPupil(im, psf_params = None):
    """
    Compute a scalar aberrated pupil, possibly with the jinc-Trick. Aplanatic factors are included, but not the pupil propagation stack or the vectorial effects

    The following influences will be included:
        1.)     Mode of plane field generation (from sinc or circle)
        2.)     Aplanatic factor
        3.)     Potential aberrations (set by set_aberration_map)

    returns image of shape [3,y_im,x_im] where x_im and y_im are the xy dimensions of the input image. In the third dimension are the field components (0: Ex, 1: Ey, 2: Ez)
    """
    psf_params = getDefaultPSF_PARAMS(psf_params)

    cos_alpha, sin_alpha = cosSinAlpha(im, psf_params)

    if psf_params.aperture_method == 'jinc':
        plane = jincAperture(im, psf_params)
    elif psf_params.aperture_method == 'hard':
        aperture = pupilAperture(im, psf_params)
        plane = aperture+0.0j
    else:
        raise ValueError('Wrong aperture_method: ' + psf_params.aperture_method + ' in PSF_PARAM structure: Choices are: jinc and hard')
    # Apply aplanatic factor:
    if not (psf_params.aplanar is None):
        plane *= aplanaticFactor(cos_alpha, psf_params.aplanar)

    # Apply aberrations and apertures
    PhaseMap = 0;
    if not (psf_params.aberration_types is None):
        PhaseMap = PhaseMap + aberrationMap(im, psf_params)

    # Apply z-offset:
    PhaseMap += defocusPhase(cos_alpha, psf_params=psf_params)
    plane *= np.exp(1j*PhaseMap)  # psf_params.off_focal_distance is used as default
    return plane

def fLensmaker(R1=52,R2=np.inf, n=1.52, d = 10): # BK 7 at 520 nm
    """
    calculate the focal length f of a thick lens.
    :param R1: radius of curvature of the first surface
    :param R2: radius of curvature of the second surface (can be infinite: np.inf)
    :param n: refractive index. Default BK7 at 520nm
    :param d: thickness in the middle. Units are mm
    :return: focal length
    """
    f = 1.0/((n-1.0)*(1.0/R1-1.0/R2+(n-1)*d/(n*R1*R2)))
    return f

def simLens(im, psf_params = None):
    """
    Compute the 2D pupil-plane field (fourier transform of the electrical field in the plane (at position PSF_PARAMS.off_focal_dist))

    The following influences will be included:
        1.)     Mode of plane field generation (from sinc or circle)
        2.)     Aplanatic factor
        3.)     The losses induced by reflections at the coverslip (via Fresnel coefficients). Only if n_cs is not set to None.
        4.)     Potential aberrations (set by set_aberration_map)
        5.)     Vectorial effects

    returns image of shape [3,y_im,x_im] where x_im and y_im are the xy dimensions of the input image. In the third dimension are the field components (0: Ex, 1: Ey, 2: Ez)
    """
    psf_params = getDefaultPSF_PARAMS(psf_params)

    plane = aberratedPupil(im, psf_params)  # scalar aberrated (potentially defocussed) pupil
#    shape = shapevec(im)[-2:]   # works also for tensorflow objects


    if psf_params.n_embedding is None:
        n_embedding = psf_params.n
    else:
        n_embedding = psf_params.n_embedding

    if psf_params.vectorized:
        # expand to 4 Dims, the 4th will contain the electric fields
        plane = plane.cat([plane, plane], -4)
        plane.dim_description = ['Ex', 'Ey', 'Ez']  # {'d3': ['Ex', 'Ey', 'Ez']}

        # Apply vectorized distortions
        polx, poly = __setPol__(im, psf_params= psf_params)

        cos_alpha, sin_alpha = cosSinAlpha(im, psf_params)
        cos_theta, sin_theta = cosSinTheta(im)
        E_radial     = cos_theta * polx - sin_theta * poly
        E_tangential = sin_theta * polx + cos_theta * poly
        if psf_params.n_cs is not None:
            (Fs1, Fp1, cos_alpha2, sin_alpha2) = FresnelCoefficients(cos_alpha, sin_alpha, n_embedding, psf_params.n_cs)
            (Fs2, Fp2, cos_alpha3, sin_alpha3) = FresnelCoefficients(cos_alpha2, sin_alpha2, psf_params.n_cs, psf_params.n)
            E_radial *= Fp1 * Fp2
            E_tangential *= Fs1 * Fs2
        E_z = E_radial * sin_alpha
        E_radial *= cos_alpha
        plane[0] *=  cos_theta * E_radial + sin_theta * E_tangential
        plane[1] *= -sin_theta * E_radial + cos_theta * E_tangential
        plane[2] *= E_z
    else:
        # plane[0] *= polx
        # plane[1] *= poly
        # plane[2] *= 0
        pass # remain scalar

    plane.name = "simLens pupil"
    return plane # *aperture  # hard edge aperture at ?

def __make_transfer__(im, psf_params = None, mode = 'ctf', dimension = 2):
    """
    Creates the transfer function
    Also adds members PSF_PARAMS

    :param im:                  input image
    :param psf_params:          PSF Parameter struct
    :param mode:                'ctf', 'otf', 'psf', 'apsf'
    :param dimension:           if 2-> only 2D, if None like input image
    :return:                    returns the transfer function (dimension x,y, potentially z and field components for ctf and apsf

    ToDo: Examples
    """
    psf_params = getDefaultPSF_PARAMS(psf_params)

    ret = simLens(im, psf_params)
    if dimension is None:
        ret = ret * __make_propagator__(im, psf_params = psf_params, doDampPupil=False)  # field (kx,ky) propagated along the z - component
    if mode == 'ctf':  # Field transfer function is ft along z axis
        if ret.shape[-3] >1:
            ret = ret.ft(axes = -3)
    elif mode ==  'apsf':                # field spread function is ft along kx-ky-axes
        ret = ret.ift2d()
    elif mode == 'psf':                  # psf: like apsf and then abs square of field components
        ret = ret.ift2d()
        ret *= ret.conjugate()
        if ret.ndim > 3:
            ret = ret.sum(axis=-4)            # add all the electric field component intensities
        ret = np.real(ret)
        ret /= np.sum(ret)
    elif mode == 'otf':                  # otf: ft (over whole space) of psf
        ret = ret.ift2d()
        ret *= ret.conjugate()
        if ret.ndim > 3:
            ret = ret.sum(axis=-4)            # add all the electric field component intensities
        ret = np.real(ret)
        ret /= np.sum(ret)
        ret = ret.ft2d(norm=None)
    ret.PSF_PARAMS = psf_params
    ret.name = mode # label the name accordingly
    return ret

def otf(im, psf_params = None):
    """
        A set of functions to compute a transfer function of an objective
            psf         Point spread function
            apsf        Field transfer function
            otf         Optical transfer function
            ctf         Spectral transfer function of the field

        Parameters
        ----------
        :param im:           The input image. It should contain the parameter pixelsize. If not the default parameters from "nip.config" will be used
        :param psf_params:   The psf parameter struct. Use via modifying th "nip.PSF_PARAMS"
        :return:             Returns the respective transfer function The image will have the extra attibute "PSF_PARAMS"

        - If the image is three dimensional, the transfer function is three dimensional as well. 2D images will give 2D transfer functions.
        - 3D method is slice propagation method
        - Transfer functions can be forced to be 2D (even for 3D input images via using: psf2d, otf2d, apsf2d, ctf2d
        - All parameters will be set via using the PSF_PARAM struct (refere the help of that!)

        See also:
        ---------
        PSF_PARAMS(), psf, psf2d, otf, otf2d, apsf, apsf2d, ctf, ctf2d, simLens, setAberrationMap


        Example:
        -------
        - Create 2D PSF of 3d image

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS();
            psf = nip.psf2d(im, para);

         - Create 3D PSF with circular polarization:

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS();
            nip.para.pol = nip.para.pols.elliptical;
            psf = nip.psf(im, para);

         -  Create 2D OTF (because the image is 2D) and add some aberrations and a soft aperture

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            soft_aperture = nip.gaussian(im.shape, 10);         #Define some soft aperture
            aber_map = nip.xx(im).normalize(1);                # Define some aberration map (x-ramp in this case)
            para.aberration_types=[para.aberration_zernikes.spheric, (3,5), aber_map]    # define list of aberrations (select from choice, zernike coeffs, or aberration map
            para.aberration_strength =[0.5,0.8,0.2];
            otf = nip.otf(im, para);

        -   Create a non-Vectorial, emission APSF in 3D based on a circle in the Fourier space instead of a jinc in the real space

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            para.vectorized = False;
            para.aplanar = para.apl.emission
            para.foc_field_method = "circle"
            apsf = nip.apsf(im, para);

        Examples regarding the PSF_PARAMS():

        -   Set up linear polarization of arbitrary angle:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS();
            para.pol = "lin";           # you might as well choose via para.pol = para.pols.lin
            para.pol_lin_angle = np.pi*3/17;

        -   Set up elliptical polarization of arbitrary angle phase shift between x and y component:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS();
            para.pol = "elliptic";           # you might as well choose via para.pol = para.pols.lin
            para.pol_xy_phase_shift = np.pi*3/17;

        -   Set up x and y components of the polarization indendently:

            import NanoImagingPack as nip;
            import numpy as np;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            polx = nip.xx(im).normalize(1)*8.0;
            poly = np.random.rand(im.shape[-2], im.shape[-1])*np.exp(-2.546j);
            para.pol = [polx, poly]
    """
    psf_params = getDefaultPSF_PARAMS(psf_params)
    return(__make_transfer__(im, psf_params = psf_params, mode = 'otf', dimension = None))


def ctf(im, psf_params = None):
    """
        A set of functions to compute a transfer function of an objective
            psf         Point spread function
            apsf        Field transfer function
            otf         Optical transfer function
            ctf         Spectral transfer function of the field

        Parameters
        ----------
        :param im:           The input image. It should contain the parameter pixelsize. If not the default parameters from "nip.config" will be used
        :param psf_params:   The psf parameter struct. Use via modifying th "nip.PSF_PARAMS()"
        :return:             Returns the respective transfer function. The image will have the extra attibute "PSF_PARAMS"

        - If the image is three dimensional, the transfer function is three dimensional as well. 2D images will give 2D transfer functions.
        - 3D method is slice propagation method
        - Transfer functions can be forced to be 2D (even for 3D input images via using: psf2d, otf2d, apsf2d, ctf2d
        - All parameters will be set via using the PSF_PARAM struct (refere the help of that!)

        See also:
        ---------
        PSF_PARAMS, psf, psf2d, otf, otf2d, apsf, apsf2d, ctf, ctf2d, simLens, setAberrationMap


        Example:
        -------
        - Create 2D PSF of 3d image

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS();
            psf = nip.psf2d(im, para);

         - Create 3D PSF with circular polarization:

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS();
            nip.para.pol = nip.para.pols.elliptical;
            psf = nip.psf(im, para);

         -  Create 2D OTF (because the image is 2D) and add some aberrations and a soft aperture

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            soft_aperture = nip.gaussian(im.shape, 10);         #Define some soft aperture
            aber_map = nip.xx(im).normalize(1);                # Define some aberration map (x-ramp in this case)
            para.aberration_types=[para.aberration_zernikes.spheric, (3,5), aber_map]    # define list of aberrations (select from choice, zernike coeffs, or aberration map
            para.aberration_strength =[0.5,0.8,0.2];
            otf = nip.otf(im, para);

        -   Create a non-Vectorial, emission APSF in 3D based on a circle in the Fourier space instead of a jinc in the real space

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            para.vectorized = False;
            para.aplanar = para.apl.emission
            para.foc_field_method = "circle"
            apsf = nip.apsf(im, para);

        Examples regarding the PSF_PARAMS():

        -   Set up linear polarization of arbitrary angle:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS();
            para.pol = "lin";           # you might as well choose via para.pol = para.pols.lin
            para.pol_lin_angle = np.pi*3/17;

        -   Set up elliptical polarization of arbitrary angle phase shift between x and y component:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS();
            para.pol = "elliptic";           # you might as well choose via para.pol = para.pols.lin
            para.pol_xy_phase_shift = np.pi*3/17;

        -   Set up x and y components of the polarization indendently:

            import NanoImagingPack as nip;
            import numpy as np;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            polx = nip.xx(im).normalize(1)*8.0;
            poly = np.random.rand(im.shape[-2], im.shape[-1])*np.exp(-2.546j);
            para.pol = [polx, poly]
    """
    psf_params = getDefaultPSF_PARAMS(psf_params)
    return(__make_transfer__(im, psf_params = psf_params, mode = 'ctf', dimension = None))


def apsf(im, psf_params = None):
    """
        A set of functions to compute a transfer function of an objective
            psf         Point spread function
            apsf        Field transfer function
            otf         Optical transfer function
            ctf         Spectral transfer function of the field

        Parameters
        ----------
        :param im:           The input image. It should contain the parameter pixelsize. If not the default parameters from "nip.config" will be used
        :param psf_params:   The psf parameter struct. Use via modifying th "nip.PSF_PARAMS()"
        :return:             Returns the respective transfer function.  The image will have the extra attibute "PSF_PARAMS"

        - If the image is three dimensional, the transfer function is three dimensional as well. 2D images will give 2D transfer functions.
        - 3D method is slice propagation method
        - Transfer functions can be forced to be 2D (even for 3D input images via using: psf2d, otf2d, apsf2d, ctf2d
        - All parameters will be set via using the PSF_PARAM struct (refere the help of that!)

        See also:
        ---------
        PSF_PARAMS, psf, psf2d, otf, otf2d, apsf, apsf2d, ctf, ctf2d, simLens, setAberrationMap


        Example:
        -------
        - Create 2D PSF of 3d image

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS();
            psf = nip.psf2d(im, para);

         - Create 3D PSF with circular polarization:

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS();
            nip.para.pol = nip.para.pols.elliptical;
            psf = nip.psf(im, para);

         -  Create 2D OTF (because the image is 2D) and add some aberrations and a soft aperture

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            soft_aperture = nip.gaussian(im.shape, 10);         #Define some soft aperture
            aber_map = nip.xx(im).normalize(1);                # Define some aberration map (x-ramp in this case)
            para.aberration_types=[para.aberration_zernikes.spheric, (3,5), aber_map]    # define list of aberrations (select from choice, zernike coeffs, or aberration map
            para.aberration_strength =[0.5,0.8,0.2];
            otf = nip.otf(im, para);

        -   Create a non-Vectorial, emission APSF in 3D based on a circle in the Fourier space instead of a jinc in the real space

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            para.vectorized = False;
            para.aplanar = para.apl.emission
            para.foc_field_method = "circle"
            apsf = nip.apsf(im, para);

        Examples regarding the PSF_PARAMS:

        -   Set up linear polarization of arbitrary angle:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS();
            para.pol = "lin";           # you might as well choose via para.pol = para.pols.lin
            para.pol_lin_angle = np.pi*3/17;

        -   Set up elliptical polarization of arbitrary angle phase shift between x and y component:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS();
            para.pol = "elliptic";           # you might as well choose via para.pol = para.pols.lin
            para.pol_xy_phase_shift = np.pi*3/17;

        -   Set up x and y components of the polarization indendently:

            import NanoImagingPack as nip;
            import numpy as np;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            polx = nip.xx(im).normalize(1)*8.0;
            poly = np.random.rand(im.shape[-2], im.shape[-1])*np.exp(-2.546j);
            para.pol = [polx, poly]
    """
    psf_params = getDefaultPSF_PARAMS(psf_params)
    return __make_transfer__(im, psf_params = psf_params, mode = 'apsf', dimension = None)


def psf(im, psf_params = None):
    """
        A set of functions to compute a transfer function of an objective
            psf         Point spread function
            apsf        Field transfer function
            otf         Optical transfer function
            ctf         Spectral transfer function of the field

        Parameters
        ----------
        :param im:           The input image. It should contain the parameter pixelsize. If not the default parameters from "nip.config" will be used
        :param psf_params:   The psf parameter struct. Use via modifying th "nip.PSF_PARAMS()"
        :return:             Returns the respective transfer function.  The image will have the extra attibute "PSF_PARAMS"

        - If the image is three dimensional, the transfer function is three dimensional as well. 2D images will give 2D transfer functions.
        - 3D method is slice propagation method
        - Transfer functions can be forced to be 2D (even for 3D input images via using: psf2d, otf2d, apsf2d, ctf2d
        - All parameters will be set via using the PSF_PARAM struct (refere the help of that!)

        See also:
        ---------
        PSF_PARAMS, psf, psf2d, otf, otf2d, apsf, apsf2d, ctf, ctf2d, simLens, setAberrationMap


        Example:
        -------
        - Create 2D PSF of 3d image

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS();
            psf = nip.psf2d(im, para);

         - Create 3D PSF with circular polarization:

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS();
            nip.para.pol = nip.para.pols.elliptical;
            psf = nip.psf(im, para);

         -  Create 2D OTF (because the image is 2D) and add some aberrations and a soft aperture

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            soft_aperture = nip.gaussian(im.shape, 10);         #Define some soft aperture
            aber_map = nip.xx(im).normalize(1);                # Define some aberration map (x-ramp in this case)
            para.aberration_types=[para.aberration_zernikes.spheric, (3,5), aber_map]    # define list of aberrations (select from choice, zernike coeffs, or aberration map
            para.aberration_strength =[0.5,0.8,0.2];
            otf = nip.otf(im, para);

        -   Create a non-Vectorial, emission APSF in 3D based on a circle in the Fourier space instead of a jinc in the real space

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            para.vectorized = False;
            para.aplanar = para.apl.emission
            para.foc_field_method = "circle"
            apsf = nip.apsf(im, para);

        Examples regarding the PSF_PARAMS:

        -   Set up linear polarization of arbitrary angle:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS();
            para.pol = "lin";           # you might as well choose via para.pol = para.pols.lin
            para.pol_lin_angle = np.pi*3/17;

        -   Set up elliptical polarization of arbitrary angle phase shift between x and y component:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS();
            para.pol = "elliptic";           # you might as well choose via para.pol = para.pols.lin
            para.pol_xy_phase_shift = np.pi*3/17;

        -   Set up x and y components of the polarization indendently:

            import NanoImagingPack as nip;
            import numpy as np;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            polx = nip.xx(im).normalize(1)*8.0;
            poly = np.random.rand(im.shape[-2], im.shape[-1])*np.exp(-2.546j);
            para.pol = [polx, poly]
    """
    psf_params = getDefaultPSF_PARAMS(psf_params)
    return __make_transfer__(im, psf_params = psf_params, mode = 'psf', dimension = None)


def otf2d(im, psf_params = None):
    """
        A set of functions to compute a transfer function of an objective
            psf         Point spread function
            apsf        Field transfer function
            otf         Optical transfer function
            ctf         Spectral transfer function of the field

        Parameters
        ----------
        :param im:           The input image. It should contain the parameter pixelsize. If not the default parameters from "nip.config" will be used
        :param psf_params:   The psf parameter struct. Use via modifying th "nip.PSF_PARAMS()"
        :return:             Returns the respective transfer function.  The image will have the extra attibute "PSF_PARAMS"

        - If the image is three dimensional, the transfer function is three dimensional as well. 2D images will give 2D transfer functions.
        - 3D method is slice propagation method
        - Transfer functions can be forced to be 2D (even for 3D input images via using: psf2d, otf2d, apsf2d, ctf2d
        - All parameters will be set via using the PSF_PARAM struct (refere the help of that!)

        See also:
        ---------
        PSF_PARAMS, psf, psf2d, otf, otf2d, apsf, apsf2d, ctf, ctf2d, simLens, setAberrationMap


        Example:
        -------
        - Create 2D PSF of 3d image

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS();
            psf = nip.psf2d(im, para);

         - Create 3D PSF with circular polarization:

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS();
            nip.para.pol = nip.para.pols.elliptical;
            psf = nip.psf(im, para);

         -  Create 2D OTF (because the image is 2D) and add some aberrations and a soft aperture

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            soft_aperture = nip.gaussian(im.shape, 10);         #Define some soft aperture
            aber_map = nip.xx(im).normalize(1);                # Define some aberration map (x-ramp in this case)
            para.aberration_types=[para.aberration_zernikes.spheric, (3,5), aber_map]    # define list of aberrations (select from choice, zernike coeffs, or aberration map
            para.aberration_strength =[0.5,0.8,0.2];
            otf = nip.otf(im, para);

        -   Create a non-Vectorial, emission APSF in 3D based on a circle in the Fourier space instead of a jinc in the real space

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            para.vectorized = False;
            para.aplanar = para.apl.emission
            para.foc_field_method = "circle"
            apsf = nip.apsf(im, para);

        Examples regarding the PSF_PARAMS:

        -   Set up linear polarization of arbitrary angle:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS();
            para.pol = "lin";           # you might as well choose via para.pol = para.pols.lin
            para.pol_lin_angle = np.pi*3/17;

        -   Set up elliptical polarization of arbitrary angle phase shift between x and y component:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS();
            para.pol = "elliptic";           # you might as well choose via para.pol = para.pols.lin
            para.pol_xy_phase_shift = np.pi*3/17;

        -   Set up x and y components of the polarization indendently:

            import NanoImagingPack as nip;
            import numpy as np;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            polx = nip.xx(im).normalize(1)*8.0;
            poly = np.random.rand(im.shape[-2], im.shape[-1])*np.exp(-2.546j);
            para.pol = [polx, poly]
    """
    psf_params = getDefaultPSF_PARAMS(psf_params)
    return(__make_transfer__(im, psf_params = psf_params, mode = 'otf', dimension = 2).squeeze())


def ctf2d(im, psf_params = None):
    """
        A set of functions to compute a transfer function of an objective
            psf         Point spread function
            apsf        Field transfer function
            otf         Optical transfer function
            ctf         Spectral transfer function of the field

        Parameters
        ----------
        :param im:           The input image. It should contain the parameter pixelsize. If not the default parameters from "nip.config" will be used
        :param psf_params:   The psf parameter struct. Use via modifying th "nip.PSF_PARAMS()"
        :return:             Returns the respective transfer function.  The image will have the extra attibute "PSF_PARAMS"

        - If the image is three dimensional, the transfer function is three dimensional as well. 2D images will give 2D transfer functions.
        - 3D method is slice propagation method
        - Transfer functions can be forced to be 2D (even for 3D input images via using: psf2d, otf2d, apsf2d, ctf2d
        - All parameters will be set via using the PSF_PARAM struct (refere the help of that!)

        See also:
        ---------
        PSF_PARAMS, psf, psf2d, otf, otf2d, apsf, apsf2d, ctf, ctf2d, simLens, setAberrationMap


        Example:
        -------
        - Create 2D PSF of 3d image

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS();
            psf = nip.psf2d(im, para);

         - Create 3D PSF with circular polarization:

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS();
            nip.para.pol = nip.para.pols.elliptical;
            psf = nip.psf(im, para);

         -  Create 2D OTF (because the image is 2D) and add some aberrations and a soft aperture

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            soft_aperture = nip.gaussian(im.shape, 10);         #Define some soft aperture
            aber_map = nip.xx(im).normalize(1);                # Define some aberration map (x-ramp in this case)
            para.aberration_types=[para.aberration_zernikes.spheric, (3,5), aber_map]    # define list of aberrations (select from choice, zernike coeffs, or aberration map
            para.aberration_strength =[0.5,0.8,0.2];
            otf = nip.otf(im, para);

        -   Create a non-Vectorial, emission APSF in 3D based on a circle in the Fourier space instead of a jinc in the real space

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            para.vectorized = False;
            para.aplanar = para.apl.emission
            para.foc_field_method = "circle"
            apsf = nip.apsf(im, para);

        Examples regarding the PSF_PARAMS:

        -   Set up linear polarization of arbitrary angle:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS();
            para.pol = "lin";           # you might as well choose via para.pol = para.pols.lin
            para.pol_lin_angle = np.pi*3/17;

        -   Set up elliptical polarization of arbitrary angle phase shift between x and y component:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS();
            para.pol = "elliptic";           # you might as well choose via para.pol = para.pols.lin
            para.pol_xy_phase_shift = np.pi*3/17;

        -   Set up x and y components of the polarization indendently:

            import NanoImagingPack as nip;
            import numpy as np;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            polx = nip.xx(im).normalize(1)*8.0;
            poly = np.random.rand(im.shape[-2], im.shape[-1])*np.exp(-2.546j);
            para.pol = [polx, poly]
    """
    psf_params = getDefaultPSF_PARAMS(psf_params)
    return(__make_transfer__(im, psf_params = psf_params, mode = 'ctf', dimension = 2).squeeze())


def apsf2d(im, psf_params = None):
    """
        A set of functions to compute a transfer function of an objective
            psf         Point spread function
            apsf        Field transfer function
            otf         Optical transfer function
            ctf         Spectral transfer function of the field

        Parameters
        ----------
        :param im:           The input image. It should contain the parameter pixelsize. If not the default parameters from "nip.config" will be used
        :param psf_params:   The psf parameter struct. Use via modifying th "nip.PSF_PARAMS()"
        :return:             Returns the respective transfer function.  The image will have the extra attibute "PSF_PARAMS"

        - If the image is three dimensional, the transfer function is three dimensional as well. 2D images will give 2D transfer functions.
        - 3D method is slice propagation method
        - Transfer functions can be forced to be 2D (even for 3D input images via using: psf2d, otf2d, apsf2d, ctf2d
        - All parameters will be set via using the PSF_PARAM struct (refere the help of that!)

        See also:
        ---------
        PSF_PARAMS, psf, psf2d, otf, otf2d, apsf, apsf2d, ctf, ctf2d, simLens, setAberrationMap


        Example:
        -------
        - Create 2D PSF of 3d image

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS();
            psf = nip.psf2d(im, para);

         - Create 3D PSF with circular polarization:

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS();
            nip.para.pol = nip.para.pols.elliptical;
            psf = nip.psf(im, para);

         -  Create 2D OTF (because the image is 2D) and add some aberrations and a soft aperture

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            soft_aperture = nip.gaussian(im.shape, 10);         #Define some soft aperture
            aber_map = nip.xx(im).normalize(1);                # Define some aberration map (x-ramp in this case)
            para.aberration_types=[para.aberration_zernikes.spheric, (3,5), aber_map]    # define list of aberrations (select from choice, zernike coeffs, or aberration map
            para.aberration_strength =[0.5,0.8,0.2];
            otf = nip.otf(im, para);

        -   Create a non-Vectorial, emission APSF in 3D based on a circle in the Fourier space instead of a jinc in the real space

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            para.vectorized = False;
            para.aplanar = para.apl.emission
            para.foc_field_method = "circle"
            apsf = nip.apsf(im, para);

        Examples regarding the PSF_PARAMS:

        -   Set up linear polarization of arbitrary angle:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS();
            para.pol = "lin";           # you might as well choose via para.pol = para.pols.lin
            para.pol_lin_angle = np.pi*3/17;

        -   Set up elliptical polarization of arbitrary angle phase shift between x and y component:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS();
            para.pol = "elliptic";           # you might as well choose via para.pol = para.pols.lin
            para.pol_xy_phase_shift = np.pi*3/17;

        -   Set up x and y components of the polarization indendently:

            import NanoImagingPack as nip;
            import numpy as np;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            polx = nip.xx(im).normalize(1)*8.0;
            poly = np.random.rand(im.shape[-2], im.shape[-1])*np.exp(-2.546j);
            para.pol = [polx, poly]
    """
    psf_params = getDefaultPSF_PARAMS(psf_params)
    return(__make_transfer__(im, psf_params = psf_params, mode = 'apsf', dimension = 2).squeeze())


def psf2d(im, psf_params = None):
    """
        A set of functions to compute a transfer function of an objective
            psf         Point spread function
            apsf        Field transfer function
            otf         Optical transfer function
            ctf         Spectral transfer function of the field

        Parameters
        ----------
        :param im:           The input image. It should contain the parameter pixelsize. If not the default parameters from "nip.config" will be used
        :param psf_params:   The psf parameter struct. Use via modifying th "nip.PSF_PARAMS()"
        :return:             Returns the respective transfer function.  The image will have the extra attibute "PSF_PARAMS"

        - If the image is three dimensional, the transfer function is three dimensional as well. 2D images will give 2D transfer functions.
        - 3D method is slice propagation method
        - Transfer functions can be forced to be 2D (even for 3D input images via using: psf2d, otf2d, apsf2d, ctf2d
        - All parameters will be set via using the PSF_PARAM struct (refere the help of that!)

        See also:
        ---------
        PSF_PARAMS, psf, psf2d, otf, otf2d, apsf, apsf2d, ctf, ctf2d, simLens, setAberrationMap


        Example:
        -------
        - Create 2D PSF of 3d image

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS();
            psf = nip.psf2d(im, para);

         - Create 3D PSF with circular polarization:

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS();
            nip.para.pol = nip.para.pols.elliptical;
            psf = nip.psf(im, para);

         -  Create 2D OTF (because the image is 2D) and add some aberrations and a soft aperture

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            soft_aperture = nip.gaussian(im.shape, 10);         #Define some soft aperture
            aber_map = nip.xx(im).normalize(1);                # Define some aberration map (x-ramp in this case)
            para.aberration_types=[para.aberration_zernikes.spheric, (3,5), aber_map]    # define list of aberrations (select from choice, zernike coeffs, or aberration map
            para.aberration_strength =[0.5,0.8,0.2];
            otf = nip.otf(im, para);

        -   Create a non-Vectorial, emission APSF in 3D based on a circle in the Fourier space instead of a jinc in the real space

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            para.vectorized = False;
            para.aplanar = para.apl.emission
            para.foc_field_method = "circle"
            apsf = nip.apsf(im, para);

        Examples regarding the PSF_PARAMS:

        -   Set up linear polarization of arbitrary angle:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS();
            para.pol = "lin";           # you might as well choose via para.pol = para.pols.lin
            para.pol_lin_angle = np.pi*3/17;

        -   Set up elliptical polarization of arbitrary angle phase shift between x and y component:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS();
            para.pol = "elliptic";           # you might as well choose via para.pol = para.pols.lin
            para.pol_xy_phase_shift = np.pi*3/17;

        -   Set up x and y components of the polarization indendently:

            import NanoImagingPack as nip;
            import numpy as np;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS();
            polx = nip.xx(im).normalize(1)*8.0;
            poly = np.random.rand(im.shape[-2], im.shape[-1])*np.exp(-2.546j);
            para.pol = [polx, poly]
    """
    psf_params = getDefaultPSF_PARAMS(psf_params)
    return __make_transfer__(im, psf_params = psf_params, mode = 'psf', dimension = 2).squeeze()

def jinc(mysize=[256,256],myscale=None):
    """
    Caculates a bessel(1,2*pi*radius)/radius   = jinc function, which describes the Airy pattern in scalar low NA approximation

    Example 1:
    pixelSize=203;  # half the Nyquist freq
    mysize=[256,256];
    lambda=488/pixelSize;
    na=0.3;
    AbbeLimit=lambda/na;   # coherent Abbe limit, central illumination, not incoherent
    ftradius=1/AbbeLimit*mysize;
    myscales=ftradius./mysize; # [100 100]/(488/0.3);
    res=jinc(mysize,myscales);  # Airy pattern (low NA) for these value (at n=1.0)

    Example 2: jinc such that the Fourier transformation has a defined radius myrad. E.g. needed for confocal pinholes
    mysize=[256,256];
    ftradius=10.0;  # pixels in Fourier space (or real space, if starting in Fourier space)
    myscales=ftradius./mysize;
    ftpinhole=jinc(mysize,myscales);  # Fourier pattern for pinhole with radius ftradius
    ftpinhole=ftpinhole/ftpinhole(MidPosX(ftpinhole),MidPosY(ftpinhole))*pi*ftradius.^2;  # normalize
    pinhole=real(ft(ftpinhole))/sqrt(prod(mysize))  # normlized to one in the disk
    """
    from scipy.special import j1
    if myscale is None:
        pixelSize=203  # half the Nyquist freq
        mylambda=488/pixelSize
        na=0.3
        AbbeLimit=mylambda/na  # coherent Abbe limit, central illumination, not incoherent
        ftradius=1/AbbeLimit*mysize
        myscales=ftradius/mysize  # [100 100]/(488/0.3);
    myradius=np.pi*rr(mysize,scale=myscale)
    midValAsg(myradius, 1.0)
    res=j1(2*myradius) /  myradius #  where(myradius == 0, 1.0e-20, myradius)
    midValAsg(res, 1.0)
    res.name = "jinc"
    return res

def PSF2ROTF(psf):
    """
        Transforms a real-valued PSF to a half-complex RFT, at the same time precompensating for the fftshift
    """
    o = image(rft(psf, shift_before=True))  # accounts for the PSF to be placed correctly
    o = o/np.max(o)
    o.name = "rotf"
    return o.astype('complex64')

# this should go into the NanoImagingPack
def convROTF(img,otf): # should go into nip
    """
        convolves with a half-complex OTF, which can be generated using PSF2ROTF
    """
    res = irft(rft(img, shift_before=False, shift_after=False) * expanddim(otf, img.ndim),shift_before=False, shift_after=False, s = img.shape)
    res.name = "convolved"
    return res

def removePhaseInt(pulse):
    """
    normalizes the phase slope to zero. Useful to visually compare results
    :param pulse: input pulse to normalize
    :return: curve with phase slope at the max intensity set to zero
    """
    pulse = np.squeeze(pulse)
    Intensity = abssqr(pulse)
    idx = np.argmax(Intensity)
    # pulse = pulse / np.sqrt(np.max(Intensity))
    phase0 = np.angle(pulse[idx % pulse.size])
    deltaPhase = np.angle(pulse[idx+1] / pulse[idx])
    return pulse * np.exp(-1j * (phase0 + deltaPhase*(np.arange(pulse.size)-idx)))


def cal_readnoise(fg, bg, numBins:int=100, validRange=None, linearity_range=None, histRange=None, CameraName=None, correctBrightness:bool=True,
 correctOffsetDrift:bool=True, exclude_hot_cold_pixels:bool=True, noisy_pixel_percentile:float=98, doPlot:bool=True, exportpath:pathlib.Path=None, exportFormat:str="png",
 brightness_blurring:bool=True, plotWithBgOffset:bool=True, plotHist:bool=False, check_bg:bool=False, saturationImage:bool=False):
    """
    Calibrates detectors by fitting a straight line through a mean-variance plot. To scale your images into photons use: (image-offset) * gain

    :param fg: A series of foreground images of the same (blurry) scene. Ideally spanning the useful range of the detector. Suggested number: 20 images
    :param bg: A series of dark images under identical camera settings (integration time, temperature, etc.). Suggested number: 20 images
    :param numBins: Number of bins for the histogram 
    :param histRange: If provided, the histogram will only be generated over this range.
    :param validRange: If provided, the gain fit will only be performed over this range of mean values.
    :param linearity_range: If provided, the linearity will only be evaluated over this range.
    :param CameraName: If provided, sets a plot title with CameraName
    :param correctBrightness: Attempt to correct the calibration for a fluctuating illumination brightness.
    :param correctOffsetDrift:
    :param exclude_hot_cold_pixels: Exclude hot and cold pixels from the fit
    :param noisy_pixel_percentile: Only include this percentile of least noisy pixels.
                                   Useful for supressing the effect of RTS noise. 
    :param doPlot: Plot the mean-variance curves
    :param exportpath: If provided, the plots will be saved to this directory.
    :param exportFormat: PNG or SVG files possible.
    :param brightness_blurring: A filter to blur the brightness estimate. Useful for sCMOS sensors
    :param plotWithBgOffset: If false, then the background value will be subtracted from the pixel ADUs in the plot.
    :param plotHist: If true, then a histogram of brightness bins will be plotted in the plot background
    :param check_bg: If true, then the background images will be checked.
    :param saturationImage: If true, then the peak of the photon transfer curve will be used to estimate the saturation level and calculate a dynamic range.
    :return: tuple of fit results (offset [adu], gain [electrons / adu], readnoise [e- RMS])
    """
    function_args = locals()

    if exportpath is not None:
        exportpath = pathlib.Path(exportpath)
        exportpath.mkdir(parents=True, exist_ok=True)

    figures = [] # collect figures for returning
    # define plotting parameters
    CM = 1/2.54 # centimetres for figure sizes
    plot_figsize = 26*CM, 15*CM # works well for screens
    AxisFontSize=12
    TextFontSize=8
    TitleFontSize=14
    # rc('font', size=AxisFontSize)  # controls default text sizes
    # rc('axes', titlesize=AxisFontSize, labelsize=AxisFontSize)  # fontsize of the axes title and number labels
    
    #-----------------------------
    # before converting to floats, generate histograms. Not needed for gain calculation, but useful diagnosis information
    fg_max_ADU = fg.max()
    hist_bins = range(0, int(bg.max()))
    dark_counts, dark_hist_bin_edges = np.histogram(bg, bins=hist_bins)
    dark_counts = dark_counts/bg.shape[0]
    hist_bins = range(0, int(fg.max()))
    bright_counts, bright_hist_bin_edges = np.histogram(fg, bins=hist_bins)
    bright_counts = bright_counts/fg.shape[0]
    #-----------------------------

    
    fg = np.squeeze(fg).astype(float)
    bg = np.squeeze(bg).astype(float)

    total_pixels = np.prod(fg.shape[1:])

    assert fg.sum() > bg.sum(), "Error, background image is brighter than foreground image.\nAnalysis will fail"

    if check_bg:
        THRESHOLD = 10
        assert check_dark(bg, threshold=THRESHOLD), f"Warning, dark image standard deviation is more than {THRESHOLD} times larger than per-pixel standard deviation. \nVerify that dark image is flat and contains no structure"

    didReduce = False
    if fg.ndim > 3:
        print('WARNING: Foreground data has more than 3 dimensions. Just taking the first of the series.')
        fg = fg[0]
        didReduce = True
    if bg.ndim > 3:
        print('WARNING: Background data has more than 3 dimensions. Just taking the first of the series.')
        bg = bg[0]
        didReduce = True
    fg = np.squeeze(fg)
    bg = np.squeeze(bg)
    if didReduce:
        # return cal_readnoise(fg,bg,numBins, validRange, CameraName, correctBrightness, correctOffsetDrift, excludeHotPixels, doPlot)
        # TODO: Needs testing
        return cal_readnoise(fg,bg, *list(function_args.values()))

    doc_dict = {}
    doc_dict["Dark images"] = (f"{bg.shape[0]}",
                            "Number of dark images used for calibration"
    )      
    doc_dict["Bright images"] = (f"{fg.shape[0]}",
                            "Number of bright images used for calibration"
    )      

    validmap = np.ones(bg.shape[-2:], dtype=bool)
    # validmap = None
    if validRange is None:
        # Underflow
        tmp_bypass = True

        if not tmp_bypass:
            print("FooFoo")
            underflow = np.sum(fg <= 0, (0,)) > 0
            validmap *= ~underflow
            numUnderflow = np.sum(underflow)
            if np.min(fg) <= 0:
                print("WARNING: "+str(numUnderflow)+" pixels with at least one zero-value (value<=0) were detected but no fit range was selected. Excluded those from the fit.")
            doc_dict["Zero-value pixels excluded"] = (f"{numUnderflow} ({numUnderflow/total_pixels:.1%})",
                                    "Pixels with at least one zero-value. Zero-valued pixels should not occur in the data, as it suggests clipped results. If any pixel has a zero-valued sample, the entire pixel is excluded from the analysis. If this value is high, the black level offset should be adjusted."
            )          
            
            overflow = 0
            numsat = 0
            relsat = 0
            maxvalstr = ""
            for MaxVal in [255,1023,4095,65535]: #TODO: not robust. Many cameras saturate at some other value
                if np.max(fg) == MaxVal:
                    overflow = np.sum(fg == MaxVal,(0,)) > 0
                    numsat = np.sum(overflow)
                    relsat = np.sum(overflow)/total_pixels
                    print("WARNING: "+str(numsat)+" pixels saturating at least once (value=="+str(MaxVal)+") were detected but no fit range was selected. Excluding those from the fit.")
                    validmap = validmap & ~ overflow
                    maxvalstr = "(=="+str(MaxVal)+")"
            doc_dict["Overflow pixels excluded"] = (f"{numsat} ({relsat:.1%})",
                                    "Max vals one of [255,1023,4095,65535]"
            )           

    # if clipRange is not None:
    #     validmap *= (np.sum(fg >= clipRange[0], axis=0) + np.sum(fg <= clipRange[1], axis=0)) > 0

    if correctOffsetDrift:
        meanoffset = np.mean(bg, (-2,-1), keepdims=True)
        refOffset = np.mean(bg)
        bg = bg - meanoffset + refOffset
        reloffset = (refOffset - meanoffset)  / np.sqrt(np.var(bg))
        if doPlot:
            fig = plt.figure(figsize=plot_figsize)
            if CameraName is not None:
                plt.title("Offset Drift ("+CameraName+")", fontsize=TitleFontSize)
            else:
                plt.title("Offset Drift", fontsize=TitleFontSize)
            plt.plot(reloffset.flat, label='Offset / Std.Dev.')
            plt.xlabel("frame no.", fontsize=AxisFontSize)
            plt.ylabel("mean offset / Std.Dev.", fontsize=AxisFontSize)

            figures.append((fig, "correctOffsetDrift"))

    # add ability to use only one backgroound image
    if bg.ndim == 2:
        bg_mean_projection = np.mean(bg)
    else:
        bg_mean_projection = np.mean(bg, (-3))
    bg_total_mean = float(np.mean(bg_mean_projection)) # don't want to return image type
    plotOffset = bg_total_mean*plotWithBgOffset
    patternVar = np.var(bg_mean_projection)

# POTENTIAL FOR ERROR CROSSING THIS LINE
    fg -= bg_mean_projection # pixel-wise background subtraction
    if correctBrightness:
        brightness = np.mean(fg, (-2,-1), keepdims=True)
        meanbright = np.mean(brightness)
        relbright = brightness/meanbright
        if doPlot:
            fig = plt.figure(figsize=plot_figsize)
            if CameraName is not None:
                plt.title("Brightness Fluctuation ("+CameraName+")", fontsize=TitleFontSize)
            else:
                plt.title("Brightness Fluctuation", fontsize=TitleFontSize)
            plt.plot(relbright.flat)
            plt.xlabel("frame no.",fontsize=AxisFontSize)
            plt.ylabel("relative brightness",fontsize=AxisFontSize)
            figures.append((fig, 'Brightness_Fluctuation'))
        fg = fg / relbright
        maxFluc = np.max(np.abs(1.0-relbright))
        doc_dict["Illumination fluctuation"] = (f"{maxFluc:.2%}",
                                "Illumination fluctuation for bright images. The fluctuation in the total amount of light between the bright frames. If this value is high, it suggests an unstable light source or possibly problems with the detector's acquisition. "
        )        
    fg_mean_projection = np.mean(fg, (-3)) 
    fg_var_projection = np.var(fg, (-3), ddof=1)
    # for single bg image
    if bg.ndim ==2:
        bg_var_projection = np.var(bg)
    else:
        bg_var_projection = np.var(bg, (-3), ddof=1)

    hotPixels = None
    if exclude_hot_cold_pixels:
        hotPixels = np.abs(bg_mean_projection - np.mean(bg_mean_projection)) > 4.0*np.sqrt(np.mean(bg_var_projection))
        numHotPixels = np.sum(hotPixels)
        doc_dict["Hot or Cold pixels excluded"] = (f"{numHotPixels:d} ({numHotPixels/total_pixels:.1%})",
                                "Total number of excluded hot or cold pixels. Pixels are considered hot or cold if their mean value in the background image is more than 4 standard deviations removed from the background values of all background image pixels."
        )
        validmap *= ~hotPixels


    noisyPixelThreshold = np.percentile(bg_var_projection, noisy_pixel_percentile)
    noisyPixels = bg_var_projection > noisyPixelThreshold # need to exclude hot pixels, which skew variance
    validmap *= ~noisyPixels

    # if validRange is given, it is for for biased image
    # we need to correct it for the unbiased image
    if validRange is not None: 
        validRange = np.array(validRange)
        validRange = validRange - bg_total_mean
    if linearity_range is not None: 
        linearity_range = np.array(linearity_range)
        linearity_range = linearity_range - bg_total_mean

    if brightness_blurring:
        # sCMOS brightnesses fluctuate too much, we need a filter
        # blur image, yielding better estimate for local brightness
        blurred = fg_mean_projection
        # Generate median projection to use to fill gaps from invalid pixels 2021-04 
        median_projection = scipy.ndimage.median_filter(fg_mean_projection, size=(7,7))
        blurred[~validmap] = median_projection[~validmap]
        blurred = gaussf(blurred, (7,7))
        fg_mean_projection = blurred[validmap] # Note that this also excludes the invalid pixels from the plot
    else:
        fg_mean_projection = fg_mean_projection[validmap]

    fg_var_projection = fg_var_projection[validmap]
    # bg_var_projection = bg_var_projection[validmap] # this is unnecessary and would falsify the readnoise estimate

    # create histRange, otherwise numpy.histogram will allocate bins right up to the hot pixels
    # validMeans = fg_mean_projection[validmap] # now that it is applied above, we don't need to use validmap here
    validMeans = fg_mean_projection
    if histRange is None:
        histRange = (np.min(validMeans), np.max(validMeans))
    elif histRange is not None:
        histRange = np.array(histRange)
        histRange = histRange - bg_total_mean


    # Bin and create histograms.
    # 
    # hist_num: total number of pixels in the bin
    # hist_mean_sum: sum of all the mean values of pixels in the bin
    # hist_var_sum: sum of all the variance values of pixels in the bin
    (hist_num, mybins) = np.histogram(fg_mean_projection, range=histRange, bins=numBins)
    (hist_mean_sum, mybins) = np.histogram(fg_mean_projection, range=histRange, bins=numBins, weights=fg_mean_projection)
    (hist_var_sum, mybins) = np.histogram(fg_mean_projection, range=histRange, bins=numBins, weights=fg_var_projection)

    binMid = (mybins[1:] + mybins[:-1]) / 2


    valid = hist_num > 0
    hist_var_sum = hist_var_sum[valid]
    hist_mean_sum = hist_mean_sum[valid]
    hist_num = hist_num[valid]
    binMid = binMid[valid]

    mean_var = hist_var_sum / hist_num # mean of variances within the bin
    mean_mean = hist_mean_sum / hist_num # mean of means within the bin

    
    if saturationImage:
        # var_peak_ix = scipy.signal.find_peaks(x=mean_var, distance=5, width=4)[0][0]
        var_peak_ix = np.argmax(mean_var)
        var_peak = binMid[var_peak_ix] #determine saturation capacity

    # Automatic range finding
    if saturationImage:
        # EMVA uses 0-70% of sat peak for gain fit and 5-95% sat peak for linearity fit. But measurement is different.
        # For unified range, 0-95% seems sensible
        if validRange is None:
            validRange = (0, 0.95*var_peak) # gain fit
        if linearity_range is None:
            linearity_range = (0, 0.95*var_peak) #for linearity fit
    else: # regular autofind
        if validRange is None:
            # percentlies
            validRange = [0, np.percentile(validMeans, 99)]
        if linearity_range is None:
            # linearity_range = (0.05*validRange[0], 0.95*validRange[1]) #for linearity fit
            linearity_range = validRange
            # percentiles

    # gain fit
    (offset, slope, variance_of_variances) = RWLSPoisson(mean_mean, mean_var, hist_num, 
                                                            validRange=validRange)

    linearity_slice = (mean_mean > linearity_range[0]) * (mean_mean < linearity_range[1])
    (lin_offset, lin_slope, lin_variance_of_variances) = RWLSPoisson(mean_mean, mean_var, 
                                                            hist_num, validRange=linearity_range)
    linfit_x = mean_mean[linearity_slice]
    linvar_fits = (lin_slope*linfit_x + lin_offset)
    dev = mean_var[linearity_slice] - linvar_fits
    rel_dev = dev/linvar_fits
    #extended linearity deviation
    linfit_x_extended = mean_mean
    linvar_fits_extended = (lin_slope*linfit_x_extended + lin_offset)
    dev_extended = mean_var - linvar_fits_extended
    rel_dev_extended = dev_extended/linvar_fits_extended

    myFit = binMid * slope + offset
    myStd = np.sqrt(variance_of_variances)
    gain = float(1.0 / slope) # don't want to return image type
    mean_el_per_exposure = np.sum(fg.mean((-3)))*gain

    # bright image dynamic range
    # bin_lims = binMid[[0,-1]]-bg_total_mean
    bin_lims = binMid[[0,-1]]
    # we want brightest to darkest

    image_dyn_range = bin_lims[1]/bin_lims[0]
    doc_dict["Bright image dynamic range [factor]"] = (f"{image_dyn_range:.4g}",
                                    "Bright image dynamic range, expressed as a factor. It represents the difference between the darkest and brightest parts of the bright calibration images. Above 20 is good."
    )
    doc_dict["Noisy pixels excluded"] = (f"{100-noisy_pixel_percentile:.0f}%",
                                    "Percentage of pixels which were excluded. By default, 2% of the noisiest background pixels are excluded from the analysis. This deals with RTS noise in CMOS cameras. CCD and scanning imagers should be unaffected."
    )
    doc_dict["Noise exclusion threshold [e-]"] = (f"{np.sqrt(noisyPixelThreshold)*gain:.1f}",
                                    "Threshold for exclusion for noisy pixels. Units are electrons."
    )
    doc_dict["Background [ADU]"] = ("{bg:.2f}".format(bg=bg_total_mean),
                                    "The background, or black level of the signal, obtained by the mean pixel value of the dark exposures. Units are ADU."
    )
    doc_dict["Gain [e- / ADU]"] = (f"{gain:.4f}",
                                    "Conversion factor (Gain) determined by fit to photon transfer curve."
    )
    Readnoise = (np.std(bg, (-3), ddof=1).mean() * gain).astype(float) # DDOF necessary, Bessel's correction
    doc_dict["Readnoise, RMS [e-]"] = (f"{Readnoise:.2f}",
                                        "The mean readnoise, calculated by multiplying the mean of the individual pixel standard deviations with the gain. Units are photoelectrons."
    )
    median_readnoise = (np.sqrt(np.median(bg_var_projection)) * gain).astype(float) # don't want to return image type
    median_readnoise = (np.median(np.std(bg, (-3), ddof=1)) * gain).astype(float) # new version try
    doc_dict["Readnoise, median  [e-]"] = (f"{median_readnoise:.2f}",
                                        "The median readnoise, which is a commonly quoted metric for CMOS cameras. Units are electrons."
    )
    if saturationImage:
        snr = var_peak/np.sqrt(np.mean(bg_var_projection))
        doc_dict["Detector dynamic range [factor]"] = (f"{snr:.0f}, {np.log10(snr)*10:.1f}",
        "Electron dynamic range of the detector (Jannesick, Photon transfer, p. 50). Expressed as a dimensionless factor."
        )
        doc_dict["Detector dynamic range [power dB]"] = (f"{np.log10(snr)*10:.1f}",
        "Electron dynamic range of the detector. Expressed power dB"
        )
        doc_dict["Detector dynamic range [root-power dB]"] = (f"{2*np.log10(snr)*10:.1f}",
        "Electron dynamic range of the detector. Expressed as root-power dB, which is twice the power dB value. Commonly used for video cameras."
        )
        doc_dict["Saturation value [ADU]"] = (f"{var_peak+plotOffset:.0f}",
        "Value at peak of photon transfer curve. Units are ADU"
        )
        doc_dict["Saturation capacity [e-]"] = (f"{var_peak*gain:.0f}",
        "Saturation capacity in units of electrons"
        )
    doc_dict["Linearity Error"] = (f"{np.mean(np.abs(rel_dev)):.1%}",
                                    "Mean absolute deviation from fit within linearity fit range."
    )
    doc_dict["Fixed pattern offset std. [e-]"] = (f"{np.sqrt(patternVar):.2f}",
                                        "Gain multiplied with standard deviation of the background mean projection. Units are electrons."
    )
    if not saturationImage:
        doc_dict["Total electrons per exposure [e-]"] = (f"{mean_el_per_exposure:.3E}",
                                        "The mean number of photoelectrons per exposure. The background is subtracted from the signal, and the sum is multiplied by the gain estimate. Incorrect if a significant number of pixels are saturated."
        )

    doc_dict_vals = {k:v[0] for k, v in doc_dict.items()}
    strio = io.StringIO("")
    yaml = ruamel.yaml.YAML()
    yaml.dump(doc_dict_vals,  strio)
    strio.seek(0)
    fig_string = strio.read()
    # import debughelper
    # debughelper.save_to_interactive({"fig_string": fig_string})

    if doPlot:
          
        fig = plt.figure(figsize=plot_figsize)

        if CameraName is not None:
            plt.title("Photon transfer curve ("+CameraName+")", fontsize=TitleFontSize)
        else:
            plt.title("Photon transfer curve", fontsize=TitleFontSize)

        biased_binMid = binMid + plotOffset
        biased_mean_mean = mean_mean + plotOffset
        plt.plot(biased_mean_mean, mean_var, 'bo', label='Brightness bins')
        # plt.errorbar(biased_binMid, myFit, myStd, label='Fit')
        plt.plot(biased_binMid, myFit, color="tab:red", label='Gain fit')
        plt.plot(biased_binMid, myFit+myStd/2, '--r', label="Error")
        plt.plot(biased_binMid, myFit-myStd/2, '--r')
        if saturationImage:
            plt.axvline(var_peak + plotOffset, 0, 1, color="tab:purple", label="Saturation capacity")
        if saturationImage: # otherwise, linearity fit is the same as gain fit, and we don't want to plot it twice
            plt.plot(linfit_x + plotOffset, linvar_fits, color="tab:green", label='Linearity fit')
        plt.legend()

        # secondary axes in photoelectrons
        ax = plt.gca()
        def adu2el(adu):
            return (adu-bg_total_mean)*gain
        def el2adu(el):
            return el/gain+bg_total_mean
        secax_x = ax.secondary_xaxis('top', functions=(adu2el, el2adu))
        secax_y = ax.secondary_yaxis('right', functions=(lambda x: x*gain**2, lambda x: x/gain**2))
        
        secax_x.set_xlabel("Pixel brightness / $photoelectrons$", fontsize=AxisFontSize)
        secax_y.set_ylabel("Pixel variance / $photoelectrons^2$", fontsize=AxisFontSize)

        plt.xlabel("Pixel brightness / $ADU$", fontsize=AxisFontSize)
        plt.ylabel("Pixel variance / $ADU^2$", fontsize=AxisFontSize)
        plt.grid()
        plt.figtext(0.02, 0.05, fig_string, fontsize=TextFontSize)

        ax.set_xlim(plotOffset-0.05*np.ptp(binMid), ax.get_xlim()[1])
        ax.set_ylim(0, ax.get_ylim()[1])
        
        fig.subplots_adjust(left=0.4)

        if plotHist:
            ax_hist = ax.twinx()
            ax_hist.bar(binMid+plotOffset, hist_num, color="gray", alpha=0.5, width=np.diff(binMid).mean()*1)
            ax_hist.set_ylim(0, 2*np.percentile(hist_num, 95))
            ax_hist.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

        showRanges = True
        if showRanges:
            ax2 = ax.twinx()
            gainrange_rectangle = matplotlib.patches.Rectangle((validRange[0]+plotOffset, 0.0), 
                np.ptp(validRange), 1e-2, facecolor="tab:red", alpha=0.5)
            linrange_rectangle = matplotlib.patches.Rectangle((linearity_range[0]+plotOffset, 1e-2), 
                np.ptp(linearity_range), 1e-2, facecolor="tab:green", alpha=0.5)
            ax2.add_patch(gainrange_rectangle)
            ax2.add_patch(linrange_rectangle)
            ax2.set_navigate(False)
            ax2.set_yticks([])

        figures.append((fig, "Photon_Calibration"))


        plotADUhistograms = True
        if plotADUhistograms:
        
            # Dark Histogram
            fig = plt.figure(figsize=plot_figsize)
            plt.fill_between(dark_hist_bin_edges[:-1], dark_counts, 0, step="mid", alpha=1, color="tab:gray", label="Dark series")
            plt.yscale("log")
            plt.xlim(-bg_total_mean/2, 2.5*bg_total_mean)
            plt.ylim(0, dark_counts.max()**1.5)
            plt.xlabel("Signal intensity / ADU")
            plt.ylabel("Counts / frame")
            plt.title("Dark histogram")
            figures.append((fig, "dark_histogram"))

            # Bright Histogram
            fig = plt.figure(figsize=plot_figsize)
            plt.fill_between(bright_hist_bin_edges[:-1], bright_counts, 0, step="mid", alpha=1, color="tab:gray", label="Bright series")
            plt.yscale("log")   
            plt.xlim(-bg_total_mean/2, fg_max_ADU + bg_total_mean/2)
            plt.ylim(0, bright_counts.max()**1.5)
            plt.xlabel("Signal intensity / ADU")
            plt.ylabel("Counts / frame")
            plt.title("Bright histogram")
            figures.append((fig, "bright_histogram"))

            # Conbined Histogram
            fig = plt.figure(figsize=plot_figsize)
            plt.fill_between(bright_hist_bin_edges[:-1], bright_counts, 0, step="mid", alpha=0.5, color="tab:green", label="Bright series")
            plt.fill_between(dark_hist_bin_edges[:-1], dark_counts, 0, step="mid", alpha=0.8, color="tab:gray", label="Dark series")
            plt.yscale("log")   
            plt.xlim(-bg_total_mean/2, fg_max_ADU + bg_total_mean/2)
            plt.ylim(0, dark_counts.max()**1.5)
            plt.title("Combined histograms")
            plt.xlabel("Signal intensity / ADU")
            plt.ylabel("Counts / frame")
            plt.legend()
            figures.append((fig, "combined_histograms"))
        
        # linearity error plot
        fig = plt.figure(figsize=plot_figsize)
        if CameraName is not None:
            plt.title("Deviation from linearity ("+CameraName+")", fontsize=TitleFontSize)
        else:
            plt.title("Deviation from linearity", fontsize=TitleFontSize)
        plt.plot(linfit_x_extended + plotOffset, rel_dev_extended*100)
        plt.ylim(min(rel_dev*100)*1.5, max(rel_dev*100)*1.5)
        plt.xlabel("Pixel brightness / $ADU$", fontsize=AxisFontSize)
        plt.ylabel('Deviation / %.', fontsize=AxisFontSize)

        # secondary axis again
        ax = plt.gca()
        secax_x = ax.secondary_xaxis('top', functions=(adu2el, el2adu))
        secax_x.set_xlabel("Pixel brightness / $photoelectrons$", fontsize=AxisFontSize)

        figures.append((fig, "linearity_error"))
            

        if exportpath is not None:
            for fig, figname in figures:
                fig.savefig(exportpath/f'{figname}.{exportFormat}')

    if exportpath is not None:
        results_dict = ruamel.yaml.comments.CommentedMap({k:v[0] for k, v in doc_dict.items()})
        results_topline_comment = """Results from calibration tool for inhomogenous image stacks.
See definitions.txt for extended definitions.
The results are in the YAML format, readable by humans and machines
Written by David McFadden, FSU Jena
NanoImagingPack library: https://github.com/bionanoimaging/NanoImagingPack
Questions, bugs and requests to: david.mcfadden777@gmail.com
heintzmann@gmail.com
        """
        results_dict.yaml_set_start_comment(results_topline_comment)
        yaml.dump(results_dict, open(exportpath/'calibration_results.txt', "w"))
        doc_dict_dict = ruamel.yaml.comments.CommentedMap({k:v[1] for k, v in doc_dict.items()})
        docs_topline_comment = """Extended definitions for calibration results.
Written in the YAML format, readable by humans and machines
Written by David McFadden, FSU Jena
NanoImagingPack library: https://github.com/bionanoimaging/NanoImagingPack
Questions, bugs and requests to: david.mcfadden777@gmail.com
heintzmann@gmail.com
        """        
        doc_dict_dict.yaml_set_start_comment(docs_topline_comment)
        yaml.dump(doc_dict_dict, open(exportpath/'definitions.txt', "w"))
    # TODO: print the yaml text to terminal?
    return (bg_total_mean, gain, Readnoise, mean_el_per_exposure, validmap, figures, doc_dict)


def check_dark(background, threshold=10):
    """Checks if a dark frame stack is truly dark, or if there is some global structure.
        Used to identify if a frame was incorrectly passed as a dark calibration frame.

    Args:
        background (_type_): 3D stack of dark frame
        threshold (int, optional): _description_. Defaults to 3.

    Returns:
        Boolean: True if dark, False if suspect structure.
    """
    dark = background.mean((0)).std()/np.mean(background.std((0))) < threshold
    return dark

