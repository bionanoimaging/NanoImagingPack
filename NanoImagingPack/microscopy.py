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
from .image import image
# from .view5d import v5 # for debugging
from matplotlib.pyplot import plot,figure, xlabel, ylabel, text, title, xlim, ylim # rc, errorbar,

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

    :param pupil:
    :param sizeZ:
    :param psf_params:
    :param mode:
    :param doDampPupil:
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

def cal_readnoise(fg,bg,numBins=100, validRange=None, CameraName=None, correctBrightness=True, correctOffsetDrift=True, excludeHotPixels=True, doPlot=True):
    """
    calibrates detectors by fitting a straight line through a mean-variance plot
    :param fg: A series of foreground images of the same (blurry) scene. Ideally spanning the useful range of the detector. Suggested number: 20 images
    :param bg: A series of dark images under identical camera settings (integration time, temperature, etc.). Suggested number: 20 images
    :param numBins: Number of bins for the fit
    :param validRange: If provided, the fit will only be performed over this range (from,to) of mean values. The values are inclusived borders
    :param doPlot: Plot the mean-variance curves
    :return: tuple of fit results (offset [adu], gain [electrons / adu], readnoise [e- RMS])
    to scale your images into photons use: (image-offset) * gain
    """
    # a_nobg = a * 1.0 - meanbg
    AxisFontSize=16
    TextFontSize=14
    TitleFontSize=20
    # rc('font', size=AxisFontSize)  # controls default text sizes
    # rc('axes', titlesize=AxisFontSize, labelsize=AxisFontSize)  # fontsize of the axes title and number labels
    fg = np.squeeze(fg)
    bg = np.squeeze(bg)
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
        return cal_readnoise(fg,bg,numBins, validRange, CameraName, correctBrightness, correctOffsetDrift, excludeHotPixels, doPlot)


    Text = "Analysed {nb:d} bright and {nd:d} dark images:\n".format(nb=fg.shape[0],nd=bg.shape[0])

    validmap = None
    if validRange is None:
        underflow = np.sum(fg <= 0, (0,)) > 0
        validmap = ~ underflow
        numUnderflow = np.sum(underflow)
        if np.min(fg) <= 0:
            print("WARNING: "+str(numUnderflow)+" pixels with at least one zero-value (value<=0) were detected but no fit range was selected. Excluded those from the fit.")
        Text = Text + "Zero-value pixels: {zv:d}".format(zv=numUnderflow)+" excluded.\n"
        overflow = None
        for MaxVal in [255,4096]:
            if np.max(fg) == MaxVal:
                overflow = np.sum(fg == MaxVal,(0,)) > 0
                relsat = np.sum(overflow)
                print("WARNING: "+str(relsat)+" pixels saturating at least once (value=="+str(MaxVal)+") were detected but no fit range was selected. Excluding those from the fit.")
                Text = Text + "Overflow (=="+str(MaxVal)+") pixels: "+str(relsat)+"  excluded.\n"
                validmap = validmap & ~ overflow
        if overflow is None:
            Text = Text + "No overflow pixels detected.\n"

    print("Calibration results:")
    if correctOffsetDrift:
        meanoffset = np.mean(bg, (-2,-1), keepdims=True)
        refOffset = np.mean(bg)
        bg = bg - meanoffset + refOffset
        reloffset = (refOffset - meanoffset)  / np.sqrt(np.var(bg))
        if doPlot:
            figure(figsize=(12, 6))
            if CameraName is not None:
                title("Offset Drift ("+CameraName+")", fontsize=TitleFontSize)
            else:
                title("Offset Drift", fontsize=TitleFontSize)
            plot(reloffset.flat, label='Offset / Std.Dev.')
            xlabel("frame no.", fontsize=AxisFontSize)
            ylabel("mean offset / Std.Dev.", fontsize=AxisFontSize)

    meanbg = np.mean(bg, (-3))
    background = np.mean(meanbg)
    patternVar = np.var(meanbg)

    if correctBrightness:
        brightness = np.mean(fg, (-2,-1), keepdims=True)
        meanbright = np.mean(brightness)
        relbright = brightness/meanbright
        if doPlot:
            figure(figsize=(12, 6))
            if CameraName is not None:
                title("Brightness Fluctuation ("+CameraName+")", fontsize=TitleFontSize)
            else:
                title("Brightness Fluctuation", fontsize=TitleFontSize)
            plot(relbright.flat)
            xlabel("frame no.",fontsize=AxisFontSize)
            ylabel("relative brightness",fontsize=AxisFontSize)
        fg = (fg - background) * relbright
        maxFluc = np.max(np.abs(1.0-relbright))
        Text = Text + "Illumination fluctuation: {bf:.2f}".format(bf=maxFluc * 100.0)+"%\n"
    meanp = np.mean(fg, (-3))
    varp = np.var(fg, (-3))
    varbg = np.var(bg, (-3))

    hotPixels = None
    if excludeHotPixels:
        hotPixels = np.abs(meanbg - np.mean(meanbg)) > 4.0*np.sqrt(np.mean(varbg))
        numHotPixels=np.sum(hotPixels)
        Text = Text + "Hot pixels (|bg mean| > 4 StdDev): "+str(numHotPixels)+" excluded.\n"
        if validmap is None:
            validmap = ~hotPixels
        else:
            validmap = validmap & ~hotPixels

    if validmap is not None:
        (histNum, mybins) = np.histogram(meanp, bins=numBins, weights=validmap+0.0)
        (histWeight, mybins) = np.histogram(meanp, bins=numBins, weights=validmap*varp)
    else:
        (histNum, mybins) = np.histogram(meanp, bins=numBins)
        (histWeight, mybins) = np.histogram(meanp, bins=numBins, weights=varp)
    binMid = (mybins[1:] + mybins[:-1]) / 2.0;

    valid = histNum > 0
    histWeight = histWeight[valid]
    histNum = histNum[valid]
    binMid = binMid[valid]

    meanvar = histWeight / histNum # this yields the mean variance curve

    (offset, slope, vv) = RWLSPoisson(binMid, meanvar, histNum, validRange=validRange)

    myFit = binMid * slope + offset
    myStd = np.sqrt(vv)
    gain = 1.0 / slope
    Text = Text + "Background [adu]: {bg:.2f}".format(bg=background) + "\n"
    Text = Text + "Gain [e- / adu]): {g:.4f}".format(g=gain) + "\n"
    if offset < 0.0:
        Text = Text + "Readnoise (fit): variance ({of:.2f}) below zero.".format(of=offset) + "\n"
    else:
        Text = Text + "Readnoise (fit): {rn:.2f}".format(rn=np.sqrt(offset)*gain)+"\n"
    Text = Text + "Fixed Pattern, gain * Std.Dev. for mean_bg: {rn:.2f}".format(rn=np.sqrt(patternVar))+"e- RMS\n"
    Readnoise = np.sqrt(np.mean(varbg)) * gain
    Text = Text + "Readnoise, gain * bg_noise: {rn:.2f}".format(rn=Readnoise)+" e- RMS\n"

    if doPlot:
        figure(figsize=(12, 8))
        if CameraName is not None:
            title("Photon Calibration ("+CameraName+")", fontsize=TitleFontSize)
        else:
            title("Photon Calibration", fontsize=TitleFontSize)
        plot(binMid, meanvar, 'bo', label='Camera Data')
        # errorbar(binMid, myStd, myStd, label='Fit')
        plot(binMid, myFit, 'r', label='Fit')
        plot(binMid, myFit+myStd, '--r')
        plot(binMid, myFit-myStd, '--r')
        maxx = np.max(binMid)
        maxy = np.max(meanvar)
        xlim(0,maxx*1.05)
        ylim(0,maxy*1.05)
        xlabel("mean signal / adu", fontsize=AxisFontSize)
        ylabel("variance / adu", fontsize=AxisFontSize)
        text(maxx/20.0, maxy*0.6, Text, fontsize=TextFontSize)

    print(Text)
    return (background, gain, Readnoise, hotPixels)
