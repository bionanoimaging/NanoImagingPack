#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:22:51 2017

@author: root

This Package should contain funcitons in order to create psfs and otfs both 2-Dimensional and 3-Dimensional
 
Also SIM stuff
"""

import numpy as np
from .util import zernike, expanddim, midValAsg
from .transformations import rft,irft,ft2d
from .coordinates import rr, phiphi, px_freq_step, zz
from .config import PSF_PARAMS, __DEFAULTS__
import numbers
import warnings
from scipy.special import j1
from .image import image
from .view5d import v5 # for debugging


def __setPol__(im, psf_params= PSF_PARAMS):
    """
    Create the polarization maps (x and y), returned as tuple of 2 2D images based on the polarization parameters (pol, pol_xy_phase_shift, pol_lin_angle) of the psf parameter structure

    :param im: input image
    :param psf_params: psf parameter structure
    :return: returns the polarization map (tuple of 2 images with x and y polarization components)
    """
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


def setAberrationMap(im, psf_params= PSF_PARAMS):
    """
    create a aberration phase map (based on Zernike polynomials) or a transmission matrix (aperture) for PSF generation

    uses:
        PSF_PARAMS.aberration_strength = None;
        PSF_PARAMS.aberration_types = None;
        PSF_PARAMS.aperture_transmission = None;



    strength:           strength of the aberration as multiples of 2pi in the phase (polynomial reach from -1 to 1)
    aberration_types:   can be
                            tuple (m, n) describing the Z^m_n polynomial
                        or phasemap

                        or string
                                piston     -> (Z0,0)
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
    transmission:   The transmission map
    """
    zernike_para = {'piston': (0,0),
                    'tiltY': (-1,1),
                    'tiltX': (1,1),
                    'astigm': (-2,2),
                    'defoc': (0,2),
                    'vastig': (2,2),
                    'vtrefoil': (-3,3),
                    'vcoma': (-1,3),
                    'hcoma': (1,3),
                    'obtrefoil': (3,3),
                    'obquadfoil': (-4,4),
                    'asti2nd': (-2,4),
                    'spheric': (0,4),
                    'vasti2nd': (2,4),
                    'vquadfoil': (4,4),

            }
    aberration_map = image(np.ones(im.shape[-2:], dtype=np.complex128))
    strength = psf_params.aberration_strength
    aberration = psf_params.aberration_types
    transmission = psf_params.aperture_transmission

    if transmission is not None:
        if isinstance(transmission, np.ndarray):
            if transmission.shape == im.shape[-2:]:
                if np.min(transmission)<0:
                    warnings.warn('Transmission mask is negative at some values')
                if np.max(transmission)>1:
                    warnings.warn('Transmission mask is larger than one at some values')
                aberration_map*= transmission
            else:
                raise ValueError('Wrong dimension of transmission matrix')
        else:
            raise ValueError('Wrong transmission -> must be 2D image or array of the same xy-dimension as the image')
    if strength is not None and aberration is not None:
        if isinstance(im, image):
            pxs=im.px_freq_step()[-2:]
        else:
            pxs = __DEFAULTS__['IMG_PIXELSIZES']
            pxs = px_freq_step(im, pxs)[-2:]

        r = rr(im.shape[-2:], scale=pxs) * psf_params.wavelength / psf_params.NA
        if isinstance(strength, numbers.Real):
            strength = [strength]
        if type(aberration) == str or isinstance(aberration,np.ndarray):
            aberration = [aberration]
        elif type(aberration) == list or type(aberration) == tuple:
            if  isinstance(aberration[0], numbers.Integral):
                aberration= [aberration]
        for s, ab in zip(strength, aberration):
       #TODO: CHEKC if ZERNICKE MAP HAS TO BE STRETCHED OVER PI or over 2PI!!!
       # ZERNICKE value range is between -1 and 1;
            if type(ab) == str:
                m = zernike_para[ab][0]
                n = zernike_para[ab][1]
                aberration_map *= np.exp(1j*s*zernike(r,m,n)*np.pi)
            elif isinstance(ab,np.ndarray):
                aberration_map *= np.exp(1j*ab)
            else:
                m = ab[0]
                n = ab[1]
                aberration_map *= np.exp(1j*s*zernike(r,m,n)*np.pi)
    return(aberration_map)


def __make_propagator__(im, psf_params = PSF_PARAMS, mode = 'Fourier'):
    """
    Compute the field propagation matrix

    :param im:              input image
    :param psf_params:      psf params structure
    :param mode:            "Fourier" the normal fourier based propagation matrix

                            TODO: Include chirp Z transform method
    :return:                Returns the phase propagation matrix (exp^i*delta_phi)
    """

    if isinstance(im, image):
        pxs = im.pixelsize
        ft_pxs = im.px_freq_step()
    else:
        pxs = __DEFAULTS__['IMG_PIXELSIZES']
        ft_pxs = px_freq_step(im, pxs)

    if (len(pxs)<3) and (len(im.shape)>2):
        axial_pxs = __DEFAULTS__['IMG_PIXELSIZES'][0]
        warnings.warn('Only 2 Dimensional image given -> using default ('+str(axial_pxs)+')')
    else:
        axial_pxs = pxs[0]
    if len(im.shape)>2:
        r = rr(im.shape[-2:], scale=ft_pxs[-2:]) * psf_params.wavelength / psf_params.NA
        r = (r <= 1) * r
        s = psf_params.NA/psf_params.n

        propagator = psf_params.n * np.pi * 2 / psf_params.wavelength * np.sqrt(1 - r ** 2 *s ** 2) * axial_pxs * zz(im)
    else:
        propagator = np.zeros(im.shape)
    return(np.exp(-1j*propagator))

def simLens(im, psf_params = PSF_PARAMS):
    """
    Compute the focal plane field (fourier transform of the electrical field in the plane (at position PSF_PARAMS.off_focal_dist)

    The following influences will be included:
        1.)     Mode of plane field generation (from sinc or circle)
        2.)     Aplanatic factor
        3.)     Potential aberrations (set by set_aberration_map)
        4.)     Vectorial effects

    returns image of shape (3,y_im,x_im) where x_im and y_im are the xy dimensions of the input image. In the third dimension are the field components (0: Ex, 1: Ey, 2: Ez)
    """

    # setting up parameters:
    NA = psf_params.NA
    wl = psf_params.wavelength
    n = psf_params.n
    if isinstance(im, image):
        pxs = im.pixelsize[-2:]
        ft_pxs = im.px_freq_step()[-2:]
    else:
        pxs = __DEFAULTS__['IMG_PIXELSIZES']
        ft_pxs = px_freq_step(im, pxs)[-2:]
    shape = im.shape[-2:]

    r = rr(shape, scale=ft_pxs) * wl / NA  # [:2]
    aperture = (r<=1)*1.0           # aperture transmission
    r = (r<=1)*r  # normalized aperture coordinate
    s = NA / n
    cos_alpha = np.sqrt(1-(s*r)**2)  # angle map cos
    sin_alpha = s*r  # angle map sin

    # Make base field
    if psf_params.foc_field_method == 'theoretical':
        np.seterr(divide='ignore', invalid='ignore')
        arg = rr(shape,scale=np.array(pxs)*2*np.pi*NA/wl)#  RH 3.2.19 was 2*mp.pi*np.sqrt((xx(shape)*self.pixelsize[0])**2+(yy(shape)*self.pixelsize[1])**2)*self.NA/self.wavelength;
        ret = 2*j1(arg)/arg  # that again creates the dip!
        ret[shape[-2]//2, shape[-1]//2] = 1
        ret = ret / np.sum(np.abs(ret))
        plane = ft2d(ret)
        np.seterr(divide='warn', invalid='warn')
    elif psf_params.foc_field_method == 'circle':
        plane = aperture*1j
    else:
        raise ValueError('Wrong focal_field_method in PSF_PARAM structure')

    # Apply aplanatic factor:
    if psf_params.aplanar is None:
        pass;
    elif psf_params.aplanar == 'excitation':
        plane *= np.sqrt(cos_alpha)
    elif psf_params.aplanar == 'emission':
        plane /= np.sqrt(cos_alpha)
    elif psf_params.aplanar == 'excitation2':
        plane *= cos_alpha
    elif psf_params.aplanar == 'emission2':
        plane /= cos_alpha
    else:
        raise ValueError('Wrong aplanatic factor in PSF_PARAM structure:  "excitation", "emission","excitation2","emission2" or None, 2 means squared aplanatic factor for flux measurement')

    # Apply aberrations and apertures
    plane *= setAberrationMap(im, psf_params)

    # Apply z-offset:
    plane *= np.exp(-2j*n*np.pi/wl * cos_alpha * psf_params.off_focal_distance)

    # expand to 4 Dims, the 4th will contain the electric fields
    plane = plane.cat([plane,plane],-4)
    plane.dim_description = {'d3': ['Ex','Ey','Ez']}

    #Apply vectorized distortions
    polx, poly = __setPol__(im, psf_params= psf_params)
    if psf_params.vectorized:
        theta = phiphi(shape)
        E_radial     = np.cos(theta)*polx-np.sin(theta)*poly
        E_tangential = np.sin(theta)*polx+np.cos(theta)*poly
        E_z = E_radial*sin_alpha
        E_radial *=cos_alpha
        plane[0]*= np.cos(theta)*E_radial+np.sin(theta)*E_tangential
        plane[1]*=-np.sin(theta)*E_radial+np.cos(theta)*E_tangential
        plane[2]*= E_z
    else:
        plane[0]*=polx
        plane[1]*=poly
        plane[2]*=0
    return plane # *aperture  # hard edge aperture at ?

def __make_transfer__(im, psf_params = PSF_PARAMS, mode = 'ctf', dimension = 2):
    """
    Creates the transfer function
    Also adds members PSF_PARAMS

    :param im:                  input image
    :param psf_params:          PSF Parameter struct
    :param mode:                'ctf', 'otf', 'psf', 'apsf'
    :param dimension:           if 2-> only 2D, if None like input image
    :return:                    returns the transfer function (dimension x,y, potentially z and field components for ctf and apsf
    """

    ret = simLens(im, psf_params)
    if dimension is None:
        ret= ret*__make_propagator__(im, psf_params = PSF_PARAMS)  # field (kx,ky) propagated along the z - component
    if mode == 'ctf':  # Field transfer function is ft along z axis
        if ret.shape[-3] >1:
            ret = ret.ft(axes = -3)
    elif mode ==  'apsf':                # field spread function is ft along kx-ky-axes
        ret = ret.ift2d()
    elif mode == 'psf':                  # psf: like apsf and then abs square of field components
        ret = ret.ift2d()
        ret *= ret.conjugate()
        ret = ret.sum(axis=0)
        ret = np.real(ret)
        ret /= np.sum(ret)
    elif mode == 'otf':                  # otf: ft (over whole space) of psf
        ret = ret.ift2d()
        ret *= ret.conjugate()
        ret = ret.sum(axis=0)
        ret = np.real(ret)
        ret /= np.sum(ret)
        ret = ret.ft2d(norm=None)
    ret.PSF_PARAMS = psf_params
    return(ret)

def otf(im, psf_params = PSF_PARAMS):
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
        PSF_PARAMS, psf, psf2d, otf, otf2d, apsf, apsf2d, ctf, ctf2d, simLens, setAberrationMap


        Example:
        -------
        - Create 2D PSF of 3d image

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS;
            psf = nip.psf2d(im, para);

         - Create 3D PSF with circular polarization:

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS;
            nip.para.pol = nip.para.pols.elliptical;
            psf = nip.psf(im, para);

         -  Create 2D OTF (because the image is 2D) and add some aberrations and a soft aperture

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            soft_aperture = nip.gaussian(im.shape, 10);         #Define some soft aperture
            aber_map = nip.xx(im).normalize(1);                # Define some aberration map (x-ramp in this case)
            para.aberration_types=[para.aberration_zernikes.spheric, (3,5), aber_map]    # define list of aberrations (select from choice, zernike coeffs, or aberration map
            para.aberration_strength =[0.5,0.8,0.2];
            otf = nip.otf(im, para);

        -   Create a non-Vectorial, emission APSF in 3D based on a circle in the Fourier space instead of a jinc in the real space

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            para.vectorized = False;
            para.aplanar = para.apl.emission
            para.foc_field_method = "circle"
            apsf = nip.apsf(im, para);

        Examples regarding the PSF_PARAMS:

        -   Set up linear polarization of arbitrary angle:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS;
            para.pol = "lin";           # you might as well choose via para.pol = para.pols.lin
            para.pol_lin_angle = np.pi*3/17;

        -   Set up elliptical polarization of arbitrary angle phase shift between x and y component:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS;
            para.pol = "elliptic";           # you might as well choose via para.pol = para.pols.lin
            para.pol_xy_phase_shift = np.pi*3/17;

        -   Set up x and y components of the polarization indendently:

            import NanoImagingPack as nip;
            import numpy as np;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            polx = nip.xx(im).normalize(1)*8.0;
            poly = np.random.rand(im.shape[-2], im.shape[-1])*np.exp(-2.546j);
            para.pol = [polx, poly]
    """
    return(__make_transfer__(im, psf_params = psf_params, mode = 'otf', dimension = None))


def ctf(im, psf_params = PSF_PARAMS):
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
            para = nip.PSF_PARAMS;
            psf = nip.psf2d(im, para);

         - Create 3D PSF with circular polarization:

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS;
            nip.para.pol = nip.para.pols.elliptical;
            psf = nip.psf(im, para);

         -  Create 2D OTF (because the image is 2D) and add some aberrations and a soft aperture

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            soft_aperture = nip.gaussian(im.shape, 10);         #Define some soft aperture
            aber_map = nip.xx(im).normalize(1);                # Define some aberration map (x-ramp in this case)
            para.aberration_types=[para.aberration_zernikes.spheric, (3,5), aber_map]    # define list of aberrations (select from choice, zernike coeffs, or aberration map
            para.aberration_strength =[0.5,0.8,0.2];
            otf = nip.otf(im, para);

        -   Create a non-Vectorial, emission APSF in 3D based on a circle in the Fourier space instead of a jinc in the real space

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            para.vectorized = False;
            para.aplanar = para.apl.emission
            para.foc_field_method = "circle"
            apsf = nip.apsf(im, para);

        Examples regarding the PSF_PARAMS:

        -   Set up linear polarization of arbitrary angle:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS;
            para.pol = "lin";           # you might as well choose via para.pol = para.pols.lin
            para.pol_lin_angle = np.pi*3/17;

        -   Set up elliptical polarization of arbitrary angle phase shift between x and y component:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS;
            para.pol = "elliptic";           # you might as well choose via para.pol = para.pols.lin
            para.pol_xy_phase_shift = np.pi*3/17;

        -   Set up x and y components of the polarization indendently:

            import NanoImagingPack as nip;
            import numpy as np;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            polx = nip.xx(im).normalize(1)*8.0;
            poly = np.random.rand(im.shape[-2], im.shape[-1])*np.exp(-2.546j);
            para.pol = [polx, poly]
    """
    return(__make_transfer__(im, psf_params = psf_params, mode = 'ctf', dimension = None))
def apsf(im, psf_params = PSF_PARAMS):
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
            para = nip.PSF_PARAMS;
            psf = nip.psf2d(im, para);

         - Create 3D PSF with circular polarization:

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS;
            nip.para.pol = nip.para.pols.elliptical;
            psf = nip.psf(im, para);

         -  Create 2D OTF (because the image is 2D) and add some aberrations and a soft aperture

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            soft_aperture = nip.gaussian(im.shape, 10);         #Define some soft aperture
            aber_map = nip.xx(im).normalize(1);                # Define some aberration map (x-ramp in this case)
            para.aberration_types=[para.aberration_zernikes.spheric, (3,5), aber_map]    # define list of aberrations (select from choice, zernike coeffs, or aberration map
            para.aberration_strength =[0.5,0.8,0.2];
            otf = nip.otf(im, para);

        -   Create a non-Vectorial, emission APSF in 3D based on a circle in the Fourier space instead of a jinc in the real space

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            para.vectorized = False;
            para.aplanar = para.apl.emission
            para.foc_field_method = "circle"
            apsf = nip.apsf(im, para);

        Examples regarding the PSF_PARAMS:

        -   Set up linear polarization of arbitrary angle:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS;
            para.pol = "lin";           # you might as well choose via para.pol = para.pols.lin
            para.pol_lin_angle = np.pi*3/17;

        -   Set up elliptical polarization of arbitrary angle phase shift between x and y component:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS;
            para.pol = "elliptic";           # you might as well choose via para.pol = para.pols.lin
            para.pol_xy_phase_shift = np.pi*3/17;

        -   Set up x and y components of the polarization indendently:

            import NanoImagingPack as nip;
            import numpy as np;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            polx = nip.xx(im).normalize(1)*8.0;
            poly = np.random.rand(im.shape[-2], im.shape[-1])*np.exp(-2.546j);
            para.pol = [polx, poly]
    """
    return(__make_transfer__(im, psf_params = psf_params, mode = 'apsf', dimension = None))
def psf(im, psf_params = PSF_PARAMS):
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
            para = nip.PSF_PARAMS;
            psf = nip.psf2d(im, para);

         - Create 3D PSF with circular polarization:

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS;
            nip.para.pol = nip.para.pols.elliptical;
            psf = nip.psf(im, para);

         -  Create 2D OTF (because the image is 2D) and add some aberrations and a soft aperture

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            soft_aperture = nip.gaussian(im.shape, 10);         #Define some soft aperture
            aber_map = nip.xx(im).normalize(1);                # Define some aberration map (x-ramp in this case)
            para.aberration_types=[para.aberration_zernikes.spheric, (3,5), aber_map]    # define list of aberrations (select from choice, zernike coeffs, or aberration map
            para.aberration_strength =[0.5,0.8,0.2];
            otf = nip.otf(im, para);

        -   Create a non-Vectorial, emission APSF in 3D based on a circle in the Fourier space instead of a jinc in the real space

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            para.vectorized = False;
            para.aplanar = para.apl.emission
            para.foc_field_method = "circle"
            apsf = nip.apsf(im, para);

        Examples regarding the PSF_PARAMS:

        -   Set up linear polarization of arbitrary angle:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS;
            para.pol = "lin";           # you might as well choose via para.pol = para.pols.lin
            para.pol_lin_angle = np.pi*3/17;

        -   Set up elliptical polarization of arbitrary angle phase shift between x and y component:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS;
            para.pol = "elliptic";           # you might as well choose via para.pol = para.pols.lin
            para.pol_xy_phase_shift = np.pi*3/17;

        -   Set up x and y components of the polarization indendently:

            import NanoImagingPack as nip;
            import numpy as np;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            polx = nip.xx(im).normalize(1)*8.0;
            poly = np.random.rand(im.shape[-2], im.shape[-1])*np.exp(-2.546j);
            para.pol = [polx, poly]
    """
    return(__make_transfer__(im, psf_params = psf_params, mode = 'psf', dimension = None))
def otf2d(im, psf_params = PSF_PARAMS):
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
            para = nip.PSF_PARAMS;
            psf = nip.psf2d(im, para);

         - Create 3D PSF with circular polarization:

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS;
            nip.para.pol = nip.para.pols.elliptical;
            psf = nip.psf(im, para);

         -  Create 2D OTF (because the image is 2D) and add some aberrations and a soft aperture

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            soft_aperture = nip.gaussian(im.shape, 10);         #Define some soft aperture
            aber_map = nip.xx(im).normalize(1);                # Define some aberration map (x-ramp in this case)
            para.aberration_types=[para.aberration_zernikes.spheric, (3,5), aber_map]    # define list of aberrations (select from choice, zernike coeffs, or aberration map
            para.aberration_strength =[0.5,0.8,0.2];
            otf = nip.otf(im, para);

        -   Create a non-Vectorial, emission APSF in 3D based on a circle in the Fourier space instead of a jinc in the real space

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            para.vectorized = False;
            para.aplanar = para.apl.emission
            para.foc_field_method = "circle"
            apsf = nip.apsf(im, para);

        Examples regarding the PSF_PARAMS:

        -   Set up linear polarization of arbitrary angle:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS;
            para.pol = "lin";           # you might as well choose via para.pol = para.pols.lin
            para.pol_lin_angle = np.pi*3/17;

        -   Set up elliptical polarization of arbitrary angle phase shift between x and y component:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS;
            para.pol = "elliptic";           # you might as well choose via para.pol = para.pols.lin
            para.pol_xy_phase_shift = np.pi*3/17;

        -   Set up x and y components of the polarization indendently:

            import NanoImagingPack as nip;
            import numpy as np;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            polx = nip.xx(im).normalize(1)*8.0;
            poly = np.random.rand(im.shape[-2], im.shape[-1])*np.exp(-2.546j);
            para.pol = [polx, poly]
    """
    return(__make_transfer__(im, psf_params = psf_params, mode = 'otf', dimension = 2).squeeze())


def ctf2d(im, psf_params = PSF_PARAMS):
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
            para = nip.PSF_PARAMS;
            psf = nip.psf2d(im, para);

         - Create 3D PSF with circular polarization:

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS;
            nip.para.pol = nip.para.pols.elliptical;
            psf = nip.psf(im, para);

         -  Create 2D OTF (because the image is 2D) and add some aberrations and a soft aperture

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            soft_aperture = nip.gaussian(im.shape, 10);         #Define some soft aperture
            aber_map = nip.xx(im).normalize(1);                # Define some aberration map (x-ramp in this case)
            para.aberration_types=[para.aberration_zernikes.spheric, (3,5), aber_map]    # define list of aberrations (select from choice, zernike coeffs, or aberration map
            para.aberration_strength =[0.5,0.8,0.2];
            otf = nip.otf(im, para);

        -   Create a non-Vectorial, emission APSF in 3D based on a circle in the Fourier space instead of a jinc in the real space

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            para.vectorized = False;
            para.aplanar = para.apl.emission
            para.foc_field_method = "circle"
            apsf = nip.apsf(im, para);

        Examples regarding the PSF_PARAMS:

        -   Set up linear polarization of arbitrary angle:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS;
            para.pol = "lin";           # you might as well choose via para.pol = para.pols.lin
            para.pol_lin_angle = np.pi*3/17;

        -   Set up elliptical polarization of arbitrary angle phase shift between x and y component:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS;
            para.pol = "elliptic";           # you might as well choose via para.pol = para.pols.lin
            para.pol_xy_phase_shift = np.pi*3/17;

        -   Set up x and y components of the polarization indendently:

            import NanoImagingPack as nip;
            import numpy as np;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            polx = nip.xx(im).normalize(1)*8.0;
            poly = np.random.rand(im.shape[-2], im.shape[-1])*np.exp(-2.546j);
            para.pol = [polx, poly]
    """
    return(__make_transfer__(im, psf_params = psf_params, mode = 'ctf', dimension = 2).squeeze())
def apsf2d(im, psf_params = PSF_PARAMS):
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
            para = nip.PSF_PARAMS;
            psf = nip.psf2d(im, para);

         - Create 3D PSF with circular polarization:

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS;
            nip.para.pol = nip.para.pols.elliptical;
            psf = nip.psf(im, para);

         -  Create 2D OTF (because the image is 2D) and add some aberrations and a soft aperture

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            soft_aperture = nip.gaussian(im.shape, 10);         #Define some soft aperture
            aber_map = nip.xx(im).normalize(1);                # Define some aberration map (x-ramp in this case)
            para.aberration_types=[para.aberration_zernikes.spheric, (3,5), aber_map]    # define list of aberrations (select from choice, zernike coeffs, or aberration map
            para.aberration_strength =[0.5,0.8,0.2];
            otf = nip.otf(im, para);

        -   Create a non-Vectorial, emission APSF in 3D based on a circle in the Fourier space instead of a jinc in the real space

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            para.vectorized = False;
            para.aplanar = para.apl.emission
            para.foc_field_method = "circle"
            apsf = nip.apsf(im, para);

        Examples regarding the PSF_PARAMS:

        -   Set up linear polarization of arbitrary angle:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS;
            para.pol = "lin";           # you might as well choose via para.pol = para.pols.lin
            para.pol_lin_angle = np.pi*3/17;

        -   Set up elliptical polarization of arbitrary angle phase shift between x and y component:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS;
            para.pol = "elliptic";           # you might as well choose via para.pol = para.pols.lin
            para.pol_xy_phase_shift = np.pi*3/17;

        -   Set up x and y components of the polarization indendently:

            import NanoImagingPack as nip;
            import numpy as np;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            polx = nip.xx(im).normalize(1)*8.0;
            poly = np.random.rand(im.shape[-2], im.shape[-1])*np.exp(-2.546j);
            para.pol = [polx, poly]
    """
    return(__make_transfer__(im, psf_params = psf_params, mode = 'apsf', dimension = 2).squeeze())
def psf2d(im, psf_params = PSF_PARAMS):
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
            para = nip.PSF_PARAMS;
            psf = nip.psf2d(im, para);

         - Create 3D PSF with circular polarization:

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            para = nip.PSF_PARAMS;
            nip.para.pol = nip.para.pols.elliptical;
            psf = nip.psf(im, para);

         -  Create 2D OTF (because the image is 2D) and add some aberrations and a soft aperture

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            soft_aperture = nip.gaussian(im.shape, 10);         #Define some soft aperture
            aber_map = nip.xx(im).normalize(1);                # Define some aberration map (x-ramp in this case)
            para.aberration_types=[para.aberration_zernikes.spheric, (3,5), aber_map]    # define list of aberrations (select from choice, zernike coeffs, or aberration map
            para.aberration_strength =[0.5,0.8,0.2];
            otf = nip.otf(im, para);

        -   Create a non-Vectorial, emission APSF in 3D based on a circle in the Fourier space instead of a jinc in the real space

            import NanoImagingPack as nip;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            para.vectorized = False;
            para.aplanar = para.apl.emission
            para.foc_field_method = "circle"
            apsf = nip.apsf(im, para);

        Examples regarding the PSF_PARAMS:

        -   Set up linear polarization of arbitrary angle:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS;
            para.pol = "lin";           # you might as well choose via para.pol = para.pols.lin
            para.pol_lin_angle = np.pi*3/17;

        -   Set up elliptical polarization of arbitrary angle phase shift between x and y component:

            import NanoImagingPack as nip;
            import numpy as np;
            para = nip.PSF_PARAMS;
            para.pol = "elliptic";           # you might as well choose via para.pol = para.pols.lin
            para.pol_xy_phase_shift = np.pi*3/17;

        -   Set up x and y components of the polarization indendently:

            import NanoImagingPack as nip;
            import numpy as np;
            im = nip.readim('erika');
            para = nip.PSF_PARAMS;
            polx = nip.xx(im).normalize(1)*8.0;
            poly = np.random.rand(im.shape[-2], im.shape[-1])*np.exp(-2.546j);
            para.pol = [polx, poly]
    """
    return(__make_transfer__(im, psf_params = psf_params, mode = 'psf', dimension = 2).squeeze())

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
    return res

def PSF2ROTF(psf):
    """
        Transforms a real-valued PSF to a half-complex RFT, at the same time precompensating for the fftshift
    """
    # TODO: CHECK HERE IF FULL_SHIFT MIGTH BE NEEDED!!!
    o = image(rft(psf,shift_before=True))  # accounts for the PSF to be placed correctly
    o = o/np.max(o)
    return o.astype('complex64')

# this should go into the NanoImagingPack
def convROTF(img,otf): # should go into nip
    """
        convolves with a half-complex OTF, which can be generated using PSF2ROTF
    """
    return irft(rft(img,shift_before=False,shift_after=False) * expanddim(otf,img.ndim),shift_before=False,shift_after=False, s = img.shape)
