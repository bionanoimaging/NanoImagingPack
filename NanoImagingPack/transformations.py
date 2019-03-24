#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:35:07 2017

@author: root
"""
import numpy as np;
import numbers
import NanoImagingPack as nip;
from .config import DBG_MSG,__DEFAULTS__
# from .util import get_type;
from .image import image, extractFt
from .view5d import v5 # for debugging
import warnings;
__REAL_AXIS__ = 0;

def resampledSize(oldsize,factors,RFT=False):
    oldsize=np.array(oldsize)
    if isinstance(factors,numbers.Number):
        factors=oldsize*0.0+factors
    else:
        factors=nip.expanddimvec(factors,len(oldsize))
    newsize=np.ceil(oldsize*factors).astype("int")
    if RFT:
        RFTMirrorAx=-1
        newsize[RFTMirrorAx]=np.ceil((oldsize[RFTMirrorAx]-1)*factors[RFTMirrorAx]+1).astype("int")
    return newsize

def RFTShift(img,maxdim=3,ShiftAfter=True):
    RFTMirrorAx=-1
    ndims=img.ndim
    axes=[d for d in range(ndims)]
    oldsize=np.array(img.shape[-ndims::])
    shift1=oldsize//2 # roll along all dimensions except for the RFT-mirrored one
    shift1[RFTMirrorAx]=0;
    shift1[:-maxdim:]=0
    if ShiftAfter==False:
        shift1=-shift1
#    print(shift1)
    return np.roll(img,shift1,axes)

def resample(img,factors=[2.0,2.0]):
    """
    resamples an image by an RFT (or FFT for complex data), extracting a ROI and performing an inverse RFT. The sum of values is kept constant.

    Parameters
    ----------
    img : tensorflow array to be convolved with the PSF
    factors (default=[2.0,2.0]: resampling factor to use (approximately). A single value applies to all dimensions, a vector specifies the trailing dimensions and others are not resampled.

    Returns
    -------
    the resampled result

    See also
    -------
    Convolve, rft irft, PSF2ROTF, preFFTShift

    Example
    -------
    """
    if np.iscomplexobj(img):
        myft=nip.ft(img)
        newsize=resampledSize(img.shape,factors)
        res=nip.ift(nip.extractFt(myft,newsize,ModifyInput=True))  # the FT can be modified since it is anyway temporarily existing only
    else:
        rf=rft(img,shift_before=True) # why is the shift necessary??
        oldsize=rf.shape
#       print(oldsize)
        newrftsize=resampledSize(oldsize,factors,RFT=True)
        newsize=resampledSize(img.shape,factors)
#       print(newsize)
        rfre=resampleRFT(rf,newrftsize,newsize,ModifyInput=True)
#       print(rfre)
        res=irft(rfre,newsize,shift_after=True)  # why is the shift necessary??
    res.pixelsize = img.pixelsize * np.array(img.shape) / np.array(res.shape) # since it was messed up before!
# no modification is needed to warrant that the integral does not change!
    return res

def resampleRFT(img, newrftsize, newfullsize, maxdim=3, ModifyInput=False):
    """
    Cuts (or expands) an RFT to the appropriate size to perform downsampling

    Parameters
    ----------
    tfin : tensorflow array to be convolved with the PSF
    newrftsize : size to cut to
    Returns
    -------
    tensorflow array
        The cut RFT

    See also
    -------
    Convolve, TFRFT, RFIRFT, PSF2ROTF, preFFTShift

    Example
    -------
    """
    RFTMirrorAx=-1
    ndims=img.ndim
    oldsize=img.shape[-ndims::]
    mycenter=np.array(oldsize)//2
    newXcenter= newrftsize[RFTMirrorAx] // 2
    mycenter[RFTMirrorAx]=newXcenter
    
    res=nip.extractFt(RFTShift(img,maxdim), newrftsize, mycenter, ModifyInput, ignoredim=img.ndim - 1)
    if (newrftsize[-1] < oldsize[-1]) and nip.iseven(newfullsize[-1]): # the slice corresponds to both sides of the fourier transform as a sum
        aslice=nip.subslice(res,-1,-1)
        res=nip.subsliceAsg(res,-1,-1,aslice*2.0)   # distribute it evenly, also to keep parseval happy and real arrays real

    return RFTShift(res,maxdim,ShiftAfter=False) # this can probably be done more efficiently directly in the rft


def downsampleConvolveROTF(img, rotf, newfullsize, maxdim=3):
    """
    performs a downsampling (determined by the rotf size) and a convolution based on a real space input and a half-complex Fourier transformed PSF, called the otf.

    This convolution is faster than full-complex FFT based convolution.
    Important: This function has an overwritten gradient implementation, to make it faster and workable also for 3D, since tensorflow has problems with the 3D rft.
    For this reason, the seoncond argument (rotf) has NOT been implemented, and just a dummy value is returned. This means:
    DO NOT USE THIS FUNCTION, if the OTF is unknown (to be reconstructed), use ConvolveReal instead!

    Parameters
    ----------
    img: img to be convolved with the PSF
    rotf: half-complex optical transfer function of psf (rotf=TFROTF(preFFTShift(PSF)))
    newfullsize: size in real space of the result (i.e. size of the psf)
    maxdim : maximum numer of dimensions to transform (counting from the right)
    Returns
    -------
    tensorflow array
        The convolved array

    See also
    -------
    Convolve, rft, irft, PSF2ROTF, preFFTShift

    Example
    -------

    """
    #    def grad(dy, variables=['tfin','otf']):

    myrft = rft3d(img, maxdim,doWarn=False)
    res = resampleRFT(myrft, rotf.shape, newfullsize, maxdim)
    newfullsz=res.shape[:-len(newfullsize)]+tuple(newfullsize)
    newfullsz=np.array(newfullsz)
    res = irft3d(res * rotf, newfullsz, maxdim,doWarn=False)
    res.pixelsize = img.pixelsize * np.array(img.shape) / np.array(res.shape) # since it was messed up before!
    return res

# TODO: After Rainers newest version shift and shift_before True for both, ift and ft -> is this ok???
def ft2d(im, shift_after = True, shift_before = True, ret ='complex', s = None, norm = "ortho"):
    """
        Perform a 2D Fourier transform of the first two dimensions only of an arbitrary stack
    """
    if np.ndim(im)<2:
        print('Too few dimensions');
        return(im);
    else:
        return(ft(im, shift_after= shift_after, shift_before= shift_before, ret = ret, axes = (-2, -1), s = s, norm = norm));

def ift2d(im, shift_after = True, shift_before = True, ret ='complex', s = None, norm = "ortho"):
    """
        Perform a 2D inverse Fourier transform of the first two dimensions only of an arbitrary stack
    """
    
    if np.ndim(im)<2:
        print('Too few dimensions');
        return(im);
    else:
        return(ift(im, shift_after= shift_after, shift_before= shift_before, ret = ret, axes = (-2, -1), s = s, norm = norm));

def ft3d(im, shift_after = True, shift_before = True, ret ='complex', s = None, norm = "ortho"):
    """
        Perform a 3D Fourier transform of the first two dimensions only of an arbitrary stack
    """
    if np.ndim(im)<3:
        print('Too few dimensions');
        return(im);
    else:
        return(ft(im, shift_after= shift_after, shift_before= shift_before, ret = ret, axes = (-3, -2, -1), s = s, norm = norm));

def ift3d(im, shift_after = True, shift_before = True, ret ='complex', s = None, norm = "ortho"):
    """
        Perform a 3D inverse Fourier transform of the first two dimensions only of an arbitrary stack
    """
    
    if np.ndim(im)<3:
        print('Too few dimensions');
        return(im);
    else:
        return(ift(im, shift_after= shift_after, shift_before= shift_before, ret = ret, axes = (-3, -2, -1), s = s, norm = norm));

# now the rft abbreviations:

def rft2d(im, shift_after=False, shift_before=False, ret='complex', s=None, norm=None, doWarn=True):
    """
        Perform a 2D Fourier transform of the first two dimensions only of an arbitrary stack
    """
    axes=(-2, -1)
    ndims=np.ndim(im)
    if ndims < 2:
        if doWarn:
            print('rft2d Warning: Less than 2 dimensions in input image');
        axes=axes[-ndims:]
    return (rft(im, shift_after=shift_after, shift_before=shift_before, ret=ret, axes=axes, s=s, norm=norm));


def irft2d(im, newsize, shift_after=False, shift_before=False, ret='complex', norm=None, doWarn=True):
    """
        Perform a 2D inverse Fourier transform of the first two dimensions only of an arbitrary stack
    """
    newsize=nip.expanddimvec(newsize,3)
    axes=(-2, -1)
    ndims=np.ndim(im)
    if ndims < 2:
        if doWarn:
            print('irft2d Warning: Less than 2 dimensions in input image');
        axes=axes[-ndims:]
        newsize = newsize[-ndims:]

    return (irft(im, newsize, shift_after=shift_after, shift_before=shift_before, ret=ret, axes=axes, norm=norm));


def rft3d(im, shift_after=False, shift_before=False, ret='complex', s=None, norm=None, doWarn=True):
    """
        Perform a 3D Fourier transform of the first two dimensions only of an arbitrary stack
    """
    axes=(-3, -2, -1)
    ndims=np.ndim(im)
    if ndims < 3:
        if doWarn:
            print('rft3d Warning: Less than 3 dimensions in input image');
        axes=axes[-ndims:]
    return (rft(im, shift_after=shift_after, shift_before=shift_before, ret=ret, axes=axes, s=s, norm=norm));


def irft3d(im, newsize, shift_after=False, shift_before=False, ret='complex', norm=None, doWarn=True):
    """
        Perform a 3D inverse Fourier transform of the first two dimensions only of an arbitrary stack
    """
    newsize=nip.expanddimvec(newsize,3)
    axes = (-3, -2, -1)
    ndims=np.ndim(im)
    if ndims < 3:
        if doWarn:
            print('irft3d Warning: Less than 3 dimensions in input image');
        axes = axes[-ndims:]
        newsize = newsize[-ndims:]
    return (irft(im, newsize, shift_after=shift_after, shift_before=shift_before, ret=ret, axes=axes, norm=norm));


def __ret_val__(im, mode):
    if mode == 'abs':
        return(np.abs(im));
    elif mode == 'phase':
        return(np.angle(im));
    elif mode == 'real':
        return(np.real(im));
    elif mode == 'imag':
        return(np.imag(im));
    else:
        return(im);
def __fill_real_return__(im, ax, real_return, origi_shape):
    """
        if real_return == 'full',
        this function makes out of a rfft result an fft result
    """
    
    if real_return == 'full':
        if type(ax) == tuple:
           ax = list(ax); 
        axis = ax[-1];       # axis of rfft;
        ax = ax[:-1];        # axis of fft

        half = im.swapaxes(axis, -1);
        if np.mod(origi_shape[axis],2) == 0:     
            half = np.flipud(np.conjugate(half[1:-1]));
        else:
            half = np.flipud(np.conjugate(half[1:]));
        half = half.swapaxes(axis, -1);
        if len(ax)>0:
            for a in ax:
                half = half.swapaxes(a,-1);
                half = half[::-1];              # Reverse the other axis since the real fft is point symmetric
                half = np.roll(half, 1,0);      # for some reason one has to roll the axis, otherwise there will be one point wrong :(
                half = half.swapaxes(a,-1);
        return(np.concatenate((im,half), axis));
    else:
        return(im);
def __check_type__(im, ft_axes, orig, name, real_axis =0, shift_axes = []):
    """
        Check the data dtype and potentially change pixelsizes
    """
    from .image import image;
    if type(orig) == image:
        im = im.view(image);            # note: view casting -> this is not the viewer!
        if type(orig.name) is str:
            im.name = name + ' of '+orig.name;
        im.info = orig.info;
        pxs = [];
        
        for a in ft_axes:
            if a not in orig.spectral_axes:
                im.spectral_axes+=[a];
        im.shift_axes= shift_axes;
        if type(orig.unit) is str:
            im.unit = '';
        for i in range(im.ndim):
            if i in ft_axes:
                if name == 'IRFT' and real_axis == i:
                    pxs += [1/(orig.pixelsize[i]*2*(orig.shape[i]-1))];
                else:
                    pxs += [1/(orig.pixelsize[i]*orig.shape[i])];
                if type(orig.unit) is str:
                    im.unit += orig.unit+'^-1 ';
            else:
                try:    # TODO: FIX THIS!!!
                    pxs += [orig.pixelsize[i]];
                except:
                    print('Error in setting pixel size')
                if type(orig.unit) is str:
                    im.unit += orig.unit+' ';
        im.pixelsize = pxs;
        return(im);
    else:
        return(im);
          
        
        # ifft shift        

def __checkAxes__(axes,im):
    """
    checks axes. If None, all axes are meant and a list of axes is created. A single int number is also cast to a list.
    :param axes:
    :return: axes
    """
    if axes is None:
            axes = list(range(len(im.shape)));
    if isinstance(axes, int):
        axes = [axes];
    # TODO: What was the reason for that?
    # try:
    #     if np.issubdtype(axes.dtype, np.integer):
    #         axes = [axes];
    # except AttributeError:
    #     pass;
    return axes

def ft(im, shift_after = True, shift_before = True, ret ='complex', axes = None, s = None, norm = 'ortho'):
    """
        Fouriertransform of image

        M - Incomming matrix
        shift -shift AFTER transformation? (True, False, 'DEFAULT')
        shift_before - shift BEFORE transformation (True, False, 'DEFAULT')

        ret - What to return
                (string values: complex , abs , phase, real, imag, 'DEFAULT')
        axes - axes over which to compute the FT -> give as tupel, list or int
                    e.g. axes = (0,2) computes over axes 0 and 2
        s - Shape (length of each transformed axis) of the output (s[0] referes to axis 0, s[1] to axis 1 etc.)
            Along any axis if the given shape is smaller than tht of the input the input is cropped, if larger its padded with zeros
            If not given, the shape of the input axes specified by axes is used

        norm: Normalization mode, None or 'ortho' or 'DEFAULT' . Refer to np.fft help for further information

                None:
                    IFT and FT scale differently
                    value at zero frequency contains number of photons in the image. Can then be used as OTF

                'ortho' (default)
                    IFT and FT scale identical
                    Value at zero freq. gives sqrt(number photons ) (check!!!)


    """
    #create axes list
    axes=__checkAxes__(axes,im)

    if shift_before == True:
        im=np.fft.ifftshift(im, axes=axes) # mid to corner
    im=np.fft.fftn(im, axes=axes, s=s, norm=norm)
    if shift_after == True:
        im=np.fft.fftshift(im, axes=axes)  # corner freq to mid freq
    return image(__ret_val__(im, ret))

def rft(im, shift_after = False, shift_before = False, ret = 'complex', axes = None,  s = None, norm = None):
    """
        real Fouriertransform of image. Note the real axis is always the last (of the given) axes

        M - Incomming matrix
        shift - ft shift yes or no?
        shift_before - shift BEFORE transformation (True, False)

        ret - What to return
                (string values: complex (default), abs (default), phase, real, imag)
        axes - axes over which to compute the FT -> give as tupel, list or int
                    e.g. axes = (-1,-3) computes over axes -1 (x) and -3 (z)
        s - Shape (length of each transformed axis) of the output (s[0] referes to axis 0, s[1] to axis 1 etc.)
            Along any axis if the given shape is smaller than tht of the input the input is cropped, if larger its padded with zeros
            If not given, the shape of the input axes specified by axes is used

        norm: Normalization mode, None or ortho (default). Refere to np.fft help for further information
                 None: (default)
                    IFT and FT scale differently
                    value at zero frequency contains number of photons in the image. Can then be used as OTF

                'ortho'
                    IFT and FT scale identical
                    Value at zero freq. gives sqrt(number photons ) (check!!!)

        full_shift:  Def = False If true, the shift operations will be performed over all axes (given by axes) including the real one. Otherwise the shift operations will exclude the real axis direction.

    """
    #create axes list
    axes=__checkAxes__(axes,im)
    real_axis = max(axes);              # always the last axis is the real one   as Default

    if (np.issubdtype(im.dtype, np.complexfloating)):
        raise ValueError("rft needs real-valued input")
        # warnings.warn('Input type is Complex -> using full fft');
        # return(ft(im, shift_after= shift, shift_before = shift_before, ret = ret, axes = axes, s = s, norm = norm));
    else:
        if shift_before == True:
            im=np.fft.ifftshift(im, axes=axes) # mid to corner
        im=np.fft.rfftn(im, axes=axes, s=s, norm=norm)
        if shift_after == True:
            shift_ax = [i for i in axes if i != real_axis];
            im=np.fft.fftshift(im, axes=shift_ax)  # corner freq to mid freq
        return image(__ret_val__(im, ret))

def ift(im, shift_after = True, shift_before = True, ret ='complex', axes = None, s = None, norm =  'ortho'):
    """
        Performs the inverse Fourier transform

        im is the input spectrum. Generally it is complex

        shift: Shift AFTER Backtransfomr
        shift_before: Sshift BEFORE Bakacktransform
        ret:   return type
                'complex' keep result complex (default)
                'real': cast result to real
        axes: which axes
        s Shape (like in in np.fft.ifft help)
        norm: normalization
                None:
                    IFT and FT scale differently
                    value at zero frequency contains number of photons in the image. Can then be used as OTF

                'ortho' (default)
                    IFT and FT scale identical
                    Value at zero freq. gives sqrt(number photons ) (check!!!)

        rfft:  Are rfft data supposed to be transformed back? (only half space!, shift does not apply for this axis! -> use if you were using force_full_fft == wrong and real_return wasn't 'full' and dtype of array wasn't complex and even axis was found!)
        real_axis: along which axes was the real fft done?

    """

    axes=__checkAxes__(axes,im)

    if shift_before == True:
        im=np.fft.ifftshift(im, axes=axes) # mid to corner
    im=np.fft.ifftn(im, axes=axes, s=s, norm=norm)
    if shift_after == True:
        im=np.fft.fftshift(im, axes=axes)  # corner freq to mid freq
    return image(__ret_val__(im, ret))


def irft(im, s,shift_after = False,shift_before = False, ret ='complex', axes = None,  norm = None):
    """
        Performs the inverse Fourier transform

        im is the input spectrum. Generally it is complex
        s is the shape of the output image. In this irft function it is mandatory to give!


        shift_after: Shift AFTER Backtransform
        shift_before: Shift BEFORE Bakacktransform
        ret:   return type
        axes: which axes
        norm: normalization
                None: (default)
                    IFT and FT scale differently
                    value at zero frequency contains number of photons in the image. Can then be used as OTF

                'ortho'
                    IFT and FT scale identical
                    Value at zero freq. gives sqrt(number photons ) (check!!!)

    """
    # create axis, shift_ax and real_ax
    axes=__checkAxes__(axes,im)
    real_axis = max(axes);              # always the last axis is the real one   as Default

    if shift_before == True:
        shift_ax = [i for i in axes if i != real_axis];
        im=np.fft.ifftshift(im, axes=shift_ax) # mid freq to corner
    im=np.fft.irfftn(im, axes=axes, s=s, norm=norm)
    if shift_after == True:
        im=np.fft.fftshift(im, axes=axes)  # corner to mid
    return image(__ret_val__(im, ret))
