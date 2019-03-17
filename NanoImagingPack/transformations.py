#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:35:07 2017

@author: root
"""
import numpy as np;
import numbers
import NanoImagingPack as nip;
from .config import DBG_MSG,__DEFAULTS__;
# from .util import get_type;
from .image import image;
from .view5d import v5 # for debugging
import warnings;
__REAL_AXIS__ = 0;

def resampledSize(oldsize,factors):
    oldsize=np.array(oldsize)
    RFTMirrorAx=-1
    if isinstance(factors,numbers.Number):
        factors=oldsize*0.0+factors
    else:
        factors=nip.expanddimvec(factors,len(oldsize))
    newsize=np.ceil(oldsize*factors).astype("int")
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
    '''
    resamples and image by an RFT, applying a phase factor and performing an inverse RFT. The sum of values is kept constant.

    Parameters
    ----------
    tfin : tensorflow array to be convolved with the PSF
    factors (default=[2.0,2.0]: resampling factor to use (approximately). A single value applies to all dimensions, a vector specifies the trailing dimensions and others are not resampled.

    Returns
    -------
    the resampled result

    See also
    -------
    Convolve, rft irft, PSF2ROTF, preFFTShift

    Example
    -------    
    '''
    rf=rft(img)
    oldsize=rf.shape
#    print(oldsize)
    newsize=resampledSize(oldsize,factors)
#    print(newsize)
    rfre=resampleRFT(rf,newsize)
#    print(rfre)
    res=irft(rfre)
    res*=np.sqrt(np.prod(img.shape)/np.prod(res.shape)) # to warrand that the integral does not change
    return res

def resampleRFT(img,newsize,maxdim=3,ModifyInput=False):
    '''
    Cuts (or expands) an RFT to the appropriate size to perform downsampling

    Parameters
    ----------
    tfin : tensorflow array to be convolved with the PSF
    newsize : size to cut to
    Returns
    -------
    tensorflow array
        The cut RFT

    See also
    -------
    Convolve, TFRFT, RFIRFT, PSF2ROTF, preFFTShift

    Example
    -------    
    '''
    RFTMirrorAx=-1
    ndims=img.ndim
    oldsize=img.shape[-ndims::]
    mycenter=np.array(oldsize)//2
    newXcenter=newsize[RFTMirrorAx]//2
    mycenter[RFTMirrorAx]=newXcenter
    return RFTShift(nip.extractFt(RFTShift(img,maxdim,ModifyInput),newsize,mycenter),maxdim,ShiftAfter=False) # this can probably be done more efficiently ...

def resizeft(data,factors=2):
    '''
        resizes data based on FFTs by estimating the nearest interger size in expanded Fourier space and using extractFt in Fourier space
    '''
    factors = nip.repToList(factors,data.ndim)
    intfac=[np.floor(data.shape[d] * factors[d]).astype("int32") for d in range(data.ndim)]
    if np.iscomplexobj(data):
        data=nip.ift(nip.extractFt(nip.ft(data),list(intfac),ModifyInput=True))  # the FT can be modified since it is anyway temporarily existing only
    else:
        data=np.real(nip.ift(nip.extractFt(nip.ft(data),list(intfac),ModifyInput=True)))  # the FT can be modified since it is anyway temporarily existing only
    return data

# TODO: After Rainers newest version shift and shift_before True for both, ift and ft -> is this ok???
def ft2d(im, shift = True, shift_before = True, ret = 'complex',  s = None, norm = None):
    '''
        Perform a 2D Fourier transform of the first two dimensions only of an arbitrary stack
    '''
    if np.ndim(im)<2:
        print('Too less dimensions');
        return(im);
    else:
        return(ft(im, shift = shift, shift_before= shift_before, ret = ret, axes = (-2,-1),  s = s, norm = norm));

def ift2d(im, shift = True,shift_before = True, ret ='complex', s = None, norm = None):
    '''
        Perform a 2D inverse Fourier transform of the first two dimensions only of an arbitrary stack
    '''
    
    if np.ndim(im)<2:
        print('Too less dimensions');
        return(im);
    else:
        return(ift(im, shift = shift,shift_before= shift_before,  ret = ret, axes = (-2,-1), s = s, norm = norm));

def ft3d(im, shift = True, shift_before = True, ret = 'complex',  s = None, norm = None):
    '''
        Perform a 3D Fourier transform of the first two dimensions only of an arbitrary stack
    '''
    if np.ndim(im)<3:
        print('Too less dimensions');
        return(im);
    else:
        return(ft(im, shift = shift, shift_before= shift_before, ret = ret, axes = (-3,-2,-1),  s = s, norm = norm));

def ift3d(im, shift = True,shift_before = True, ret ='complex', s = None, norm = None):
    '''
        Perform a 3D inverse Fourier transform of the first two dimensions only of an arbitrary stack
    '''
    
    if np.ndim(im)<3:
        print('Too less dimensions');
        return(im);
    else:
        return(ift(im, shift = shift,shift_before= shift_before,  ret = ret, axes = (-2,-1), s = s, norm = norm));
        
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
    '''
        if real_return == 'full',
        this function makes out of a rfft result an fft result
    '''
    
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
    '''
        Check the data dtype and potentially change pixelsizes
    '''
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

def ft(im, shift = True, shift_before = True, ret = 'complex', axes = None,  s = None, norm = None):
    '''
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
        
        norm: Normalization mode, None or 'ortho' or 'DEFAULT' . Refere to np.fft help for further information
        
                None: 
                    IFT and FT scaling differntly
                    value at zero frequency contains number of photons in the image
                
                Ortho
                    IFT and FT scale same
                    Value at zero freq. gives sqrt(number photons ) (check!!!)
                
        
    '''
    #create axes list
    if axes == None:
            axes = list(range(len(im.shape)));
    if type(axes) == int:
        axes = [axes];
    try: 
        if np.issubdtype(axes.dtype, np.integer):
            axes = [axes];
    except AttributeError:
        pass;
    if shift_before == True: 
        if shift == True:
            return image((__check_type__(__ret_val__(np.fft.fftshift((np.fft.fftn(np.fft.ifftshift(im, axes = axes), axes = axes, s = s, norm = norm)), axes = axes), ret),axes,im, 'FT', shift_axes = axes)))
        else:
            return image((__check_type__(__ret_val__(np.fft.fftn(np.fft.ifftshift(im, axes = axes), axes = axes, s = s, norm = norm), ret),axes,im, 'FT')))
    else:
        if shift == True:
            return image((__check_type__(__ret_val__(np.fft.fftshift((np.fft.fftn(im, axes = axes, s = s, norm = norm)), axes = axes), ret),axes,im, 'FT', shift_axes = axes)))
        else:
            return image((__check_type__(__ret_val__(np.fft.fftn(im, axes = axes, s = s, norm = norm), ret),axes,im, 'FT')))

def rft(im, shift = False, shift_before = False, ret = 'complex', axes = None,  s = None, norm = None, full_shift = False):
    '''
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
                None: 
                    IFT and FT scaling differntly
                    value at zero frequency contains number of photons in the image
                
                Ortho
                    IFT and FT scale same
                    Value at zero freq. gives sqrt(number photons ) (check!!!)

        full_shift:  Def = False If true, the shift operations will be performed over all axes (given by axes) including the real one. Otherwise the shift operations will exclude the real axis direction.
        
    '''
    #create axes list
    if axes is None:
            axes = list(range(len(im.shape)));
    if isinstance(axes, int):
        axes = [axes];
    real_axis = max(axes);              # always the last axis is the real one   as Default

        # TODO: What was the reason for that?
    # try:
    #     if np.issubdtype(axes.dtype, np.integer):
    #         axes = [axes];
    # except AttributeError:
    #     pass;

    if (np.issubdtype(im.dtype, np.complexfloating)):
        warnings.warn('Input type is Complex -> using full fft');
        return(ft(im, shift = shift, shift_before = shift_before, ret = ret, axes = axes, s = s, norm = norm));
    else:
        if full_shift == False:
            shift_ax = [i for i in axes if i != real_axis];
        else:
            shift_ax = axes;

        if shift_before == True:
            if shift == True:
                return image((__check_type__(__ret_val__(np.fft.fftshift((np.fft.rfftn(np.fft.ifftshift(im, axes=shift_ax), axes=axes, s=s, norm=norm)), axes=shift_ax), ret), axes, im, 'FT', shift_axes=shift_ax)))
            else:
                return image((__check_type__(__ret_val__(np.fft.rfftn(np.fft.ifftshift(im, axes=shift_ax), axes=axes, s=s, norm=norm), ret), axes, im, 'FT')))
        else:
            if shift == True:
                return image((__check_type__(__ret_val__(np.fft.fftshift((np.fft.rfftn(im, axes=axes, s=s, norm=norm)), axes=shift_ax), ret), axes, im, 'FT', shift_axes=shift_ax)))
            else:

                return image((__check_type__(__ret_val__(np.fft.rfftn(im, axes=axes, s=s, norm=norm), ret), axes, im, 'FT')))


# DEPRICATED:
    # if (np.issubdtype(im.dtype, np.floating) or np.issubdtype(im.dtype, np.integer)):
    #     if real_axis == None:
    #         real_axis = axes[-1];
    #     if real_axis not in axes:
    #         real_axis = axes[-1];
    #         DBG_MSG('Real axis is not in axes list -> taking last one (number '+str(real_axis)+')', 2);
    #     if np.mod(im.shape[real_axis],2) == 1:   # This stuff ensures that only axis with even dimension will be used for rFFT!
    #         ax_index = 0;
    #         while np.mod(im.shape[axes[ax_index]],2) == 1:
    #             ax_index+=1;
    #             if ax_index >= len(axes):
    #                 ax_index = -1;
    #                 break;
    #         if ax_index >= 0:
    #             DBG_MSG('Axis has odd shape ... taking axis '+str(axes[ax_index]),2);
    #             real_axis = axes[ax_index];
    #         else:
    #             DBG_MSG('All axis have odd dimension size -> taking full fft!!',2);
    #             real_axis = -1;
    #     __REAL_AXIS__ = real_axis;
    #     if real_axis >=0:
    #         ret_im = im.swapaxes(real_axis, axes[-1]);
    #         if shift_before == True:
    #             if real_return == 'full':
    #                 ret_im = np.fft.fftshift(ret_im, axes = axes);
    #             else:
    #                 ret_im = np.fft.fftshift(ret_im, axes = axes);  # axes[:-1]   RH 22.12.2018
    #         ret_im = np.fft.rfftn(ret_im, axes = axes, s = s, norm = norm)
    #         ret_im = __fill_real_return__(ret_im, ax =axes, real_return=real_return, origi_shape=im.shape);
    #         s_axes=[];
    #         if shift == True:
    #             if real_return == 'full':
    #                 ret_im = np.fft.fftshift(ret_im, axes = axes);
    #                 s_axes = axes;
    #             else:
    #                 ret_im = np.fft.fftshift(ret_im, axes = axes); # axes[-1]  RH 22.12.2018
    #                 s_axes = axes[:-1];
    #         ret_im = __ret_val__(ret_im, ret);
    #         ret_im = np.swapaxes(ret_im, real_axis, axes[-1]);
    #         return image((__check_type__(ret_im, axes, im, 'FT', shift_axes = s_axes)))
    #     else:
    #
    #         if shift_before == True:
    #             if shift == True:
    #                 return image((__check_type__(__ret_val__(np.fft.fftshift((np.fft.fftn(np.fft.fftshift(im, axes = axes), axes = axes, s = s, norm = norm)), axes = axes), ret),axes,im, 'FT', shift_axes = axes)))
    #             else:
    #                 return image((__check_type__(__ret_val__(np.fft.fftn(np.fft.fftshift(im, axes = axes), axes = axes, s = s, norm = norm), ret),axes,im, 'FT')))
    #         else:
    #             if shift == True:
    #                 return image((__check_type__(__ret_val__(np.fft.fftshift((np.fft.fftn(im, axes = axes, s = s, norm = norm)), axes = axes), ret),axes,im, 'FT', shift_axes = axes)))
    #             else:
    #                 return image((__check_type__(__ret_val__(np.fft.fftn(im, axes = axes, s = s, norm = norm), ret),axes,im, 'FT')))
    # else:
    #     DBG_MSG('Complex data. rft not possible. Doing FFT',2);
    #     if shift_before == True:
    #         if shift == True:
    #             return image((__check_type__(__ret_val__(np.fft.fftshift((np.fft.fftn(np.fft.fftshift(im, axes = axes), axes = axes, s = s, norm = norm)), axes = axes), ret),axes,im, 'FT', shift_axes = axes)))
    #         else:
    #             return image((__check_type__(__ret_val__(np.fft.fftn(np.fft.fftshift(im, axes = axes), axes = axes, s = s, norm = norm), ret),axes,im, 'FT')))
    #     else:
    #         if shift == True:
    #             return image((__check_type__(__ret_val__(np.fft.fftshift((np.fft.fftn(im, axes = axes, s = s, norm = norm)), axes = axes), ret),axes,im, 'FT', shift_axes = axes)))
    #         else:
    #             return image((__check_type__(__ret_val__(np.fft.fftn(im, axes = axes, s = s, norm = norm), ret),axes,im, 'FT')))

def ift(im, shift = True,shift_before = True, ret ='complex', axes = None, s = None, norm = None):
    '''
        Performs the inverse Fourier transform
        
        im is the input spectrum. Generally it is complex
        
        shift: Shift AFTER Backtransfomr
        shift_before: Sshift BEFORE Bakacktransform
        ret:   return type
        axes: which axes
        s Shape (like in in np.fft.ifft help)
        norm: normalization
        rfft:  Are rfft data supposed to be transformed back? (only half space!, shift does not apply for this axis! -> use if you were using force_full_fft == wrong and real_return wasn't 'full' and dtype of array wasn't complex and even axis was found!)
        real_axis: along which axes was the real fft done?
        
    ''' 
    

    
    if type(axes) == int:
        axes = [axes];
    try: 
        if np.issubdtype(axes.dtype, np.integer):
            axes = [axes];
    except AttributeError:
        pass;
    
    if axes== None:
        axes = list(range(len(im.shape)));
    
    if shift_before == True: 
        if shift == True:

            return image((__check_type__(__ret_val__(np.fft.fftshift((np.fft.ifftn(np.fft.ifftshift(im, axes = axes), axes = axes, s = s, norm = norm)), axes = axes), ret),axes,im, 'IFT', shift_axes = axes)))
        else:
            return image((__check_type__(__ret_val__(np.fft.ifftn(np.fft.ifftshift(im, axes = axes), axes = axes, s = s, norm = norm), ret),axes,im, 'IFT')))
    else:
        if shift == True:

            return image((__check_type__(__ret_val__(np.fft.fftshift((np.fft.ifftn(im, axes = axes, s = s, norm = norm)), axes = axes), ret),axes,im, 'IFT', shift_axes = axes)))
        else:
            return image((__check_type__(__ret_val__(np.fft.ifftn(im, axes = axes, s = s, norm = norm), ret),axes,im, 'IFT')))


def irft(im, s,shift = False,shift_before = False, ret ='complex', axes = None,  norm = None, full_shift = False):
    '''
        Performs the inverse Fourier transform
        
        im is the input spectrum. Generally it is complex
        s is the shape of the output image. In this irft function it is mandatory to give!


        shift: Shift AFTER Backtransfomr
        shift_before: Sshift BEFORE Bakacktransform
        ret:   return type
        axes: which axes
        norm: normalization

    '''
    # create axis, shift_ax and real_ax
    if axes is None:
            axes = list(range(len(im.shape)));
    if isinstance(axes, int):
        axes = [axes];
    real_axis = max(axes);              # always the last axis is the real one   as Default
    if full_shift == False:
        shift_ax = [i for i in axes if i != real_axis];
    else:
        shift_ax = axes;

    if shift_before == True:
        if shift == True:

            return image((__check_type__(__ret_val__(np.fft.fftshift((np.fft.irfftn(np.fft.ifftshift(im, axes=shift_ax), axes=axes, s=s, norm=norm)), axes=shift_ax), ret), axes, im, 'IFT', shift_axes=shift_ax)))
        else:
            return image((__check_type__(__ret_val__(np.fft.irfftn(np.fft.ifftshift(im, axes=shift_ax), axes=axes, s=s, norm=norm), ret), axes, im, 'IFT')))
    else:
        if shift == True:
            return image((__check_type__(
                __ret_val__(np.fft.fftshift((np.fft.irfftn(im, axes=axes, s=s, norm=norm)), axes=shift_ax), ret), axes, im, 'IFT', shift_axes=shift_ax)))
        else:
            return image((__check_type__(__ret_val__(np.fft.ifftn(im, axes=axes, s=s, norm=norm), ret), axes, im, 'IFT')))
