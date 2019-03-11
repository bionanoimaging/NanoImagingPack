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

__REAL_AXIS__ = 0;

def ResampledSize(oldsize,factors):
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

def Resample(img,factors=2.0):
    rf=rft(img)
    oldsize=rf.shape
#    print(oldsize)
    newsize=ResampledSize(oldsize,factors)
#    print(newsize)
    rfre=ResampleRFT(rf,newsize)
#    print(rfre)
    return irft(rfre)

def ResampleRFT(img,newsize,maxdim=3):
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
    return RFTShift(nip.centered_extract(RFTShift(img,maxdim),newsize,mycenter),maxdim,ShiftAfter=False) # this can probably be done more efficiently ...

def resizeft(data,factors=2):
    '''
        resizes data based on FFTs by estimating the nearest interger size in expanded Fourier space and using centered_extract in Fourier space
    '''
    factors = nip.repToList(factors,data.ndim)
    intfac=[np.floor(data.shape[d] * factors[d]).astype("int32") for d in range(data.ndim)]
    data=np.real(nip.ift(nip.centered_extract(nip.ft(data),list(intfac))))
    return data


def ft2d(im, shift = 'DEFAULT', shift_before = 'DEFAULT', ret = 'DEFAULT',  s = None, norm = 'DEFAULT'):
    '''
        Perform a 2D Fourier transform of the first two dimensions only of an arbitrary stack
    '''
    if np.ndim(im)<2:
        print('Too less dimensions');
        return(im);
    else:
        return(ft(im, shift = shift, shift_before= shift_before, ret = ret, axes = (-2,-1),  s = s, norm = norm));

def ift2d(im, shift = 'DEFAULT',shift_before = 'DEFAULT', ret ='DEFAULT', s = None, norm = 'DEFAULT'):
    '''
        Perform a 2D inverse Fourier transform of the first two dimensions only of an arbitrary stack
    '''
    
    if np.ndim(im)<2:
        print('Too less dimensions');
        return(im);
    else:
        return(ift(im, shift = shift,shift_before= shift_before,  ret = ret, axes = (-2,-1), s = s, norm = norm));


def ft3d(im, shift = 'DEFAULT', shift_before = 'DEFAULT', ret = 'DEFAULT',  s = None, norm = 'DEFAULT'):
    '''
        Perform a 3D Fourier transform of the first two dimensions only of an arbitrary stack
    '''
    if np.ndim(im)<3:
        print('Too less dimensions');
        return(im);
    else:
        return(ft(im, shift = shift, shift_before= shift_before, ret = ret, axes = (-3,-2,-1),  s = s, norm = norm));
        


def ift3d(im, shift = 'DEFAULT',shift_before = 'DEFAULT', ret ='DEFAULT', s = None, norm = 'DEFAULT'):
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
                pxs += [orig.pixelsize[i]];
                if type(orig.unit) is str:
                    im.unit += orig.unit+' ';
        im.pixelsize = pxs;
        return(im);
    else:
        return(im);
          
        
        # ifft shift        

def ft(im, shift = 'DEFAULT', shift_before = 'DEFAULT', ret = 'DEFAULT', axes = None,  s = None, norm = 'DEFAULT'):
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
    if shift == 'DEFAULT': shift = __DEFAULTS__['FT_SHIFT'];
    if shift_before == 'DEFAULT': shift_before = __DEFAULTS__['FT_SHIFT_FIRST'];
    if ret == 'DEFAULT': ret = __DEFAULTS__['FT_RETURN'];
    if norm == 'DEFAULT': norm = __DEFAULTS__['FT_NORM'];
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
            return image((__check_type__(__ret_val__(np.fft.fftshift((np.fft.fftn(np.fft.fftshift(im, axes = axes), axes = axes, s = s, norm = norm)), axes = axes), ret),axes,im, 'FT', shift_axes = axes)))
        else:
            return image((__check_type__(__ret_val__(np.fft.fftn(np.fft.fftshift(im, axes = axes), axes = axes, s = s, norm = norm), ret),axes,im, 'FT')))
    else:
        if shift == True:
            return image((__check_type__(__ret_val__(np.fft.fftshift((np.fft.fftn(im, axes = axes, s = s, norm = norm)), axes = axes), ret),axes,im, 'FT', shift_axes = axes)))
        else:
            return image((__check_type__(__ret_val__(np.fft.fftn(im, axes = axes, s = s, norm = norm), ret),axes,im, 'FT')))

def rft(im, shift = 'DEFAULT', shift_before = 'DEFAULT', ret = 'DEFAULT', axes = None,  s = None, norm = 'DEFAULT', real_return = 'DEFAULT', real_axis = None):
    '''
        real Fouriertransform of image
        
        M - Incomming matrix
        shift - ft shift yes or no?
        shift_before - shift BEFORE transformation (True, False, 'DEFAULT')
        
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

        real_return: if the input is real, rfft is used inf real_return == 'full' the spectrum will be filled up as it would be complex spectrum
        
        
        real_axis: along which axis you want to perform the rfft
        
    '''
    global __REAL_AXIS__;
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
    if shift == 'DEFAULT': shift = __DEFAULTS__['RFT_SHIFT'];
    if shift_before == 'DEFAULT': shift_before = __DEFAULTS__['RFT_SHIFT_FIRST'];
    if ret == 'DEFAULT': ret = __DEFAULTS__['RFT_RETURN'];
    if norm == 'DEFAULT': norm = __DEFAULTS__['RFT_NORM'];

    if (np.issubdtype(im.dtype, np.floating) or np.issubdtype(im.dtype, np.integer)):
        if real_axis == None:
            real_axis = axes[-1];
        if real_axis not in axes:
            real_axis = axes[-1];
            DBG_MSG('Real axis is not in axes list -> taking last one (number '+str(real_axis)+')', 2);
        if np.mod(im.shape[real_axis],2) == 1:   # This stuff ensures that only axis with even dimension will be used for rFFT!
            ax_index = 0;
            while np.mod(im.shape[axes[ax_index]],2) == 1:
                ax_index+=1;
                if ax_index >= len(axes):
                    ax_index = -1;
                    break;
            if ax_index >= 0:        
                DBG_MSG('Axis has odd shape ... taking axis '+str(axes[ax_index]),2);
                real_axis = axes[ax_index];
            else:
                DBG_MSG('All axis have odd dimension size -> taking full fft!!',2);
                real_axis = -1;
        __REAL_AXIS__ = real_axis;
        if real_axis >=0:
            ret_im = im.swapaxes(real_axis, axes[-1]);     
            if shift_before == True:
                if real_return == 'full':
                    ret_im = np.fft.fftshift(ret_im, axes = axes);
                else:
                    ret_im = np.fft.fftshift(ret_im, axes = axes);  # axes[:-1]   RH 22.12.2018 
            ret_im = np.fft.rfftn(ret_im, axes = axes, s = s, norm = norm)                        
            ret_im = __fill_real_return__(ret_im, ax =axes, real_return=real_return, origi_shape=im.shape);
            s_axes=[];
            if shift == True:
                if real_return == 'full':
                    ret_im = np.fft.fftshift(ret_im, axes = axes);
                    s_axes = axes;
                else:
                    ret_im = np.fft.fftshift(ret_im, axes = axes); # axes[-1]  RH 22.12.2018 
                    s_axes = axes[:-1];
            ret_im = __ret_val__(ret_im, ret);
            ret_im = np.swapaxes(ret_im, real_axis, axes[-1]);
            return image((__check_type__(ret_im, axes, im, 'FT', shift_axes = s_axes)))
        else:
            
            if shift_before == True: 
                if shift == True:
                    return image((__check_type__(__ret_val__(np.fft.fftshift((np.fft.fftn(np.fft.fftshift(im, axes = axes), axes = axes, s = s, norm = norm)), axes = axes), ret),axes,im, 'FT', shift_axes = axes)))
                else:
                    return image((__check_type__(__ret_val__(np.fft.fftn(np.fft.fftshift(im, axes = axes), axes = axes, s = s, norm = norm), ret),axes,im, 'FT')))
            else:
                if shift == True:
                    return image((__check_type__(__ret_val__(np.fft.fftshift((np.fft.fftn(im, axes = axes, s = s, norm = norm)), axes = axes), ret),axes,im, 'FT', shift_axes = axes)))
                else:
                    return image((__check_type__(__ret_val__(np.fft.fftn(im, axes = axes, s = s, norm = norm), ret),axes,im, 'FT')))
    else:
        DBG_MSG('Complex data. rft not possible. Doing FFT',2);
        if shift_before == True: 
            if shift == True:
                return image((__check_type__(__ret_val__(np.fft.fftshift((np.fft.fftn(np.fft.fftshift(im, axes = axes), axes = axes, s = s, norm = norm)), axes = axes), ret),axes,im, 'FT', shift_axes = axes)))
            else:
                return image((__check_type__(__ret_val__(np.fft.fftn(np.fft.fftshift(im, axes = axes), axes = axes, s = s, norm = norm), ret),axes,im, 'FT')))
        else:
            if shift == True:
                return image((__check_type__(__ret_val__(np.fft.fftshift((np.fft.fftn(im, axes = axes, s = s, norm = norm)), axes = axes), ret),axes,im, 'FT', shift_axes = axes)))
            else:
                return image((__check_type__(__ret_val__(np.fft.fftn(im, axes = axes, s = s, norm = norm), ret),axes,im, 'FT')))


def ift(im, shift = 'DEFAULT',shift_before = 'DEFAULT', ret ='DEFAULT', axes = None, s = None, norm = 'DEFAULT'):
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
    
    if shift == 'DEFAULT': shift = __DEFAULTS__['IFT_SHIFT'];
    if shift_before == 'DEFAULT': shift_before = __DEFAULTS__['IFT_SHIFT_FIRST'];
    if ret == 'DEFAULT': ret = __DEFAULTS__['IFT_RETURN'];
    if norm == 'DEFAULT': norm = __DEFAULTS__['IFT_NORM'];

    
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
            return image((__check_type__(__ret_val__(np.fft.ifftshift((np.fft.ifftn(np.fft.ifftshift(im, axes = axes), axes = axes, s = s, norm = norm)), axes = axes), ret),axes,im, 'IFT', shift_axes = axes)))
        else:
            return image((__check_type__(__ret_val__(np.fft.ifftn(np.fft.ifftshift(im, axes = axes), axes = axes, s = s, norm = norm), ret),axes,im, 'IFT')))
    else:
        if shift == True:
            return image((__check_type__(__ret_val__(np.fft.ifftshift((np.fft.ifftn(im, axes = axes, s = s, norm = norm)), axes = axes), ret),axes,im, 'IFT', shift_axes = axes)))
        else:
            return image((__check_type__(__ret_val__(np.fft.ifftn(im, axes = axes, s = s, norm = norm), ret),axes,im, 'IFT')))
        

def irft(im, shift = 'DEFAULT',shift_before = 'DEFAULT', ret ='DEFAULT', axes = None, s = None, norm = 'DEFAULT', real_axis = 'DEFAULT'):
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
                    Can be None:
                        The last axis will be taken
                    Can be 'GLOBAL'
                        A stored axis will be taken
                    Can be 'DEFAULT'
                        Default value will be used
        
    ''' 
    
    if shift == 'DEFAULT': shift = __DEFAULTS__['IRFT_SHIFT'];
    if shift_before == 'DEFAULT': shift_before = __DEFAULTS__['IRFT_SHIFT_FIRST'];
    if ret == 'DEFAULT': ret = __DEFAULTS__['IRFT_RETURN'];
    if norm == 'DEFAULT': norm = __DEFAULTS__['IRFT_NORM'];
    if real_axis == 'DEFAULT': real_axis = __DEFAULTS__['IRFT_REAL_AXIS'];
    
    if type(axes) == int:
        axes = [axes];
    try: 
        if np.issubdtype(axes.dtype, np.integer):
            axes = [axes];
    except AttributeError:
        pass;
    
    if axes== None:
        axes = list(range(len(im.shape)));
    
    if  real_axis == None:
        real_axis = axes[len(axes)-1];
        print('No real axis given -> taking last one (number: '+str(real_axis)+')');
    elif real_axis == 'GLOBAL':
        real_axis = __REAL_AXIS__;
        print('Taking real axis from global variable (number: '+str(real_axis)+')!');

    if real_axis in axes:
        im = im.swapaxes(axes[-1], real_axis)   
        s_axes =[];
        if shift_before == True:        
            if shift:
                #iftim = __ret_val__(np.fft.ifftshift(np.fft.irfftn(np.fft.ifftshift(im,axes = axes[:-1]), norm = norm, axes = axes, s = s),axes = axes[:-1]), ret);
                iftim = __ret_val__(np.fft.ifftshift(np.fft.irfftn(np.fft.ifftshift(im,axes = axes[:-1]), norm = norm, axes = axes, s = s),axes = axes), ret);
                s_axes = axes[:-1];
            else:
                iftim = __ret_val__(np.fft.irfftn(np.fft.ifftshift(im,axes = axes[:-1]), norm = norm, axes = axes, s = s), ret);
        else:    
            if shift:
                #iftim = __ret_val__(np.fft.ifftshift(np.fft.irfftn(im, norm = norm, axes = axes, s = s),axes = axes[:-1]), ret);
                iftim = __ret_val__(np.fft.ifftshift(np.fft.irfftn(im, norm = norm, axes = axes, s = s),axes = axes), ret);
                s_axes = axes;
            else:
                iftim = __ret_val__(np.fft.irfftn(im, norm = norm, axes = axes, s = s), ret);
        return image(__check_type__(iftim.swapaxes(axes[-1], real_axis),axes,im, 'IRFT', axes[-1], shift_axes = s_axes))
    else:
        print('Real axis not in ift axes list -> Performing normal ifft!')
        return image(ift(im, shift = shift,shift_before = shift_before, ret =ret, axes = axes, s = s, norm =norm))
     


#def irft(im, shift = 'DEFAULT',shift_before = 'DEFAULT', ret ='DEFAULT', axes = None, s = None, norm = 'DEFAULT', real_axis = 'DEFAULT'):
#    '''
#        Performs the inverse Fourier transform
#        
#        im is the input spectrum. Generally it is complex
#        
#        shift: Shift AFTER Backtransfomr
#        shift_before: Sshift BEFORE Bakacktransform
#        ret:   return type
#        axes: which axes
#        s Shape (like in in np.fft.ifft help)
#        norm: normalization
#        rfft:  Are rfft data supposed to be transformed back? (only half space!, shift does not apply for this axis! -> use if you were using force_full_fft == wrong and real_return wasn't 'full' and dtype of array wasn't complex and even axis was found!)
#        real_axis: along which axes was the real fft done? 
#                    Can be None:
#                        The last axis will be taken
#                    Can be 'GLOBAL'
#                        A stored axis will be taken
#                    Can be 'DEFAULT'
#                        Default value will be used
#        
#    ''' 
#    
#    if shift == 'DEFAULT': shift = __DEFAULTS__['IFT_SHIFT'];
#    if shift_before == 'DEFAULT': shift_before = __DEFAULTS__['IFT_SHIFT_FIRST'];
#    if ret == 'DEFAULT': ret = __DEFAULTS__['IFT_RETURN'];
#    if norm == 'DEFAULT': norm = __DEFAULTS__['IFT_NORM'];
#    if real_axis == 'DEFAULT': real_axis = __DEFAULTS__['IFT_REAL_AXIS'];
#    
#    if type(axes) == int:
#        axes = [axes];
#    try: 
#        if np.issubdtype(axes.dtype, np.integer):
#            axes = [axes];
#    except AttributeError:
#        pass;
#    
#    if axes== None:
#        axes = list(range(len(im.shape)));
#    
#    if  real_axis == None:
#        real_axis = axes[len(axes)-1];
#        import copy;                             
#        shift_ax = copy.copy(axes);
#        shift_ax.remove(real_axis);
#        if shift_ax == None: shift_ax = [];            
#        print('No real axis given -> taking last one (number: '+str(real_axis)+')');
#    elif real_axis == 'GLOBAL':
#        real_axis = __REAL_AXIS__;
#        print('Taking real axis from global variable!');
#
#    try:
#        import copy;                             
#        shift_ax = copy.copy(axes);
#        shift_ax = shift_ax.remove(real_axis);
#    except ValueError:
#        print('Real axis not in ift axes list -> Performing normal ifft!')
#        if shift_before == True: 
#            if shift == True:
#                return(__check_type__(__ret_val__(np.fft.ifftshift((np.fft.ifftn(np.fft.ifftshift(im, axes = axes), axes = axes, s = s, norm = norm)), axes = axes), ret),axes,im, 'IFT'));
#            else:
#                return(__check_type__(__ret_val__(np.fft.ifftn(np.fft.ifftshift(im, axes = axes), axes = axes, s = s, norm = norm), ret),axes,im, 'IFT'));
#        else:
#            if shift == True:
#                return(__check_type__(__ret_val__(np.fft.ifftshift((np.fft.ifftn(im, axes = axes, s = s, norm = norm)), axes = axes), ret),axes,im, 'IFT'));
#            else:
#                return(__check_type__(__ret_val__(np.fft.ifftn(im, axes = axes, s = s, norm = norm), ret),axes,im, 'IFT'));
#
#        
#    if shift_before == True:        
#        
#        if shift:
#            im = im.swapaxes(axes[-1], real_axis)    
#            iftim = __ret_val__(np.fft.ifftshift(np.fft.irfftn(np.fft.ifftshift(im,axes = shift_ax), norm = norm, axes = axes, s = s),axes = shift_ax), ret);
#            return(__check_type__(iftim.swapaxes(axes[-1], real_axis),axes,im, 'IRFT', axes[-1]))
#        else:
#            im = im.swapaxes(axes[-1], real_axis)    
#            iftim = __ret_val__(np.fft.irfftn(np.fft.ifftshift(im,axes = shift_ax), norm = norm, axes = axes, s = s), ret);
#            return(__check_type__(iftim.swapaxes(axes[-1], real_axis),axes,im, 'IRFT', axes[-1]))
#    else:    
#        if shift:
#            im = im.swapaxes(axes[-1], real_axis)    
#            iftim = __ret_val__(np.fft.ifftshift(np.fft.irfftn(im, norm = norm, axes = axes, s = s),axes = shift_ax), ret);
#            return(__check_type__(iftim.swapaxes(axes[-1], real_axis),axes,im, 'IRFT', axes[-1]))
#        else:
#            im = im.swapaxes(axes[-1], real_axis);
#            return(__check_type__(np.swapaxes(np.fft.irfftn(im, norm = norm, axes = axes, s = s), axes[-1], real_axis),axes,im, 'IRFT', axes[-1]));
#        

'''
 BELOW: DEPRICATED CODE!
'''
#
#def ft(M, shift = True, real = False, ret = 'abs', axes =None, s = None, norm = None ):
#    '''
#        Fouriertransform of image
#        
#        M - Incomming matrix
#        shift - ft shift yes or no?
#        real - Real input FT?
#        ret - What to return 
#                (string values: abs (default), phase, real, imag, complex)
#        axes - axes over which to compute the FT -> give as tupel 
#                    e.g. axes = (-1,-3) computes over axes -1 (x) and -3 (z)
#        s - Shape (length of each transformed axis) of the output (s[0] referes to axis 0, s[1] to axis 1 etc.)
#            Along any axis if the given shape is smaller than tht of the input the input is cropped, if larger its padded with zeros
#            If not given, the shape of the input axes specified by axes is used
#        
#        norm: Normalization mode, None or ortho. Refere to np.fft help for further information
#
#    '''
#    
#    import numpy as np;
#    if (M.ndim >1) and (type(axes)==tuple or axes == None):
#        if real:
#            if shift:
#                if ret == 'abs':
#                    return(np.abs(np.fft.fftshift(np.fft.rfftn(M,s = s, axes = axes, norm = norm),axes = axes)));
#                elif ret == 'phase':
#                    return(np.angle(np.fft.fftshift(np.fft.rfftn(M,s = s, axes = axes, norm = norm),axes = axes),False));
#                elif ret == 'real':
#                    return(np.real(np.fft.fftshift(np.fft.rfftn(M,s = s, axes = axes, norm = norm),axes = axes)));
#                elif ret == 'imag':
#                    return(np.imag(np.fft.fftshift(np.fft.rfftn(M,s = s, axes = axes, norm = norm),axes = axes)));
#                elif ret == 'complex':
#                    return(np.fft.fftshift(np.fft.fftn(M,s = s, axes = axes, norm = norm)));
#                else:
#                    print('Return value not known! Use "abs", "phase", "real", "imag" or "complex"');
#                    return(M);
#            else:
#                if ret == 'abs':
#                    return(np.abs(np.fft.rfftn(M,s = s, axes = axes, norm = norm)));
#                elif ret == 'phase':
#                    return(np.angle(np.fft.rfftn(M,s = s, axes = axes, norm = norm)),False);
#                elif ret == 'real':
#                    return(np.real(np.fft.rfftn(M,s = s, axes = axes, norm = norm)));
#                elif ret == 'imag':
#                    return(np.imag(np.fft.rfftn(M,s = s, axes = axes, norm = norm)));
#                elif ret == 'complex':
#                    return(np.fft.rfftn(M,s = s, axes = axes, norm = norm));
#                else:
#                    print('Return value not known! Use "abs", "phase", "real", "imag" or "complex"');
#                    return(M);
#        else:
#            if shift:
#                if ret == 'abs':
#                    return(np.abs(np.fft.fftshift(np.fft.fftn(M,s = s, axes = axes, norm = norm),axes = axes)));
#                elif ret == 'phase':
#                    return(np.angle(np.fft.fftshift(np.fft.fftn(M,s = s, axes = axes, norm = norm),axes = axes),False));
#                elif ret == 'real':
#                    return(np.real(np.fft.fftshift(np.fft.fftn(M,s = s, axes = axes, norm = norm),axes = axes)));
#                elif ret == 'imag':
#                    return(np.imag(np.fft.fftshift(np.fft.fftn(M,s = s, axes = axes, norm = norm),axes = axes)));
#                elif ret == 'complex':
#                    return(np.fft.fftshift(np.fft.fftn(M,s = s, axes = axes, norm = norm),axes = axes));
#                else:
#                    print('Return value not known! Use "abs", "phase", "real", "imag" or "complex"');
#                    return(M);
#            else:
#                if ret == 'abs':
#                    return(np.abs(np.fft.fftn(M,s = s, axes = axes, norm = norm)));
#                elif ret == 'phase':
#                    return(np.angle(np.fft.fftn(M,s = s, axes = axes, norm = norm)),False);
#                elif ret == 'real':
#                    return(np.fft.fftshift(np.fft.fftn(M,s = s, axes = axes, norm = norm)));
#                elif ret == 'imag':
#                    return(np.imag(np.fft.fftn(M,s = s, axes = axes, norm = norm)));
#                elif ret == 'complex':
#                    return(np.fft.fftn(M,s = s, axes = axes, norm = norm));
#                else:
#                    print('Return value not known! Use "abs", "phase", "real", "imag" or "complex"');
#                    return(M);
#    else:
#        if axes == None:
#            axes = -1;
#        if real:
#            if shift:
#                if ret == 'abs':
#                    return(np.abs(np.fft.fftshift(np.fft.rfft(M,n = s, axis = axes, norm = norm),axes = axes)));
#                elif ret == 'phase':
#                    return(np.angle(np.fft.fftshift(np.fft.rfft(M,n = s, axis = axes, norm = norm),axes = axes),False));
#                elif ret == 'real':
#                    return(np.real(np.fft.fftshift(np.fft.rfft(M,n = s, axis = axes, norm = norm),axes = axes)));
#                elif ret == 'imag':
#                    return(np.imag(np.fft.fftshift(np.fft.rfft(M,n = s, axis = axes, norm = norm),axes = axes)));
#                elif ret == 'complex':
#                    return(np.fft.fftshift(np.fft.fft(M,n = s, axis = axes, norm = norm),axes = axes));
#                else:
#                    print('Return value not known! Use "abs", "phase", "real", "imag" or "complex"');
#                    return(M);
#            else:
#                if ret == 'abs':
#                    return(np.abs(np.fft.rfft(M,n = s, axis = axes, norm = norm)));
#                elif ret == 'phase':
#                    return(np.angle(np.fft.rfft(M,n = s, axis = axes, norm = norm)),False);
#                elif ret == 'real':
#                    return(np.real(np.fft.rfft(M,n = s, axis = axes, norm = norm)));
#                elif ret == 'imag':
#                    return(np.imag(np.fft.rfft(M,n = s, axis = axes, norm = norm)));
#                elif ret == 'complex':
#                    return(np.fft.rfft(M,n = s, axis = axes, norm = norm));
#                else:
#                    print('Return value not known! Use "abs", "phase", "real", "imag" or "complex"');
#                    return(M);
#        else:
#            if shift:
#                if ret == 'abs':
#                    return(np.abs(np.fft.fftshift(np.fft.fft(M,n = s, axis = axes, norm = norm))));
#                elif ret == 'phase':
#                    return(np.angle(np.fft.fftshift(np.fft.fft(M,n = s, axis = axes, norm = norm)),False));
#                elif ret == 'real':
#                    return(np.real(np.fft.fftshift(np.fft.fft(M,n = s, axis = axes, norm = norm))));
#                elif ret == 'imag':
#                    return(np.imag(np.fft.fftshift(np.fft.fft(M,n = s, axis = axes, norm = norm))));
#                elif ret == 'complex':
#                    return(np.fft.fftshift(np.fft.fft(M,n = s, axis = axes, norm = norm)));
#                else:
#                    print('Return value not known! Use "abs", "phase", "real", "imag" or "complex"');
#                    return(M);
#            else:
#                if ret == 'abs':
#                    return(np.abs(np.fft.fft(M,n = s, axis = axes, norm = norm)));
#                elif ret == 'phase':
#                    return(np.angle(np.fft.fft(M,n = s, axis = axes, norm = norm)),False);
#                elif ret == 'real':
#                    return(np.fft.fftshift(np.fft.fft(M,n = s, axis = axes, norm = norm)));
#                elif ret == 'imag':
#                    return(np.imag(np.fft.fft(M,n = s, axis = axes, norm = norm)));
#                elif ret == 'complex':
#                    return(np.fft.fft(M,n = s, axis = axes, norm = norm));
#                else:
#                    print('Return value not known! Use "abs", "phase", "real", "imag" or "complex"');
#                    return(M);
#
#def ift(M, shift = False, real = False, ret = 'abs', axes =None, s = None, norm = None ):
#    '''
#        Inverse fouriertransform of image
#        
#        M - Incomming matrix
#        shift - ft shift yes or no?
#        real - Real input FT?
#        ret - What to return 
#                (string values: abs (default), phase, real, imag, complex)
#        axes - axes over which to compute the FT -> give as tupel 
#                    e.g. axes = (0,2) computes over axes 0 and 2
#        s - Shape (length of each transformed axis) of the output (s[0] referes to axis 0, s[1] to axis 1 etc.)
#            Along any axis if the given shape is smaller than tht of the input the input is cropped, if larger its padded with zeros
#            If not given, the shape of the input axes specified by axes is used
#        
#        norm: Normalization mode, None or ortho. Refere to np.fft help for further information
#
#    '''
#    import numpy as np;
#    if (M.ndim >1) and (type(axes)==tuple or axes == None):
#        if real:
#            if shift:
#                if ret == 'abs':
#                    return(np.abs(np.fft.ifftshift(np.fft.irfftn(M,s = s, axes = axes, norm = norm),axes = axes)));
#                elif ret == 'phase':
#                    return(np.angle(np.fft.ifftshift(np.fft.irfftn(M,s = s, axes = axes, norm = norm),axes = axes),False));
#                elif ret == 'real':
#                    return(np.real(np.fft.ifftshift(np.fft.irfftn(M,s = s, axes = axes, norm = norm),axes = axes)));
#                elif ret == 'imag':
#                    return(np.imag(np.fft.ifftshift(np.fft.irfftn(M,s = s, axes = axes, norm = norm),axes = axes)));
#                elif ret == 'complex':
#                    return(np.fft.ifftshift(np.fft.irfftn(M,s = s, axes = axes, norm = norm),axes = axes));
#                else:
#                    print('Return value not known! Use "abs", "phase", "real", "imag" or "complex"');
#                    return(M);
#            else:
#                if ret == 'abs':
#                    return(np.abs(np.fft.irfftn(M,s = s, axes = axes, norm = norm)));
#                elif ret == 'phase':
#                    return(np.angle(np.fft.irfftn(M,s = s, axes = axes, norm = norm)),False);
#                elif ret == 'real':
#                    return(np.real(np.fft.irfftn(M,s = s, axes = axes, norm = norm)));
#                elif ret == 'imag':
#                    return(np.imag(np.fft.irfftn(M,s = s, axes = axes, norm = norm)));
#                elif ret == 'complex':
#                    return(np.fft.irfftn(M,s = s, axes = axes, norm = norm));
#                else:
#                    print('Return value not known! Use "abs", "phase", "real", "imag" or "complex"');
#                    return(M);
#        else:
#            if shift:
#                if ret == 'abs':
#                    return(np.abs(np.fft.ifftshift(np.fft.ifftn(M,s = s, axes = axes, norm = norm),axes = axes)));
#                elif ret == 'phase':
#                    return(np.angle(np.fft.ifftshift(np.fft.ifftn(M,s = s, axes = axes, norm = norm),axes = axes),False));
#                elif ret == 'real':
#                    return(np.real(np.fft.ifftshift(np.fft.ifftn(M,s = s, axes = axes, norm = norm),axes = axes)));
#                elif ret == 'imag':
#                    return(np.imag(np.fft.ifftshift(np.fft.ifftn(M,s = s, axes = axes, norm = norm),axes = axes)));
#                elif ret == 'complex':
#                    return(np.fft.ifftshift(np.fft.ifftn(M,s = s, axes = axes, norm = norm),axes = axes));
#                else:
#                    print('Return value not known! Use "abs", "phase", "real", "imag" or "complex"');
#                    return(M);
#            else:
#                if ret == 'abs':
#                    return(np.abs(np.fft.ifftn(M,s = s, axes = axes, norm = norm)));
#                elif ret == 'phase':
#                    return(np.angle(np.fft.ifftn(M,s = s, axes = axes, norm = norm)),False);
#                elif ret == 'real':
#                    return(np.fft.fftshift(np.fft.ifftn(M,s = s, axes = axes, norm = norm)));
#                elif ret == 'imag':
#                    return(np.imag(np.fft.ifftn(M,s = s, axes = axes, norm = norm)));
#                elif ret == 'complex':
#                    return(np.fft.ifftn(M,s = s, axes = axes, norm = norm));
#                else:
#                    print('Return value not known! Use "abs", "phase", "real", "imag" or "complex"');
#                    return(M);
#    else:
#        if real:
#            if shift:
#                if ret == 'abs':
#                    return(np.abs(np.fft.ifftshift(np.fft.irfft(M,n = s, axis = axes, norm = norm),axes = axes)));
#                elif ret == 'phase':
#                    return(np.angle(np.fft.ifftshift(np.fft.irfft(M,n = s, axis = axes, norm = norm),axes = axes),False));
#                elif ret == 'real':
#                    return(np.real(np.fft.ifftshift(np.fft.irfft(M,n = s, axis = axes, norm = norm),axes = axes)));
#                elif ret == 'imag':
#                    return(np.imag(np.fft.ifftshift(np.fft.irfft(M,n = s, axis = axes, norm = norm),axes = axes)));
#                elif ret == 'complex':
#                    return(np.fft.ifftshift(np.fft.irfft(M,n = s, axis = axes, norm = norm),axes = axes));
#                else:
#                    print('Return value not known! Use "abs", "phase", "real", "imag" or "complex"');
#                    return(M);
#            else:
#                if ret == 'abs':
#                    return(np.abs(np.fft.irfft(M,n = s, axis = axes, norm = norm)));
#                elif ret == 'phase':
#                    return(np.angle(np.fft.irfft(M,n = s, axis = axes, norm = norm)),False);
#                elif ret == 'real':
#                    return(np.real(np.fft.irfft(M,n = s, axis = axes, norm = norm)));
#                elif ret == 'imag':
#                    return(np.imag(np.fft.irfft(M,n = s, axis = axes, norm = norm)));
#                elif ret == 'complex':
#                    return(np.fft.ifft(M,n = s, axis = axes, norm = norm));
#                else:
#                    print('Return value not known! Use "abs", "phase", "real", "imag" or "complex"');
#                    return(M);
#        else:
#            if shift:
#                if ret == 'abs':
#                    return(np.abs(np.fft.ifftshift(np.fft.ifft(M,n = s, axis = axes, norm = norm),axes = axes)));
#                elif ret == 'phase':
#                    return(np.angle(np.fft.ifftshift(np.fft.ifft(M,n = s, axis = axes, norm = norm),axes = axes),False));
#                elif ret == 'real':
#                    return(np.real(np.fft.ifftshift(np.fft.ifft(M,n = s, axis = axes, norm = norm),axes = axes)));
#                elif ret == 'imag':
#                    return(np.imag(np.fft.ifftshift(np.fft.ifft(M,n = s, axis = axes, norm = norm),axes = axes)));
#                elif ret == 'complex':
#                    return(np.fft.ifftshift(np.fft.ifft(M,n = s, axis = axes, norm = norm),axes = axes));
#                else:
#                    print('Return value not known! Use "abs", "phase", "real", "imag" or "complex"');
#                    return(M);
#            else:
#                if ret == 'abs':
#                    return(np.abs(np.fft.ifft(M,n = s, axis = axes, norm = norm)));
#                elif ret == 'phase':
#                    return(np.angle(np.fft.ifft(M,n = s, axis = axes, norm = norm)),False);
#                elif ret == 'real':
#                    return(np.fft.fftshift(np.fft.ifft(M,n = s, axis = axes, norm = norm)));
#                elif ret == 'imag':
#                    return(np.imag(np.fft.ifft(M,n = s, axis = axes, norm = norm)));
#                elif ret == 'complex':
#                    return(np.fft.ifft(M,n = s, axis = axes, norm = norm));
#                else:
#                    print('Return value not known! Use "abs", "phase", "real", "imag" or "complex"');
#                    return(M);
#    
#
#
#
#
