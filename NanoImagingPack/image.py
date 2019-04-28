#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:21:21 2017

@author: Christian Karras, Rainer Heintzmann

 All kind of image handling
 
 -> read
 -> write
 -> Dampedge
 -> correlations -> create correlations etc.
 -> algin (modes:2D, 3D)
 -> rotate
 -> noise

"""

import numbers
import tifffile as tif
import numbers
import numpy as np
from numpy.matlib import repmat
from .FileUtils import list_files, get_sorted_file_list
from os.path import join, isdir, splitext, isfile, split, isfile, join, splitext, split, basename
from os import mkdir, listdir
from scipy import misc
from scipy.ndimage import rotate
from scipy.ndimage.measurements import center_of_mass as cm

from .functions import gaussian, coshalf
from .config import DBG_MSG, __DEFAULTS__
from .noise import poisson
from . import coordinates
from .view import graph, view
from .view5d import v5, JNIUS_RUNNING # for debugging
from . import util
import warnings

from pkg_resources import resource_filename
from IPython.lib.pretty import pretty

defaultDataType=np.float32
defaultCpxDataType=np.complex64


def save_to_3D_tif(directory, file_prototype, save_name, sort='date', key=None):
    """
        load a stack of 2D images and save it as 3D Stack

        directory:   directory of the images
        file_prototype: string: all files where the name contains the strings be loaded
        save_name: save name
        sort: how to sort files ('name' or 'date' or 'integer_key')
            integer key: give key character, after which an integer number will be searched
    """

    flist = get_sorted_file_list(directory, file_prototype, sort, key)
    print(flist)
    img = np.asarray([readim(join(directory, file)) for file in flist])
    img = np.swapaxes(img, 0, 1)
    img = np.swapaxes(img, 1, 2)
    imsave(img, join(directory, save_name))


'''
    Todo: Stack images!!!
'''


def imsave(img, path, form='tif', rescale=True, BitDepth=16, Floating=False, truncate=True):
    """
        Save images

        path - image path
        form - format:
                    for tiff files, the fifffile package will be used (supports 3D Tiff),
                    otherwise PIL
        rescale:
                scale image up to maximum bit Depth
        BitDepth - give BitDepth -> generally ony 1, 8, 16, 32 and 64 makes sense due to int casting, if 'auto' -> no changes will be made
        Floating - Save as Floating point image (carefull, that might not always be possible and only rarely supported by common viewers)
        truncate - truncate values below zero
    """

    folder = split(path)[0]
    if not isdir(folder):
        mkdir(folder)
        print('Creating Folder ... ' + folder)

    ext = splitext(path)[-1][1:]
    if ext == '':
        path += '.' + form

    if util.get_type(img)[1] == 'image':
        metadata = {'name': img.name, 'pixelsize': str(img.pixelsize), 'units': img.unit, 'info': img.info}
    elif util.get_type(img)[1] == 'ndarray':
        metadata = {}
    else:
        raise TypeError('Wrong type. Can only save ndarrays or image type')
    if Floating == False:
        if truncate:
            img = img * (img > 0)

        if BitDepth != 'auto':
            if rescale:
                if BitDepth == 1:
                    img = img / np.max(img) * 255
                else:
                    img = img / np.max(img) * (2 ** BitDepth - 1)
            if np.max(img) >= 2 ** BitDepth:
                print('WARNING! Image maximum larger than ' + str(2 ** BitDepth - 1) + '! RESCALING!!!')
                img = img / np.max(img) * (2 ** BitDepth - 1)
            if np.min(img) >= 0:
                if BitDepth <= 8:
                    img = np.uint8(img)
                elif BitDepth <= 16:
                    img = np.uint16(img)
                elif BitDepth <= 32:
                    img = np.uint32(img)
                else:
                    img = np.uint64(img)
            else:
                print('ATTENTION: NEGATIVE VALUES IN IMAGE! Using int casting, not uint!')
                if BitDepth <= 8:
                    img = np.int8(img)
                elif BitDepth <= 16:
                    img = np.int16(img)
                elif BitDepth <= 32:
                    img = np.int32(img)
                else:
                    img = np.int64(img)

    if form in __DEFAULTS__['IMG_TIFF_FORMATS']:
        tif.imsave(path, img, metadata=metadata)  # RH 2.2.19 deleted: np.transpose
    else:
        import PIL
        # img = np;   # RH 2.2.19 deleted: np.transpose   CK commented line
        img = PIL.Image.fromarray(img)
        if BitDepth == 1:
            img = img.convert("1")
        else:
            img = img.convert("L")

        img.save(path, form)
    return img


# def readim(path =resource_filename("NanoImagingPack","resources/todesstern.tif"), which = None, pixelsize = None):
def readim(path=None, which=None, pixelsize=None):
    """
             reads an image

             if nothing given reads todesstern.tif
             path -> path of image
                         if nothing: optens death star image
                         'lena'  - image of lena
                         'erika'  - image of erika
                         'orka'   - iamge of an orka
                         'house'   - iamge of a house
                         'todesstern' - image of death star
                         'resolution_512' -Kai's resolution test image
                         'resolution_fine' -Kai's resolution test image (more pixels)
                         'MITO_SIM'   - SIM stack (3 phases, 3 directions) of BPAE mitochondria (l_ex = 561 nm, l_em approx 600 nm, px_size = 62.5 nm)
             which is  which images should be read IN CASE OF 3D TIFF ?  -> can be list or tuple or range
    """
    l = []
    if path is None:
        path = __DEFAULTS__['IMG_DEFAULT_IMG_NAME']
    for p in __DEFAULTS__['IMG_DEFAULT_IMG_FOLDERS']:
        l += (list_files(p, ''))
    default_names = [splitext(split(el)[1])[0] for el in l]
    if path in default_names:
        path = l[default_names.index(path)]

    if isfile(path):
        ext = splitext(path)[-1][1:]

        if ext.lower() in __DEFAULTS__['IMG_TIFF_FORMATS']:

            if which is None:
                img = (tif.imread(path))  # RH 2.2.19 deleted: np.transpose
            else:
                img = (tif.imread(path, key=which))  # RH 2.2.19 deleted: np.transpose
            img = img.view(image)
            # img.pixelsize = pixelsize;
            img.set_pixelsize(pixelsize)
        elif ext.lower() in __DEFAULTS__['IMG_ZEISS_FORMATS']:
            # TODO HERE: READ ONLY SELECTED SLICES OF THE IMAGE
            from .EXTERNAL.Gohlke_czi_reader import imread as cziread
            if __DEFAULTS__['IMG_SQUEEZE_ZEISS']:
                img, meta = cziread(path)
                img = img.squeeze().view(image)  # RH 2.2.19 deleted: np.transpose
            else:
                img, meta = cziread(path)
                img = img.view(image)  # RH 2.2.19 deleted: np.transpose
            img = img.view(image, pixelsize)
            img.metadata = meta
            # TODO: Pixelsizes aus Metadaten fischen
            img.set_pixelsize(pixelsize)
        else:
            try:
                import PIL.Image as IM
                img = (np.array(IM.open(path)))  # RH 2.2.19 deleted: np.transpose
                img = img.view(image)
                img.set_pixelsize(pixelsize)
            except OSError:
                raise ValueError('No valid image file')
        img.name = splitext(basename(path))[0]
        return img
    else:
        raise ValueError('No valid filename')


def readtimeseries(path, filename='', roi=[-1, -1, -1, -1], channel=0, ret_old_img_dim=False, axis=-3):
    """
        This reads a set of 2D Tiff files and creates one 3-D Stack from it.
        Path: The folder, containing the tiff-files
        Filename (optional): Only files containing this filename will be read, can also be a list of filenames which are inteted to be read
        channel: in case of multi channel image: which channel to be read!


         roi = [x,y, width_x, width_y] -> if nothting given than the whole image will be loaded
        ret_old_img_dim : return old image dimension (maximum dimension of all images read!)
        At which axis you want to concatenate the images (default is -3, that means z-direction)
    """
    if type(filename) != list:
        file_list = [split(f)[1] for f in list_files(path=path, file_prototype=filename, subfolders=False)]
        imtypes = __DEFAULTS__['IMG_TIFF_FORMATS'] + __DEFAULTS__['IMG_IMG_FORMATS']
        name_list = [f for f in file_list if splitext(f)[1][1:] in imtypes]
        # file_list = [f for f in listdir(path) if isfile(join(path, f))];
    # DEPRICATED:
    #        file_list.sort();
    #        name_list =[];
    #        for f in file_list:    #create name list
    #            body,ext = splitext(f);
    #            if filename == '':
    #                if (ext.lower =='.tif') or  (ext.lower() =='.tiff') or (ext.lower() == '.png') or (ext.lower() == '.bmp'):
    #                    name_list.append(f);    # Create list with names
    #            else:
    #                if ((ext =='.tif') or (ext =='.TIF') or (ext =='.tiff') or (ext =='.TIFF') or ext == '.png' or ext == '.bmp') and (body.find(filename)>=0):
    #                    name_list.append(f);    # Create list with names
    else:
        name_list = filename

    max_im_dim = 0
    print('Reading images ...')
    number = 0
    # dim = [0,0];
    # im_list =None;
    final_im = []
    for name in name_list:
        print(name, end=' ... ', flush=True)
        if roi == [-1, -1, -1, -1]:
            if splitext(name)[1][1:] == 'png' or splitext(name)[1][1:] == 'bmp':
                im = misc.imread(path + name)
            else:
                im = tif.imread(path + name)
        else:
            if splitext(name)[1][1:] == 'png' or splitext(name)[1][1:] == 'bmp':
                im = misc.imread(path + name)[roi[1]:roi[3] + roi[1], roi[0]:roi[2] + roi[0]]
            else:
                im = tif.imread(path + name)[roi[1]:roi[3] + roi[1], roi[0]:roi[2] + roi[0]]
        # im = im.transpose();     # CK 20.02.19 transpose outcommented
        print(' Shape: ' + str(im.shape), end=' ; ', flush=True)
        if im.ndim > max_im_dim:
            max_im_dim = im.ndim
        if np.ndim(im) != 2:
            print('Reading channel ' + str(channel))
            im = im[:, :, channel]
        if number == 0:
            final_im = im
        else:
            final_im = cat((final_im, im), axis)
        number += 1

    '''
    DEPRECIATET CODE:                
                dim = [np.size(im, axis =0), np.size(im, axis =1)];
                number +=1;
                if im_list is None:
                    im_list = np.expand_dims(im,2);
                else:
                    im_list = np.concatenate((im_list,np.expand_dims(im,2)),axis =2)
            else:
                if (dim == [np.size(im, axis =0), np.size(im, axis =1)]):
                    number +=1;
                    if im_list is None:
                        im_list = np.expand_dims(im,2);
                    else:
                        im_list = np.concatenate((im_list,np.expand_dims(im,2)),axis =2)
                else:
                    print('Wrong size of image '+ name);
     '''
    print()
    print(str(number) + ' images read!')

    if ret_old_img_dim:
        return image(final_im, name=splitext(basename(path))[0]), max_im_dim
    else:
        return image(final_im, name=splitext(basename(path))[0])


def gaussf(img, kernelSigma):
    """
    performs an N-dimensional Gaussian convolution, based on Fourier-transforms.

    :param img: input image to convolve
    :param kernelSigma: sizes of the kernel stated as StdDev. If smaller thant the dimensions of the image, the other kernelsizes are assumed to be zero
    :return: the convolved image
    """
    kernelSigma = util.repToList(kernelSigma, img.ndim)
    kernel = gaussian(img.shape, kernelSigma)
    return convolve(img, kernel, norm2nd=True)


def findBg(img, kernelSigma=3.0):
    """
    estimates the background value by convolving first with a Gaussian and then looking for the minimum.

    :param img: input image to find the background for
    :param kernelSigma: size of the kernel
    :return:
    """
    return np.min(gaussf(img, kernelSigma))


def DampEdge(img, width=None, rwidth=0.1, axes=None, func=coshalf, method="damp", sigma=4.0):
    """
        DampEdge function

        im  image to damp edges

        rwidth : relative width (default : 0.1 meaning 10%)
            width in relation to the image size along this dimenions. Can be a single number or a tuple

        width (None: rwidht is used, else width takes precedence)
            -> characteristic absolute width for damping
            -> can be integer, than every (given) axis is damped by the same size
            -> can be list or tupel -> than individual damping for given axis

        axes-> which axes to be damped (default is None, meaning all axes)

        func   - which function shall be used for damping -> some are stated in functions.py, first element should be x, second one the length (Damping length!)
                e.g. cossqr, coshalf, linear
                default: coshalf

        method -> which method should be used?
                -> "zero" : dims down to zero
                -> "damp" : blurs to an averaged mean (default)
                -> "moisan" : HF/LF split method according to Moisan, J Math Imaging Vis (2011) 39: 161–179, DOI 10.1007/s10851-010-0227-1

        return image with damped edges

        Example:
            import NanoImagingPack as nip
            nip.DampEdge(nip.readim()[400:,200:])
        TODO in FUTURE: padding of the image before damping
    """
    img = img.astype(defaultDataType)
    res = np.ones(img.shape)
    if width == None:
        width = tuple(np.round(np.array(img.shape) * np.array(rwidth)).astype("int"))

    if axes == None:
        axes = np.arange(0, img.ndim).tolist()
    if type(width) == int:
        width = [width]
    if type(width) == tuple:
        width = list(width)
    if len(width) < len(axes):
        ext = np.ones(len(axes) - len(width)) * width[-1]
        width.extend(list(ext.astype(int)))

    res = img
    mysum = util.zeros(img.shape)
    sz = img.shape
    den = -2 * len(set(axes))  # use only the counting dimensions
    for i in range(len(img.shape)):

        if i in axes:
            line = np.arange(0, img.shape[i], 1)
            myramp = util.make_damp_ramp(width[i], func)
            if method == "zero":
                line = cat((myramp[::-1], np.ones(img.shape[i] - 2 * width[i]), myramp), -1)
                goal = 0.0  # dim down to zero
            elif method == "moisan":
                top = util.subslice(img, i, 0)
                bottom = util.subslice(img, i, -1)
                mysum = util.subsliceAsg(mysum, i, 0, bottom - top + util.subslice(mysum, i, 0))
                mysum = util.subsliceAsg(mysum, i, -1, top - bottom + util.subslice(mysum, i, -1))
                den = den + 2 * np.cos(2 * np.pi * coordinates.ramp(util.dimVec(i, sz[i], len(sz)), i, freq='ftfreq'))
            elif method == "damp":
                line = cat((myramp[::-1], np.ones(img.shape[i] - 2 * width[i] + 1), myramp[:-1]), 0)  # to make it perfectly cyclic
                top = util.subslice(img, i, 0)
                bottom = util.subslice(img, i, -1)
                goal = (top + bottom) / 2.0
                goal = gaussf(goal, sigma)
            else:
                raise ValueError("DampEdge: Unknown method. Choose: damp, moisan or zero.")
            # res = res.swapaxes(0,i); # The broadcasting works only for Python versions >3.5
            #            res = res.swapaxes(len(img.shape)-1,i); # The broadcasting works only for Python versions >3.5
            if method != "moisan":
                line = util.castdim(line, img.ndim, i)  # The broadcasting works only for Python versions >3.5
                try:
                    res = res * line + (1.0 - line) * goal
                except ValueError:
                    print('Broadcasting failed! Maybe the Python version is too old ... - Now we have to use repmat and reshape :(')
                    res *= np.reshape(repmat(line, 1, np.prod(res.shape[1:])), res.shape, order='F')
    if method == "moisan":
        den = util.midValAsg(image(den), 1)  # to avoid the division by zero error
        den = ft(mysum) / den
        den = util.midValAsg(den, 0)  # kill the zero frequency
        den = np.real(ift(den))
        res = img - den

    # return(res)
    return res  # CK: (__cast__(img*res.view(image),img));  What is this?? It should return res and not img*res


def DampOutside(img, width=None, rwidth=0.1, usepixels=3, mykernel=None, kernelpower=3):
    """
    DampOutside function


    Extrapolates the data by filling in blurred information outside the edges. This is a bit like DampEdge
    but using a normalized convolution with a kernel.

    Parameters
    ----------
        img         image to pad damp edges

        width       (None): rwidth is used, else width takes precedence
                     -> characteristic absolute width for damping
                     -> can be integer, than every (given) axis is damped by the same size
                     -> can be list or tupel or list of tuples -> than individual damping for given axis

        rwidth      relative width (default : 0.1 meaning 10%)
                    width in relation to the image size along its dimenions. Can be a single number or a tuple or list of tuples

        usepixels   pixels to be used for convolution, should be a single number

        mykernel    (None): kernel to be used for convolution, default is r^-3
                    must be an image of the same size as the padded image

        kernelpower factor to be used in the default kernel, default is 3

    Returns
    -------
        img : image with padded and damped edges

    Examples
    --------
        import NanoImagingPack as nip
        img = nip.readim()

        DampOutside(img)
        DampOutside(img, [4,20])
        DampOutside(img, rwidth = [[0,0.3],[0.5,0.2]])
    """
    from .coordinates import rr
    from .transformations import ft, ift

    width = np.array(width)
    rwidth = np.array(rwidth)
    mykernel = np.array(mykernel)

    if width.any() == None:
        try:
            if img.ndim == 1:
                if rwidth.ndim == 0:
                    rwidth = np.array([[rwidth, rwidth]])
                elif len(rwidth) == 2:
                    rwidth = np.array([rwidth])
                elif rwidth.ndim != 2:
                    rwidth = np.array([[rwidth[0], rwidth[0]]])
            if rwidth.ndim <= 1:
                width = np.ceil(img.shape * rwidth / 2).astype(int)
                width = np.array([width, width]).T
            elif rwidth.ndim == 2:
                width = np.ceil([rwidth[i] * img.shape[i] for i in range(len(rwidth))]).astype(int)
        except:
            raise ValueError('rwidth did not match image dimension')
    else:
        if width.ndim == 0:
            width = np.array([width for i in range(len(img.shape))])
            width = np.array([width, width]).T
        if width.ndim == 1:
            if img.ndim > 1:
                if len(width) != len(img.shape):
                    raise ValueError('width did not match image dimension')
                width = np.array([width, width]).T
            else:
                if len(width) == 2:
                    width = np.array([width])
                elif len(width) == 1:
                    width = np.array([[width[0], width[0]]])
                else:
                    raise ValueError('width did not match image dimension')

    size = np.array(img.shape)
    newsize = size + np.sum(width, axis=1)

    ### build the kernel
    if mykernel.any() != None:
        if not np.array_equal(mykernel.shape, newsize):
            raise ValueError('mykernel did not match new image size')

    if mykernel.any() == None:
        r = rr(newsize)
        r[r == 0] = np.inf
        mykernel = (1 / r) - 1 / np.linalg.norm(np.sum(width, axis=1) * np.sqrt(2))
        mykernel[mykernel < 0] = 0
        mykernel = mykernel ** kernelpower

    transfer = ft(mykernel)

    wimg = np.zeros(size) + 1
    mask = np.zeros(newsize) + 1
    nimg = img.__copy__()

    grid = np.meshgrid(*[np.arange(usepixels, s - usepixels) for s in size], indexing='ij')

    wimg[tuple(grid)] = 0
    nimg[tuple(grid)] = 0

    grid = np.meshgrid(*[np.arange(width[i, 0], newsize[i] - width[i, 1]) for i in range(len(newsize))], indexing='ij')
    mask[tuple(grid)] = 0

    nimg = np.pad(nimg, width, 'constant')
    wimg = np.pad(wimg, width, 'constant')

    nimg2 = np.real(ift(transfer * ft(nimg)))
    wimg2 = np.real(ift(transfer * ft(wimg)))

    flag = mask != 0

    nimg[flag] = nimg2[flag] / wimg2[flag]
    nimg[tuple(grid)] = img

    return util.__cast__(nimg, img)


def __check_complex__(im):
    """
        checks the type an image:
            returns True if the image is complex
            otherwise returns False:
    """
    return not (np.issubdtype(im.dtype, np.floating) or np.issubdtype(im.dtype, np.integer))


def make_odd(M, ax):
    """
    Make a image odd in the given axis by removing one pixel line
    """
    if np.mod(np.size(M, axis=ax), 2) == 0:
        M = np.delete(M, 0, ax)
    return util.__cast__(M, M)


def match_size(im1, im2, axes=None, padmode='constant', clip_offset=None, odd=False):
    """
    Adjust the sizes of 2 images
    Both images must have the same dimensions.

    :param im1:          Image1
    :param im2:          Image2
    :param ax:          axes along which the images should be adjusted
    :param padmode:     clip: clips the larger image to the size of the smaller one, whereas the first part is used!
                        constant: fill up with zeros symmetrical (e.g. same size above and below)
                        const_below: fill up below
    :param clip_offset: for clipping: what is the offset of the at which axis for the clipping?
    :param odd:         Make the images odd (Potentially to be depricated in newer versions)
    :return:            A tuple of the adjusted images

    ----------------------------------------------------------------------------------------
    Example:

    import NanoImagingPack as nip;
    im1 = nip.readim();         # 800X800 image
    im2 = nip.readim('erika');  # 256X256 image

    new_im1, new_im2 = nip.match_size(im1, im2, 0, 'constant')
    """
    if np.ndim(im1) == np.ndim(im2):
        if isinstance(axes, numbers.Integral):
            axes = [axes]
        elif isinstance(axes, list) or isinstance(axes, tuple):
            pass;
        else:
            raise ValueError("Wrong data type for axes: integer, list or tuple")
        if clip_offset is None:
            clip_offset = [0 for i in axes]
        elif isinstance(clip_offset, numbers.Integral):
            clip_offset = [clip_offset for i in axes]
        elif isinstance(clip_offset, list) or isinstance(clip_offset, tuple):
            clip_offset = list(clip_offset)
            if len(clip_offset) < len(axes):
                util.adjust_lists(clip_offset, axes, 0)
        else:
            raise ValueError("Wrong data type for clip_offset: None,integer, list or tuple")
        for ax, offset in zip(axes, clip_offset):
            if odd:
                im1 = make_odd(im1, ax)
                im2 = make_odd(im2, ax)
            diff = np.size(im1, axis=ax) - np.size(im2, axis=ax)  # Get difference in size
            if padmode == 'clip':
                im1 = np.swapaxes(im1, 0, ax)
                im2 = np.swapaxes(im2, 0, ax)
                if np.size(im1, axis=0) > np.size(im2, axis=0):
                    im1 = im1[offset:np.size(im2, axis=0) + offset]
                elif np.size(im1, axis=0) < np.size(im2, axis=0):
                    im2 = im2[offset:np.size(im1, axis=0) + offset]
                im1 = np.swapaxes(im1, 0, ax)
                im2 = np.swapaxes(im2, 0, ax)
            elif padmode == 'constant':
                padding = []
                for i in range(np.ndim(im1)):
                    if i == ax:
                        padding.append((np.abs(diff // 2), np.abs(diff // 2 + np.mod(diff, 2))))
                    else:
                        padding.append((0, 0))
                if diff < 0:
                    im1 = np.lib.pad(im1, tuple(padding), padmode)
                else:
                    im2 = np.lib.pad(im2, tuple(padding), padmode)
            elif padmode == 'const_below':
                padding = []
                for i in range(np.ndim(im1)):
                    if i == ax:
                        padding.append((0, np.abs(diff)))
                    else:
                        padding.append((0, 0))
                if diff < 0:
                    im1 = np.lib.pad(im1, tuple(padding), 'constant')
                else:
                    im2 = np.lib.pad(im2, tuple(padding), 'constant')

    else:
        raise ValueError('Cannot match sizes as arrays have different dimensions!')

    return util.__cast__(im1, im1), util.__cast__(im2, im2)


# TODO:
# def optimize_shift(im1,im2, method = 'optimize_max'):
#    '''
#        takes two images and corrects the shift between them:
#
#            im1: image1
#            im2: image2
#            method: which method to use?
#                        'optimize_max':    optimized maximum of correlation
#
#
#    '''
#    if im1.shape != im2.shape:
#        print('Error: images have different shapes!');
#    else:
#        if im1.ndim != 2:
#            print('Error: optimizer currently only works for 2D images');
#        else:
#            corr = correl(im1, im2)[0];
#            max_corr = np.max(corr);
#            argsmax_corr = max_coord(np.argmax(corr));
#            im2 = shift(im2, im2.shape//2)


def FRC(im1, im2, pixel_size=62, num_rings=10, correct_shift=True):
    """
        Compute the Fourier ring correlation (frc) between 2 2D-images

        im1, im2: the images
        pixel_size: pixelsize; -> can be tuple or lis or number
        num_rings: number of ringst to compute the FRC
        correct_shift: corrects a possible shift between the images -> until now: no sub-pixel-shift!

        Notice:
            Although the function does adjust the images sizes if the images are not square shaped and have the same dimensions it is recommened to only use square images of the same sizes, since the code is not sufficiently testet!

        Returns a tupel:  (FRC strength, spatial frequency [1/unit(pixel_size)])
    """
    if (im1.ndim == 2) and (im2.ndim == 2):
        if im1.shape[0] > im1.shape[1]:
            print('Adjusting shape of image 1 in y -direction')
            im1 = np.lib.pad(im1, ((0, 0), (0, im1.shape[0] - im1.shape[1])), 'constant')
        if im1.shape[1] > im1.shape[0]:
            print('Adjusting shape of image 1 in x -direction')
            im1 = np.lib.pad(im1, ((0, im1.shape[0] - im1.shape[1]), (0, 0)), 'constant')

        im1, im2 = match_size(im1, im2, padmode='const_below', axes=0, odd=False)
        im1, im2 = match_size(im1, im2, padmode='const_below', axes=1, odd=False)
        '''
        if im1.shape[0] > im2.shape[0]:
            print('Matching sizes of image 1 and image 2 in x direction')
            im2 = np.lib.pad(im2, (im1.shape[0]-im2.shape[0],0), 'constant')
        if im1.shape[1] > im2.shape[1]:
            print('Matching sizes of image 1 and image 2 in y direction')
            im2 = np.lib.pad(im2, (0,im1.shape[1]-im2.shape[1]), 'constant')

        if im1.shape[0] < im2.shape[0]:
            print('Matching sizes of image 1 and image 2 in x direction')
            im2 = im2[:im1.shape[0],:]
        if im1.shape[1] < im2.shape[1]:
            print('Matching sizes of image 1 and image 2 in y direction')
            im2 = im2[:,:im1.shape[0]]
        '''

        if correct_shift == True:
            corr = correl(im1, im2, matchsizes=True) # RH: deleted np.abs. Why??? Anticorrel is not correl
            argmax_corr = util.max_coord(corr)
            print(argmax_corr)
            im2 = shift_center(im2, -im2.shape[0] // 2 + argmax_corr[0], -im2.shape[1] // 2 + argmax_corr[1])
            print(util.max_coord(correl(im1, im2, matchsizes=True))) # Rh see above

        im1 = ft(im1, shift_after=True, shift_before=False, ret='complex', axes=(0, 1))
        im2 = ft(im2, shift_after=True, shift_before=False, ret='complex', axes=(0, 1))

        if isinstance(im1, image):
            pxs = image.pixelsize
        else:
            import numbers
            if isinstance(pixel_size, numbers.Number):
                pxs = [pixel_size, pixel_size]
            elif (type(pixel_size) == tuple or type(pixel_size) == list) and len(pixel_size) > 1:
                pxs = pixel_size[:2]
            else:
                raise ValueError('Pixelsize must be list, tuple or number')
        f_step = [1 / (p * s) for (p, s) in zip(pxs, im1.shape)]
        max_f = [1 / (2 * p) - 1 / (p * s) for (p, s) in zip(pxs, im1.shape)]
        k_max = min(max_f)
        max_pixel = [k_max / f for f in f_step]
        rad_x = np.linspace(0, max_pixel[0], num_rings + 1)
        rad_y = np.linspace(0, max_pixel[1], num_rings + 1)

        try:
            vol_im1 = np.reshape(repmat(im1, num_rings, 1), (num_rings, im1.shape[0], im1.shape[1]))
            vol_im1 = vol_im1.swapaxes(0, 1)
            vol_im1 = vol_im1.swapaxes(1, 2)
            vol_im2 = np.reshape(repmat(im2, num_rings, 1), (num_rings, im2.shape[0], im2.shape[1]))
            vol_im2 = vol_im2.swapaxes(0, 1)
            vol_im2 = vol_im2.swapaxes(1, 2)
            mask = ((coordinates.xx((im1.shape[0], im1.shape[1], num_rings))) ** 2 / rad_x[1:] ** 2 + (coordinates.yy((im1.shape[0], im1.shape[1], num_rings))) ** 2 / rad_y[1:] ** 2 < 1) * 1 - (
                    (coordinates.xx((im1.shape[0], im1.shape[1], num_rings))) ** 2 / rad_x[:rad_x.size - 1] ** 2 + (coordinates.yy((im1.shape[0], im1.shape[1], num_rings))) ** 2 / rad_y[
                                                                                                                                                                :rad_x.size - 1] ** 2 < 1) * 1

            fcr = np.sum(vol_im1 * np.conjugate(vol_im2) * mask, axis=(0, 1)) / (
                np.sqrt(np.sum(np.abs(vol_im1) ** 2 * mask, axis=(0, 1)) * np.sum(np.abs(vol_im2) ** 2 * mask, axis=(0, 1))))
        except MemoryError:
            print('Memory problem! Analyzing rings sequentially!')
            for i in range(num_rings):
                print('Ringnumber ' + str(i))
                mask = ((coordinates.xx((im1.shape[0], im1.shape[1])) ** 2 / rad_x[i + 1] ** 2 + coordinates.yy((im1.shape[0], im1.shape[1])) ** 2 / rad_x[i + 1] ** 2) < 1) * 1 - (
                        (coordinates.xx((im1.shape[0], im1.shape[1])) ** 2 / rad_x[i] ** 2 + coordinates.yy((im1.shape[0], im1.shape[1])) ** 2 / rad_y[i] ** 2) < 1) * 1
                el = np.sum(im1 * np.conjugate(im2) * mask) / (np.sqrt(np.sum(np.abs(im1) ** 2 * mask) * np.sum(np.abs(im2) ** 2 * mask)))
                if i == 0:
                    fcr = el
                else:
                    fcr = cat((fcr, el), 0)
        return np.abs(fcr), np.linspace(0, k_max, num_rings)
    else:
        raise TypeError('Wrong image dimension! Only 2 Dimensional images allowed')
        return


# def supersample(M, direction)

def threshold(im, t1, t2=None):
    """
        Threshold image
        returns a binary image -> if pixelvalue >= t1 -> 1, else: 0

        if t2 is given than it returns 1 for pixelvalues between t1 and t2
    """
    if t2 is None:
        return (im >= t1) * 1
    else:
        if t1 > t2:
            h = t2
            t2 = t1
            t1 = h
        return util.__cast__((im >= t1) * (im <= t2), im)


def get_max(M, region=[-1, -1, -1, -1]):
    """
    Get maximum value in a certain region

    region = [x,y, width_x, width_y]

    !!!only works for 2D right now!!!
    """
    if region[0] != -1:
        M = M[region[1] - region[3] // 2:region[1] + region[3] // 2, region[0] - region[2] // 2:region[0] + region[2] // 2]
    MAX = util.max_coord(M)
    return region[0] - region[2] // 2 + MAX[1], region[1] - region[3] // 2 + MAX[0]


def adjust_dims(imlist, maxdim=None):
    """
        This functions takes a tupel of a list of images and adds dimensions in a way that all images in the list (or the tupel) have the same number of dimensions afterwards. T
        Maxdim defines the dimension number of the final images.
        If maxdim is smaller than the dimension size in one image or not given, maxdim will be ignored and the dimension number of that image with the most dimensions will be used instead

        Extra dimensions will be added at the end
    """

    def __exp_dims__(im):
        for i in range(im.ndim, maxdim):
            im = np.expand_dims(im, 0)  # i RH 2.2.19
        return im

    err = False
    if type(imlist) == list or type(imlist) == tuple:
        dimsize_list = []
        for im in imlist:
            if isinstance(im, np.ndarray):
                dimsize_list.append(im.ndim)
            else:
                err = True
        if maxdim is None or maxdim < max(dimsize_list):
            # if maxdim != None: print('Given maximum dimension too small -> adjusting it to ' + str(maxdim));
            maxdim = max(dimsize_list)
        if err == False:
            imlist = [__exp_dims__(im) for im in imlist]
        else:
            raise TypeError('Wrong data input -> give list')
    else:
        raise TypeError('Wrong data input')
    return imlist


def toClipboard(im, separator='\t', decimal_delimiter='.', transpose=False):
    import win32clipboard as clipboard
    '''
        Save image to clipboard
        only works with 1D or 2D images
    '''

    # save to clipboard

    # TODO: nD darstellung
    # Put string into clipboard (open, clear, set, close)
    if transpose:
        im = im.transpose
    s = np.array2string(im)
    s = s.replace(']\n ', '\n')
    s = s.replace('\n ', '')
    s = s.replace('[', '')
    s = s.replace(']', '')
    s = s.replace('.', decimal_delimiter)
    pos = 0
    while pos >= 0:
        pos = s.find(' ')
        if pos != len(s) - 1:
            if s[pos + 1] == ' ':
                s = s[:pos] + s[1 + pos:]
            else:
                s = s[:pos] + separator + s[1 + pos:]
        else:
            s = s[:pos]

    #    for i in im:
    #        s+= str(i)+separator;
    clipboard.OpenClipboard()
    clipboard.EmptyClipboard()
    clipboard.SetClipboardText(s)
    clipboard.CloseClipboard()


def catE(*argv, matchsizes = False):
    """
        A shorthand for concatenating along the 4th (element dimension).
        Calls cat(imlist,-4)
    """
    if len(argv)==1 and (type(argv[0]) is list or type(argv[0]) is tuple):
        argv=argv[0]
    if len(argv) == 1:
        res = argv[0]
    else:
        res = cat(argv, -4, matchsizes=matchsizes)
    res.dim_description = util.caller_args("catE(")
    return res


def cat(imlist, axis=None, destdims=None, matchsizes = False):
    """
        This function takes a list or a tuple of images and stacks them at the given axis.
        If the images have different dimensions the dimensions will be adjusted (using adjust_dims)

        If the axis is larger than the image dimensions the image dimensions will be expanded

        If the images do have different sizes, the sizes will also be adjusted by using the match_size function

        Be aware that axis count is according to the numpy convention, e.g. Z,Y,X meaning ax=0 for Z!
        default: axis=-(numdims+1), meaning the unused dimension (before the first). Positive values will adress the axis in the numpy convention and throw an error if over the limit. Negative values will automatically expand.
        destdims (default: None): The number of destination dimensions to expand to
    """
    imlist = list(imlist)
    imlist = [x for x in imlist if x is not None]
    for i in range(len(imlist)):
        if not isinstance(imlist[i], image):
            imlist[i] = image(imlist[i])
    pixelsize = util.longestPixelsize(imlist)

    imlist = tuple(imlist)
    shapes = np.asarray([list(im.shape) for im in imlist])

    if axis is None:
        axis = -len(shapes[0])-1 # -shapes.shape[1] - 1

    if destdims == None:
        if axis < 0:
            maxdims = -axis
        #        imlist = adjust_dims(imlist, maxdims);
        else:
            maxdims = axis + 1
    else:
        if axis + 1 > destdims:
            print('WARNING: Axis larger than destdims -> adjusting destims to ' + str(axis + 1))
            destdims = axis + 1
        maxdims = destdims

    imlist = adjust_dims(imlist, maxdims)
    shapes = np.asarray([list(im.shape) for im in imlist])

    if axis < 0:
        ax = shapes.shape[1] + axis
    else:
        ax = axis
    for i in range(shapes.shape[1]):
        if (np.max(shapes[:, i]) != np.min(shapes[:, i])) and (i != ax):
            if matchsizes:
                imlist = [match_size(im, imlist[np.argmax(shapes[:, i])], i, padmode='constant', odd=False)[0] for im in imlist]
            else:
                raise ValueError("cat, dimension "+str(i)+": Sizes are not matching. Adjust sizes or use the flag matchsizes=True.")
            # return(np.concatenate((imlist),axis).squeeze());
    return image(np.concatenate(imlist, axis), pixelsize = pixelsize)


def histogram(im, name='', bins=65535, range=None, normed=False, weights=None, density=None):
    h = np.histogram(im, bins=bins)
    graph(y=h[0], x=h[1][:len(h[0])], title='Histogram of image ' + name, x_label='Bins', y_label='Counts', legend=[])
    return h


def shift(im, delta, axes=0, pixelwise=False):
    """
        Shifts an image im for a distance of delta pixels in the given direction using the FT shift theorem
        shift is done wia phaseramp (rft for real input), except of if pixelwise = True, than its shifted in real space for the given amount of pixels

        :param im:              Image to be shifted
        :param delta:           Shift  (float, int -> same shift for all direction or list of float and ints -> shift vectors)
        :param axes:            axes for the shift
        :param pixelwise:
        :return:

        ---------------------------------------------------------------------------------------------------
        Examples:

            import NanoImagingPack as nip;
            im = nip.readim('MITO_SIM');
            im1 = nip.shift(im, [3.5,32.4,-56.2]);                  # shift the matrix im for 3.5 pixels in direction 0 (z), 32.4 pixels in direction 1(y) and -56.2 pixels in direction 2 (x)
            im2 = nip.shift(im, 30, 2, pixelwise = True);      # shifts the matrix imfor 30 pixels in direction 2 (mostly z) for full pixel width (not using fts)

    """
    M = im
    old_arr = M
    direction = axes
    # unifiying input
    if type(delta) == tuple:
        delta = list(delta)
    if type(delta) == list:
        axes = list(range(len(delta)))
    elif isinstance(delta, numbers.Real):
        axes = [direction]
        delta = [delta]

    t = [(0, 0) for i in range(M.ndim)]
    i = 0;
    for d, ax in zip(delta, axes):
        if abs(d) > im.shape[ax]:
            warnings.warn('Shifting out of dimension for axis '+str(ax));
            delta[i] = im.shape[ax];
        if d > 0:
            t[ax] = (int(np.ceil(np.abs(d))), 0)
        else:
            t[ax] = (0, int(np.ceil(np.abs(d))))
        i+=1;
    if pixelwise:
        offset = []
        for d, ax in zip(delta, axes):
            if d > 0:
                offset += [0]
            else:
                offset += [-d]
        M = np.lib.pad(M, tuple(t), 'constant')
        M = match_size(M, old_arr, axes=axes, padmode='clip', clip_offset=offset)[0]
    else:
        # padding image with zeros

        #old_shape = M.shape
        # for d,ax in zip(delta, axes):
        #     t[ax] = (int(np.ceil(np.abs(d)/2)),int(np.ceil(np.abs(d)/2)));

        #M = np.lib.pad(M, tuple(t), 'constant')  # Change Boundaries to avoid ringing

        # FT image
        if M.dtype == np.complexfloating:
            FT = ft(M, shift_after=True, shift_before=False, axes=axes, s=None, norm=None, ret='complex')
            real_ax = -1
        else:
            FT = rft(M, shift_after=True, shift_before=False, axes=axes, s=None, norm=None, ret='complex')
            real_ax = max(axes)
        # Make and apply phase ramp
        phaseramp = np.zeros(FT.shape)
        for d, ax in zip(delta, axes):
            if ax == real_ax:
                phaseramp += coordinates.ramp(FT.shape, ramp_dim=ax, placement='positive') * 2 * np.pi * d / (M.shape[ax])
            else:
                phaseramp += coordinates.ramp(FT.shape, ramp_dim=ax, placement='center') * 2 * np.pi * d / (M.shape[ax])
        phaseramp = np.exp(-1j * phaseramp)
        if M.dtype == np.complexfloating:
            M = ift(FT * phaseramp, shift_after=False, shift_before=True, axes=axes, s=None, norm=None, ret='complex')
        else:
            M = irft(FT * phaseramp, s=M.shape, shift_after=False, shift_before=True, axes=axes, norm=None, ret='real')

        # clipping rims
        for d, ax in zip(delta, axes):
            M = M.swapaxes(0, ax)
            if d<0:
                M[int(np.floor(d)):] = 0;
            else:
                M[:int(np.floor(d))] =0;
        #     M = M[int(np.ceil(np.abs(d))):old_shape[ax] + int(np.ceil(np.abs(d)))]
            M = M.swapaxes(0, ax)
    return util.__cast__(M, old_arr)


def shiftx(M, delta):
    """
        shift image M for delta pixels in x-direction
    """
    return shift(M, delta, 0)


def shifty(M, delta):
    """
        shift image M for delta pixels in y-direction
    """

    return shift(M, delta, 1)


def shiftz(M, delta):
    """
        shift image M for delta pixels in z-direction
    """
    return shift(M, delta, 2)


def shift_center(M, x, y):
    """
        Shifts the center of an image by full pixels

        !!!only works for 2D right now!!!

        use the "shift" method for subpixel shifts
    """
    non = lambda s: s if s < 0 else None
    mom = lambda s: max(0, s)
    New = np.zeros_like(M)
    New[mom(x):non(x), mom(y):non(y)] = M[mom(-x):non(-x), mom(-y):non(-y)]
    return util.__cast__(New, M)


def __correllator__(im1, im2, axes=None, mode='convolution', phase_only=False, norm2nd=False, matchSizes=False):
    """
        Correlator for images

         If the images have different sizes, zeros will be padded at the rims
         im1,im2: Images
         axes:   along which axes
         mode= 'convolution' or 'correlation'
         phase only: Phase correlation

         If inputs are real than it tries to use the rfft

    """
    mindim = np.minimum(np.ndim(im1),np.ndim(im2))
#    if np.ndim(im1) == np.ndim(im2):
    old_arr = im1
    if matchSizes:
        for axis in range(np.ndim(im1)):
            if (im1.shape[axis] != im2.shape[axis]):  # RH. I changed this. Broadcasting is a useful feature and error checking too!
                im1, im2 = match_size(im1, im2, axis, padmode='constant', odd=False)
                print('Matching sizes at axis ' + str(axis))

    # create axes list
    if axes is None:
        axes = list(range(-mindim,0)) # list(range(len(im1.shape)))
    if type(axes) == int:
        axes = [axes]
    try:
        if np.issubdtype(axes.dtype, np.integer):
            axes = [axes]
    except AttributeError:
        pass;

    if im1.dtype == np.complexfloating or im2.dtype == np.complexfloating:
        FT1 = ft(im1, shift_after=False, shift_before=False, norm=None, ret='complex', axes=axes)
        FT2 = ft(im2, shift_after=False, shift_before=True, norm=None, ret='complex', axes=axes)
    else:

        FT1 = rft(im1, shift_after=False, shift_before=False, norm=None, ret='complex', axes=axes)
        FT2 = rft(im2, shift_after=False, shift_before=True, norm=None, ret='complex', axes=axes)

    if norm2nd == True:
        FT2 = FT2 / np.abs(FT2.flat[0])

    if mode == 'convolution':
        if phase_only:
            cor = np.exp(1j * (np.angle(FT1) + np.angle(FT2)))
        else:
            cor = FT1 * FT2
    elif mode == 'correlation':
        if phase_only:
            cor = np.exp(1j * (np.angle(FT1) - np.angle(FT2)))
        else:
            cor = FT1 * FT2.conjugate()
    else:
        raise ValueError('Wrong mode')
        return im1
    if im1.dtype == np.complexfloating or im2.dtype == np.complexfloating:
        return util.__cast__(ift(cor, shift_after=False, shift_before=False, norm=None, ret='complex', axes=axes), old_arr)
    else:
        if __DEFAULTS__['CC_ABS_RETURN']:
            return util.__cast__(irft(cor, s=old_arr.shape, shift_after=False, shift_before=False, norm=None, ret='abs', axes=axes), old_arr)
        else:
            return util.__cast__(irft(cor, s=old_arr.shape, shift_after=False, shift_before=False, norm=None, axes=axes), old_arr)
# else:
#     raise ValueError('Images have different dimensions')
#     return im1


def correl(im1, im2, axes=None, phase_only=False, matchSizes=False):
    """
        Correlator for images

         If the images have different sizes, zeros will be padded at the rims
         im1,im2: Images
         full_fft: apply full fft? or is real enough
         axes:   along which axes
         phase only: Phase correlation
         matchSizes: if True, the sizes will be atomatically expanded to match the larger one
    """
    return __correllator__(im1, im2, axes=axes, mode='correlation', phase_only=phase_only, matchSizes=matchSizes)


def convolve(im1, im2, full_fft=False, axes=None, phase_only=False, norm2nd=False, matchSizes=False):
    """
        Convolve two images for images

         If the images have different sizes, zeros will be padded at the rims
         im1,im2: Images
         full_fft: apply full fft? or is real enough
         axes:   along which axes
         phase only: Phase correlation (default: False)
         norm2nd : normalizes the second argument to one (keeps the mean of the first argument after convolution), (default: False)
         matchSizes: if True, the sizes will be atomatically expanded to match the larger one
    """
    return __correllator__(im1, im2, axes=axes, mode='convolution', phase_only=phase_only, norm2nd=norm2nd, matchSizes=matchSizes)



def shear(img, shearamount=10, shearDir=0, shearOrtho=None, center='center', padding=True):
    """
    shear function

    Shears an image im for a distance of shearamount pixels in the given direction using the FT shift theorem.

    Parameters
    ----------
        img         image to be sheared

        shearamount maximum shearamout, float or int

        shearDir    axis to be shifted

        shearOrtho  the axis to which the shear is carried out

        center      relative center of the shearing, only important if padding == False
                    possible values: 'center' (default), 'positive', 'negative'

        padding     extra padding of the image, default is true

    Returns
    -------
        img : sheared image

    Examples
    --------
        import NanoImagingPack as nip
        img = nip.readim()

        shear(img, 40)
        shear(img, 40, padding = False)
        shear(img, 40, shearDir = 1, padding = False, center = 'positive')
    """
    from .coordinates import ramp

    if shearOrtho == None:
        shearOrtho = shearDir - 1
    if shearOrtho == shearDir:
        raise ValueError('shearDir and shearOrtho must be different values!')

    if padding == True:
        padVal = np.ceil(np.abs(shearamount)).astype(int)
        padVec = np.zeros((img.ndim, 2)).astype(int)

        if center == 'center':
            padVec[shearDir] = [np.ceil(padVal / 2).astype(int), np.floor(padVal / 2).astype(int)]
        elif center == 'positive':
            if shearamount > 0:
                padVec[shearDir] = [0, padVal]
            else:
                padVec[shearDir] = [padVal, 0]
        else:
            if shearamount < 0:
                padVec[shearDir] = [0, padVal]
            else:
                padVec[shearDir] = [padVal, 0]

        img_pad = DampOutside(img, width=padVec)

    else:
        img_pad = img

    fft = np.fft.fft(img_pad, axis=shearDir)
    fx = ramp(img_pad.shape, shearDir, freq='fftfreq', shift=True)
    sx = shearamount * ramp(img_pad.shape, shearOrtho, center) / (img_pad.shape[shearOrtho] - 1)

    myshear = np.exp(-1j * 2 * np.pi * (fx * sx))
    result = np.real(np.fft.ifft(fft * myshear, axis=shearDir))

    return util.__cast__(result, img)


def rot_2D_im(M, angle):
    """
        Rotates 2D image
        Maintains the size of the image and fills gaps due to rotation up with zeros
        Parts of the image that will be outside of the boarders (aka the edges of the old image) will be clipped!
    """
    import scipy.ndimage as image
    return image.interpolation.rotate(M, angle, reshape=False)


def centroid(im):
    """
        returns tupel with the center of mass of the image
    """
    return cm(np.asarray(im))


# def extract_c(im, center = None, roi = (100,100), axes_center  = None, axes_roi = None, extend ='DEFAULT'):
#     """
#         DEPRECATED (OBSOLETE). replaced by extract
#         Extract a roi around a center coordinate
#         if center is none (default), the roi will be in the center of the image
#         otherwise center should be a tuple
#
#         roi -> edge length of the cube to be extracted -> can be tuple or list or number -> if its a number it will be squareshaped for each direction
#
#         axex_center: to which axes (tupel) do the center cooridnates refere
#         axes_roi: to which axes do the roi cooridnates refere:
#     """
#     if roi is None:
#         return(im);
#     else:
#         if center is None:
#             center = (im.shape[0]//2, im.shape[1]//2);
#
#         if len(center) > np.ndim(im):
#             print('Too much dimensions for center coordinates');
#         else:
#             import numbers
#             if isinstance(roi, numbers.Number):
#                roi = tuple(int(roi) for i in im.shape);
#             if axes_center is None:
#                 axes_center = tuple(range(len(center)));
#             if axes_roi is None:
#                 axes_roi = tuple(range(len(roi)))
#             if len(axes_center) != len(center):
#                 print('axes_center and center must have same length');
#             else:
#                 roi_list = [];
#                 new_ax = [];
#                 for i in range(im.ndim):
#                     if i in axes_center:
#                         if i in axes_roi:
#                             new_ax.append(i);
#                             roi_list.append((center[axes_center.index(i)]-roi[axes_roi.index(i)]//2,center[axes_center.index(i)]+roi[axes_roi.index(i)]//2))
#                 return(extractROI(im, roi_list, tuple(new_ax), extend))


def line_cut(im, coord1=0, coord2=None, thickness=10):
    """
        extract a line cut from an image
        The used method:  image is rotated in the way, that the line is horizontal and then extracted
            if the image has more than 2 Dimensions, the linecut will be done through all 2D images in the stack (first two axes, first one is x, second one is y)

        coord1:
            -> first supporting coordinate of the line cut:
                if tupel or list -> starting coordinate of the line
                if float or int:  this is the tilting angle of the line (in degree)

        coord2:
            -> second supporting coordinate of the line cut:
                if tupel or list
                    -> end coordinate of the line (if coord1 is a coordinate)
                    -> support point of the line (if coord1 is an angle)
                if None: the line cut will go through the center of the image and be symmetrical (e.g. from coord1 throuhg the center and have the double length of the distance coord1 - center )

                    Note: for the angle: the y-axis goes in negative direction!!!
                                         the angle is typically defined (atan(DeltaY/DeltaX))

        thickness is averaging around the line (default = 10)

        EXAMPLES:
            line_cut(im,coord1 = 40)
                creates a line cut through the center and an angle of 40 Degree through the whole image

            line_cut(im,coord1 = 40, coord2 = (120,200)
                creates a line cut through pixel (120,200) and an angle of 40 Degree through the whole image


            line_cut(im,coord1 = (120,200), coord2 = (500,300),thickness = 20)
                creates a line cut from (120,200) to (500,300), averages ove 20 points around the cut

        returns 1D- array with linecut and rotation angle
    """
    through_all = False
    if coord2 is None:
        coord2 = (np.array(im.shape[:2][::-1]) - 1) / 2.
        through_all = True

    if type(coord1) == tuple or type(coord1) == list:
        try:
            m = (coord2[1] - coord1[1]) / (coord2[0] - coord1[0])
            alpha = np.arctan(m) * 180 / np.pi
        except ZeroDivisionError:
            alpha = 90.0

    else:
        alpha = coord1
        coord1 = coord2
        through_all = True

    def __get_line__(img):
        im_rot = rotate(img, -alpha)  # rotate image so, that the line is parallel to the x axis
        org_center = (np.array(img.shape[:2][::-1]) - 1) / 2.  # old center
        rot_center = (np.array(im_rot.shape[:2][::-1]) - 1) / 2.  # center of roted image
        org1 = coord1 - org_center
        org2 = coord2 - org_center
        a = np.deg2rad(-alpha)
        new1 = tuple(np.array([org1[0] * np.cos(a) - org1[1] * np.sin(a), org1[0] * np.sin(a) + org1[1] * np.cos(a)]) + rot_center)
        new2 = tuple(np.array([org2[0] * np.cos(a) - org2[1] * np.sin(a), org2[0] * np.sin(a) + org2[1] * np.cos(a)]) + rot_center)

        # THIS IS FOR DEBUGGING TO SHOW THE ROTATED IMGE AND THE POINTS
        #        v = view(im_rot)
        #        v.set_mark(new1);
        #        v.set_mark(new2);
        #
        if through_all:
            lower_thresh = 0
            upper_thresh = im_rot.shape[0]
        else:
            lower_thresh = np.trunc(np.min([new1[0], new2[0]])).astype(int)
            upper_thresh = np.ceil(np.max([new1[0], new2[0]])).astype(int)

        if thickness == 1:
            # this is in case that the coordinate to we want to extract is not-integer -> than we take tha neighboring elements and weight them!
            gewicht = 1 - (new1[1] - np.trunc(new1[1]))
            line = gewicht * im_rot[lower_thresh:upper_thresh, np.floor(new1[1]).astype(int)] + (1 - gewicht) * im_rot[lower_thresh:upper_thresh, np.ceil(new1[1]).astype(int)]
        else:
            # if we average over several we don't do any weighting!
            line = np.mean(im_rot[lower_thresh:upper_thresh, np.round(new1[1] - thickness / 2).astype(int):np.round(new1[1] + thickness / 2).astype(int)], axis=1)
        return line

    if im.ndim == 2:
        return __get_line__(im)
    else:
        im = np.reshape(im, (im.shape[0], im.shape[1], np.prod(np.asarray(im.shape)[2:])))
        return [__get_line__(im[:, :, i]) for i in range(im.shape[2])]


def extractFt(img, ROIsize=None, mycenter=None, ModifyInput=False, ignoredim=None):
    """
    Identical to exctract, but fixes the zero positon (highest frequency) for even array sizes

    Parameters
    ----------
        ROIsize: size of the ROI to extract. Will automatically be limited by the array sizes when applied. If ROIsize==None the original size is used
        centerpos: center of the ROI in source image to exatract
        PadValue (default=0) : Value to assign to the padded area. If PadValue==None, no padding is performed and the non-existing regions are pruned.
        ModifyInput: If True, the input img will be modified a bit, but this saves computation time and memory. Use only, if you are discarding img anyway.
        ignoredim: a dimension that is ignored in the process of adjusting the extracted bit. This is useful for RFTs and the RFT transform direction.

    Example:
    ----------
        nip.extractFt(nip.ft(nip.readim()),[400,400]) # subsamples

    See also
    -------
    extract

    """
    mysize = img.shape
    if ROIsize is None:
        ROIsize = mysize
    else:
        ROIsize = util.expanddimvec(ROIsize, len(mysize), mysize)

    res = extract(img, ROIsize, mycenter, checkComplex=False)
    if ModifyInput == False:
        img = img + 0.0  # make a copy
    #    if not (ignoredim is None):
    res = fixFtAfterExtract(res, img, ignoredim)  # MODIFES also the img input!

    return res


def fixFtAfterExtract(res, img, ignoredim):
    """
    corrects for the even-size issues when extracting. ATTENTION! THIS ROUTINE MODIFIES img

    Parameters
    ----------
        res: result after extracting the normal way
        img: input image before extraction (WILL BE MODIFED!)

    See also
    -------
    extract

    """
    szold = img.shape
    midold = img.mid()
    sznew = res.shape
    midnew = res.mid()
    for d in range(len(szold)):
        if d != ignoredim:
            if (sznew[d] > szold[d]) and util.iseven(szold[d]):  # the slice needs to be "distributed" to low and high frequency position
                ROILeftPos = midnew[d] - midold[d]  # position of the old border pixel in the new array
                aslice = util.subslice(res, d, ROILeftPos)
                res = util.subsliceAsg(res, d, ROILeftPos + szold[d], aslice / 2.0)  # distribute it evenly, also to keep parseval happy and real arrays real
                res = util.subsliceAsg(res, d, ROILeftPos, aslice / 2.0)  # distribute it evenly, also to keep parseval happy and real arrays real
            if (sznew[d] < szold[d]) and util.iseven(sznew[d]):  # the slice corresponds to both sides of the fourier transform as a sum
                ROIRightPos = szold[d] - (midold[d] - midnew[d])  # position of the removed top border pixel in the old array
                aslice = util.subslice(img, d, ROIRightPos)  # this is one pixel beyond what is cut out
                res = util.subsliceCenteredAdd(res, d, 0, aslice)  # sum both
                ROILeftPos = (midold[d] - midnew[d])  # position of the new corner start positon in the old array
                img = util.subsliceCenteredAdd(img, d, ROILeftPos, aslice)  # to adress the corners (and edges in 3D!) correctly

    #            print("Dimension ignored:"+str(d))
    return res


def extract(img, ROIsize=None, centerpos=None, PadValue=0.0, checkComplex=True):
    """
        extracts a part in an n-dimensional array based on stating the destination ROI size and center in the source

        ROIsize: size of the ROI to extract. Will automatically be limited by the array sizes when applied. If ROIsize==None the original size is used
        centerpos: center of the ROI in source image to exatract
        PadValue (default=0) : Value to assign to the padded area. If PadValue==None, no padding is performed and the non-existing regions are pruned.

        Example:
            nip.centered_extract(nip.readim(),[799,799],[-1,-1],100) # extracts the right bottom quarter of the image

        RH Version
    """
    if checkComplex:
        if np.iscomplexobj(img):
            raise ValueError(
                "Found complex-valued input image. For Fourier-space extraction use extractFt, which handles the borders or use checkComplex=False as an argument to this function")

    mysize = img.shape

    if ROIsize is None:
        ROIsize = mysize
    else:
        ROIsize = util.expanddimvec(ROIsize, len(mysize), mysize)

    if centerpos is None:
        centerpos = [sd // 2 for sd in mysize]
    else:
        centerpos = util.coordsToPos(centerpos, mysize)

    #    print(ROIcoords(centerpos,ROIsize,img.ndim))
    res = img[util.ROIcoords(centerpos, ROIsize, img.ndim)]
    if PadValue is None:
        return res
    else:  # perform padding
        pads = [(max(0, ROIsize[d] // 2 - centerpos[d]), max(0, centerpos[d] + ROIsize[d] - mysize[d] - ROIsize[d] // 2)) for d in range(img.ndim)]
        #        print(pads)
        res = image(np.pad(res, tuple(pads), 'constant', constant_values=PadValue), pixelsize=res.pixelsize)  # ,PadValue
        return res


def extractROI(im, roi=[(0, 10), (0, 10)], axes=None, extend='DEFAULT'):
    """
        returns sub image

            im is the image
            roi is the region of interests  -> this must be a list of tupels
                                                     each tupel gives the minium and the maximum to clip for the given axis!


            axes is a list of axis, if not given the first axes will be used, if given its length has to have the length of the roi list

            extend: True or False, if true, the image will be padded with zeros if the roi exceeds boarders

            example:
                extract(im, roi = [(20,30), (100,300)], axes = (0,2));
                   will clip the image im between 20 and 30 in the x-axis and 100 and 300 in the z-axis

    """
    old_arr = im
    if extend == 'DEFAULT':
        extend = __DEFAULTS__['EXTRACT_EXTEND']

    if type(roi) != list and type(roi) != tuple:
        raise ValueError('roi must be a list or a tuple')
    import numbers
    if isinstance(roi[0], numbers.Number):
        if len(roi) > 1:
            roi = [roi]
            print('Warning: Roi is list of numbers -> taking first two as roi for axis 0')
        else:
            raise ValueError('Lacking roi information')
    if roi is None:
        return util.__cast__(im, old_arr)
    else:
        padding = [[0, 0] for i in range(im.ndim)]
        if axes is None:
            if len(roi) > np.ndim(im):
                print('Error: to much Rois in list or image dimension too small')
            else:
                for r in enumerate(roi):
                    im = im.swapaxes(0, r[0])
                    if r[1][0] < 0:
                        low = 0
                        padding[r[0]][0] = np.abs(r[1][0])
                    else:
                        low = r[1][0];
                    if r[1][1] > im.shape[0]:
                        up = im.shape[0]
                        padding[r[0]][1] = np.abs(r[1][1] - im.shape[0])
                    else:
                        up = r[1][1]
                    im = im[low:up]
                    im = im.swapaxes(0, r[0])
        else:
            if (type(axes) != tuple) or (max(axes) > np.ndim(im)):
                raise ValueError('Either axes is not a tupel or has too much elements')
            else:
                for r in enumerate(roi):
                    im = im.swapaxes(0, axes[r[0]])
                    if r[1][0] < 0:
                        low = 0
                        padding[r[0]][0] = np.abs(r[1][0])
                    else:
                        low = r[1][0];
                    if r[1][1] > im.shape[0]:
                        up = im.shape[0]
                        padding[r[0]][1] = np.abs(r[1][1] - im.shape[0])
                    else:
                        up = r[1][1]
                    im = im[low:up]
                    im = im.swapaxes(0, axes[r[0]])
        if extend == False:
            return util.__cast__(im, old_arr)
        else:
            padding = tuple([tuple(i) for i in padding])
            return util.__cast__(np.lib.pad(im, padding, 'constant'), old_arr)


#class roi:
#    def __init__(im):
#    '''
#        for defining Rois 
#        
#        
#        A roi is defined as list of dictionaries:
#            Each element in the list represents the ROI along one axis
#            In the dictionary, there are the following entries:
#                'axis'
#                'start'
#                'stop'
#                'center'
#                'width'
#    '''
#        pass;
#    def get_center_coords(self):
#        pass;
#    def get_min_max_rois();:
#    def 
#    

from .transformations import ft, ift, rft, irft, ft2d, ift2d, ft3d, ift3d

class image(np.ndarray):
    """
    Image class that inherets from numpy nd Array
    All methods and parameters vom ndarray can be used


    default image is 128X128 filled with zeros:


    Additional parametes:
        pixelsize          list, tuple or number
                                if number the pixelsize is considered to be constant (that value) for all dimensions
                                if lenght of list or tuple is smaller than dimensions, the pxsize of remaining dimensions is set to 1

        unit              string of unit of pixelsize
        info              additional information
        name              image name

        dim_description: list here (as list of strings) potential names along different dimensions (is dictionary)

                            -> example: if along the 4th dimension (ax 3) you have channels 'red','green','blue' -> set it up with im.dim_description['d3'] = ['r','g','b'];

    example:
            import NanoImagingPack as nip;
            x = nip.image();
            x = nip.image((256,256,10))
            x = nip.image(np.arange(0,100,1).reshape(10,10))

    """
    
    def __new__(cls, MyArray = None, pixelsize = None, unit = '', info = '', name = None):
        import numbers
        if MyArray is None:
            MyArray = util.zeros((128,128))
        if type(MyArray) is list or type(MyArray) is tuple:
            res = 1
            for k in MyArray: res*= isinstance(k, numbers.Integral);
            if res == 0:
                raise ValueError('Only integers are allowed in lists or tuples for creating and image')
            else:
                MyArray = util.zeros(MyArray)
        obj = np.asarray(MyArray).view(cls)
        # Here all extradata goes in:
        obj.info = info
        if unit == '':
            obj.unit = __DEFAULTS__['IMG_PIXEL_UNITS']
        else:
            obj.unit = unit
        obj.im_number = 0

        if __DEFAULTS__['IMG_NUMBERING']:
            max_im_number = 0
            for l in locals().values():
                if isinstance(l, image):
                    if l.im_number >= max_im_number: max_im_number = l.im_number+1;
            obj.im_number = max_im_number
        else:
            obj.im_number = 0

        # if name is None:
        #     name = 'Img Nr'+str(obj.im_number)

        obj.spectral_axes =[]
        obj.ax_shifted = []
        obj.metadata = None

#        obj.dim_description = {'d0': [],'d1': [],'d2': [],'d3': [],'d4': [],'d5': []}
        obj.dim_description = None

        obj.name = name
        if obj.pixelsize is None:
            obj.pixelsize = pixelsize
        #     obj.pixelsize = MyArray.ndim*[1.0] # [i*0+1.0 for i in MyArray.shape];
        #     if hasattr(MyArray, 'pixelsize'):
        #         p=MyArray.pixelsize
        #     else:
        #         p = __DEFAULTS__['IMG_PIXELSIZES']
        #     for i in range(len(MyArray.shape)):
        #         if i >= len(p):
        #             obj.pixelsize[-i] = p[0]
        #         else:
        #             obj.pixelsize[-i] = p[-i]
        #
        elif type(pixelsize) == list or type(pixelsize) == tuple:
            if type(pixelsize) == tuple: pixelsize = list(pixelsize);
            if len(pixelsize) > MyArray.ndim: pixelsize = pixelsize[:MyArray.ndim];
            if len(pixelsize) < MyArray.ndim: pixelsize += (MyArray.ndim-len(pixelsize))*[pixelsize] # [i*0+1.0 for i in range(MyArray.ndim-len(pixelsize))];
            obj.pixelsize = pixelsize
        elif isinstance(pixelsize, numbers.Number) :
            obj.pixelsize = MyArray.ndim*[pixelsize] # [i*0+pixelsize for i in MyArray.shape];
        else:
            raise ValueError('Pixelsize must be list, tuple or number')
        return obj

    def __repr__(self):   # will be called in the "print" representation
         return "image"+str(self.shape)

    def _repr_pretty_(self, p, cycle):
#        print("Here is _repr_pretty_  !!! "+__DEFAULTS__['IMG_VIEWER']);
        if cycle:
            p.text('An image')
        else:
            if len(self.shape) == 0:
                print(self)
            else:
                if __DEFAULTS__['IMG_VIEWER'] == 'NIP_VIEW':
                    if len(self.shape) == 1:
                        try:
                            graph(self, title = self.name)
                        except:
                            graph(self)

                    else:
                        self.v = view(self)
                elif __DEFAULTS__['IMG_VIEWER'] == 'VIEW5D':  # RH 3.2.19
                    mysize=self.shape
                    if (len(mysize) == 1) and (mysize[0] < 5):
                        print("np.image"+str(self))
                    elif (len(mysize) == 2) and (mysize[0] < 5) and (mysize[1] < 5):
                        print("np.image"+str(self))
                    elif type(self) == list and len(list) > 10:
                        print("list of images of length " + str(len(self)))
                    else:
                        # if self.name is None:
                        #     self.name = util.caller_string(3)  # use the name of the caller
                        self.v = v5(self)
                elif __DEFAULTS__['IMG_VIEWER'] == 'INFO':
                    print('Image :'+self.name)
                    print('Shape: '+str(self.shape))
                    print('Pixelsize: '+str(self.pixelsize))
                    print('Units: '+self.unit)
                    print('Info: '+self.info)
                    print('')
                    print('Array-Info:')
                    print(self.__array_interface__)
                    print('')
                    print('Array flags:')
                    print(self.flags)
                else:
                    print(self)

    def set_pixelsize(self, pixelsize):
        import numbers
        if pixelsize is None:
            p = __DEFAULTS__['IMG_PIXELSIZES']
            if p is not None:
                self.pixelsize = self.ndim*[1.0] # [i*0+1.0 for i in self.shape];
                self.pixelsize[-len(self.shape)::] = p[-len(self.shape)::]
            else:
                self.pixelsize = p
        elif type(pixelsize) == list or type(pixelsize) == tuple:
            if type(pixelsize) == tuple: pixelsize = list(pixelsize);
            if len(pixelsize) > self.ndim: pixelsize = pixelsize[-self.ndim:];
            if len(pixelsize) < self.ndim: pixelsize = [i*0+1.0 for i in range(self.ndim-len(pixelsize))] + pixelsize;
            self.pixelsize = pixelsize
        elif isinstance(pixelsize, numbers.Number) :
            self.pixelsize = [i*0+pixelsize for i in self.shape]
        else:
            raise ValueError('Pixelsize must be list, tuple or number')

    def __compare_pixel_sizes__(self, im2):
        """
            compares the pixelsize setting of the present image with a second image im2
        """
        if isinstance(im2, image):
                    for pxs1, pxs2 in zip(self.pixelsize, im2.pixelsize):
                        if pxs1 != pxs2:
                            print('Warning: images have different pixelsizes! Computing FRC based of pixelsize of image 1')

    def __get_img_coord_from_roi__(self,roi_coord,roi = None, axes = None):
        
        if type(roi_coord) != tuple and type(roi_coord) != list:
            raise TypeError('Coords must be a tuple or a list of the coordinates')
        else:
            if len(roi_coord) > self.ndim:
                print('Waring: More coordinates than dimensions')
            if axes is None:
                if len(roi_coord) > self.ndim:
                    roi_coord = roi_coord[:self.ndim]
            else:
                if max(axes)>self.ndim:
                    print('Maximum dimension is '+str(self.ndim)+' -> Higher axis will be ignored')
                pos = list(-1*np.ones((min((max(axes)+1, self.ndim)))))
                for a in enumerate(axes):

                    if a[1] < self.ndim:
                        print(pos[a[1]])
                        pos[a[1]] = roi[a[0]]
                roi = pos

            # GGF hier die rois nach axesn ordnen
            glob_coords = []
            for c in enumerate(roi_coord):
                try:
                    glob_coords += [c[1]+roi[c[0]][0]]
                except:
                    glob_coords += [c[1]]
            return tuple(glob_coords)

    def imsave(self, path = None, form = 'tif', rescale = True, BitDepth = 16, Floating = False, truncate = True):
        """
            Like the imsave method, but:
                    - If no path given the default directory (as stated in config) will be used for directory, and the image name for the file name
                    - Image information (e.g. pixelsize) will be saved in the metadata of the file
        """
        if path is None:
            path = join(__DEFAULTS__['DIRECTORY'],self.name)
        imsave(self, path = path, form = form, rescale = rescale, BitDepth = BitDepth, Floating = Floating, truncate = truncate)

    def midSlice(self, ax = 0):
        """
            returns the subvolume by extracting the middle position along the given axis (as tuple) as seen for ft, i.e. im.shape//2
            ----
            ax : which axes (list of all axes)
            If nothing given, it returns the value at the mid pos for all axis
        """

        return self[self.mid(0)]

    def midVal(self, ax = None):
        """
            returns the value of the midpos of the given axis (as tuple) as seen for ft, i.e. im.shape//2
            ----
            ax : which axes (list of all axes)
            If nothing given, it returns the value at the mid pos for all axis
        """
        return self[self.mid()]

    def midValAsg(self, val):
        """
            assigns a value to the middle coordinate and returns the modified image
            ----
            val: value to assign
        """
        return util.midValAsg(self, val)

    def mid(self, ax = None):
        """
            returns the midpos of the given axis (as tuple) as seen for ft, i.e. im.shape//2
            ----
            ax : which axes (list of all axes)
            If nothing given, it returns the mid pos for all axis
        """
        import numbers
        if ax is None:
            ax = [s for s in range(self.ndim)]
        elif isinstance(ax, numbers.Integral):
            ax = [ax]
        elif type(ax) == list or type(ax) == tuple:
            ax= list(ax)
        else:
            raise TypeError('Wrong data type for axis')
        pos = []
        for i in range(self.ndim):
            if i in ax:
                pos += [self.shape[i]//2]
            else:
                pos+=[slice(0,self.shape[i])]
        return tuple(pos)

    def expanddim(self, ndims, trailing = False):
        # TODO: Dealing with pixel sizes of expanded image
        return util.expanddim(self, ndims = ndims, trailing= trailing)


    def ft(self, shift = True ,shift_before = True, ret = 'complex', axes = None,  s = None, norm = None): # RH: Changed this
        #im = ft(self, shift = shift, shift_before= shift_before,ret = ret, axes = axes,  s = s, norm = norm);
        return ft(self, shift_after= shift, shift_before= shift_before, ret = ret, axes = axes, s = s, norm = norm)

    def ift(self, shift = True,shift_before = True, ret ='complex', axes = None, s = None, norm = None):  # RH: Changed this
        return ift(self, shift_after= shift, shift_before =shift_before, ret = ret, axes = axes, s = s, norm = norm)

    def ift2d(self, shift = True,shift_before = True, ret ='complex', s = None, norm = None):
        return ift2d(self, shift_after= shift, shift_before =shift_before, ret = ret, s = s, norm = norm)

    def ift3d(self, shift = True,shift_before = True, ret ='complex', s = None, norm = None):
        return ift3d(self, shift_after= shift, shift_before =shift_before, ret = ret, s = s, norm = norm)

    def ft2d(self, shift = True ,shift_before = True, ret = 'complex',  s = None, norm = None):
        return ft2d(self, shift_after= shift, shift_before= shift_before, ret = ret, s = s, norm = norm)

    def ft3d(self, shift = True ,shift_before = True, ret = 'complex',  s = None, norm = None):
        return ft3d(self, shift_after= shift, shift_before= shift_before, ret = ret, s = s, norm = norm)

    def rft(self, shift_after = False, shift_before = False, ret = 'complex', axes = None,  s = None, norm = None):
        return rft(self, shift_after = shift, shift_before = shift_before, ret = ret, axes = axes, s = s, norm = norm)
    def irft(self,s, shift_after = False,shift_before = False, ret ='complex', axes = None,  norm = None):
        return irft(self, s, shift_after = shift, shift_before = shift_before, ret =ret, axes = axes, norm = norm)
    def poisson(self, NPhot = 100):
        im = poisson(self, NPhot).view(image)
        im.info = self.info+ '\n Poission noise, Maximum Photon number = '+str(NPhot)+'\n'
        im.name = self.name+ ' (Poisson, NPhot = '+str(NPhot)+')'
        return im

    def DampEdge(self, width = None, rwidth = 0.1 ,axes =None,func = coshalf, method = "damp", sigma = 4.0):
        im = DampEdge(self, width = width, rwidth=rwidth, axes =axes, func = func, method=method, sigma=sigma)
        im.info += 'Damp Edged, width: '+str(width)+', method = '+func.__name__+'\n'
        return im

    def DampOutside(self, width=None, rwidth=0.1, usepixels=3, mykernel=None, kernelpower=3):
        im = DampOutside(self, width=width, rwidth=rwidth, usepixels=usepixels, mykernel=mykernel,
                         kernelpower=kernelpower)
        im.info += 'Padded damping, width: ' + str(width) + '\n'
        return im

    def check_complex(self):
        return __check_complex__(self)

    def make_odd(self,ax):
        return make_odd(self, ax)

    def match_size(self,im2,axes = 0, padmode ='constant', odd = False):
        ret_im1, ret_im2 = match_size(self,im2,axes = axes, padmode =padmode, odd = odd)
        if isinstance(im2, image):
            ret_im2.__array_finalize__(im2)
        return ret_im1, ret_im2

    def FRC(self,im2, num_rings = 10, correct_shift = True):
        self.__compare_pixel_sizes__(im2)
        if (self.shape[0] != self.shape[1]) or self.pixelsize[0] != self.pixelsize[1]:
            print('Warning: Image is not quadratic! Be aware of numerical errors!')
        return FRC(self, im2, pixel_size = self.pixelsize, num_rings = num_rings, correct_shift = True)

    #    def __array_wrap__(self, arr, context = None):
#        return('bam')
    def px_freq_step(self):
        """
            returns the frequency step in of one pixel in the fourier space for a given image as a list for the different coordinates
        """
        return coordinates.px_freq_step(self.shape, self.pixelsize)

    def max_freq(self):
        """
            Returns the maximum frequency for each dimension which can be transferred
        """
        return [fs * s / 2 - fs for fs, s in zip(self.px_freq_step(), self.shape)]

    def threshold(self, t1, t2 =None):
        return threshold(self, t1, t2)

    def histogram(self,name ='', bins=65535, range=None, normed=False, weights=None, density=None):
        return histogram(self, name ='', bins=65535, range=range, normed=normed, weights=weights, density=density)

    def cat(self, imlist, axis = None, destdims = None):
        if isinstance(imlist, np.ndarray):
            im =cat([imlist,self],axis=axis, destdims = destdims)
        elif isinstance(imlist, list) or isinstance(imlist, tuple):
            im =cat(list(imlist)+[self],axis= axis, destdims = destdims)

        else:
            raise TypeError('Imlist is wrong data type')
        im = im.view(image)
        im.__array_finalize__(self)
        return im

    def shift(self,delta,direction =0, pixelwise = False):
        im =shift(self, delta,direction,pixelwise).view(image)
        im.__array_finalize__(self)
        return im

    def shiftx(self,delta):
        return self.shift(delta, direction = 0)

    def shifty(self,delta):
        return self.shift(delta, direction = 1)

    def shiftz(self,delta):
        return self.shift(delta, direction = 2)

    def line_cut(self, coord1 = 0, coord2 = None,thickness = 10):
        return line_cut(self, coord1 = 0, coord2 = None, thickness = 10)

    def __correllator__(self,im2, axes = None, mode = 'convolution', phase_only = True):
        self.__compare_pixel_sizes__(im2)
        im = __correllator__(self,im2, axes = axes, mode = mode, phase_only = phase_only).view(image)
        im.__array_finalize__(self)
        return im

    def correl_phase(self, im2,  axes = None):
        """
            Phase correlation with image 2
        """
        return self.__correllator__(im2, axes = axes, mode ='correlation', phase_only = True)

    def correl(self, im2, axes = None,phase_only = False):
        return self.__correllator__(im2, axes = axes, mode ='correlation', phase_only = phase_only)

    def convolve(self, im2,  axes = None,phase_only = False):
        return self.__correllator__(im2, axes = axes, mode ='convolution', phase_only = phase_only)

    # Todo: DEPRICATED: DELET
    # def supersample(self, factor = 2, axis = (0,1)):
    #
    #     return(supersample(self, factor, axis))

    def normalize(self, mode, r = None):    
        return util.normalize(self, mode, r)

    def extract_coordinate(self, c):
        return util.extract_coordinate(self, c)
    
    def bfp_image(self, wavelength, focal_length):
        """
            This returns the fourier transform of the image but with given coordinates as if you were placing the image in the Front focal plane and observing the pattern in the back focal plane
            Make sure the units of wavelengths and focal length are correct

            Only takes first 2 Dimensions
        """
        im = ft2d(self)
        im.info = 'BFP image of '+self.name+' using a lens of focal length '+str(focal_length)
        im.unit = 'same as focal length'
        im.pixelsize[0] = im.pixelsize[0]*wavelength*focal_length
        im.pixelsize[1] = im.pixelsize[1]*wavelength*focal_length
        return im

    def rotate(self, angle, axes =(1,0)):
        """
            rotate round axis in a certain angle in degree
        """
        return rotate(self, angle, axes = axes, reshape = False)
    

    def extractROI(self, roi = [(0, 10), (0, 10)], axes = None, extend='DEFAULT'):
        im = extractROI(self, roi = roi, axes = axes, extend = extend).view(image)
        im.__array_finalize__(self)
        return im

    def extract(self, ROIsize=None, centerpos=None, PadValue=0.0, checkComplex=True):
         im = extract(self, ROIsize=ROIsize, centerpos=centerpos, PadValue=PadValue, checkComplex=checkComplex).view(image)
         im.__array_finalize__(self)
         return im

    # def extract_c(self, center = None, roi = (100,100), axes_center  = None, axes_roi = None, extend = 'DEFAULT'):
    #     im = extract_c(self, center = center, roi = roi, axes_center = axes_center, axes_roi = axes_roi, extend = extend).view(image);
    #     im.__array_finalize__(self);
    #     return(im);

    
    def max_coord(self, roi = None, axes = None, ret = 'global'):
        """
            Maximum coordinate in certain roi (list of (min, max))
            ret:
                if global -> returns global coordinates of image, otherwise coords in that roi
        """
        coord = self.extractROI(roi, axes).extract_coordinat(np.argmax(self.extractROI(roi, axes)))
        if ret == 'global':
            return self.__get_img_coord_from_roi__(coord, roi = roi, axes = axes)
        else:
            return coord
    def min_coord(self, roi = None, axes = None, ret = 'global'):
        """
            Minimum coordinate in certain roi (list of (min, max))
            ret:
                if global -> returns global coordinates of image, otherwise coords in that roi
        """
        coord = self.extractROI(roi, axes).extract_coordinat(np.argmin(self.extractROI(roi, axes)))
        if ret == 'global':
            return self.__get_img_coord_from_roi__(coord, roi = roi, axes = axes)
        else:
            return coord
        
    def centroid(self, roi = None, axes = None, ret = 'global'):
        """
            Centroid coordinate in certain roi (list of (min, max))
            ret:
                if global -> returns global coordinates of image, otherwise coords in that roi
        """
        coord = cm(np.asarray(self.extractROI(roi, axes)))
        if ret == 'global':
            return self.__get_img_coord_from_roi__(coord, roi = roi, axes = axes)
        else:
            return coord

#     def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):  # this method is called whenever you use a ufunc
#         f = {
#             "reduce": ufunc.reduce,
#             "accumulate": ufunc.accumulate,
#             "reduceat": ufunc.reduceat,
#             "outer": ufunc.outer,
#             "at": ufunc.at,
#             "__call__": ufunc,
#         }
#         [inputs[i] = inputs[i]  for i in inputs]
#         output = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
# #        output = image(f[method](*inputs, **kwargs))  # convert the inputs to np.ndarray to prevent recursion, call the function, then cast it back as ExampleTensor
#         output.__dict__ = self.__dict__  # carry forward attributes
#         return output

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):   # example method from: https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, image):
                in_no.append(i)
                args.append(input_.view(np.ndarray))  # converts to a numpy view
            else:
                args.append(input_)

        outputs = kwargs.pop('out', None)
        out_no = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, image):
                    out_no.append(j)
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        # info = {}
        # if in_no:
        #     info['inputs'] = in_no
        # if out_no:
        #     info['outputs'] = out_no

        results = super(image, self).__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        # if method == 'at':
        #     if isinstance(inputs[0], image):
        #         inputs[0].info = info
        #     return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(image)   # also calles __arrray_finalize__ which copies the pixelsizes
                         if output is None else output)
                        for result, output in zip(results, outputs))

        if method == 'reduce' and isinstance(inputs[0],image) and inputs[0].pixelsize is not None:
            keepdims = kwargs['keepdims']
            for i, output_ in enumerate(results):
                if isinstance(output_, image):
                    if keepdims is False:  # else the pixelsizes are OK
                        axes = kwargs['axis']
                        ndims=len(inputs[0].pixelsize)
                        if axes is None:
                            output_.pixelsize = None
                        else:
                            if not isinstance(axes, tuple):
                                axes = (axes,)
                            output_.pixelsize = [inputs[0].pixelsize[e] for e in range(ndims) if e not in axes and ((e-ndims) not in axes)]
                    else:
                        output_.pixelsize = inputs[0].pixelsize.copy()

        if method=="__call__":
            for i, output_ in enumerate(results):
                output_.pixelsize = util.longestPixelsize(inputs)

        # if results and isinstance(results[0], image):
        #     results[0].info = info

        return results[0] if len(results) == 1 else results

    # def __array_wrap__(self, result):
    #     return image(result)  # can add other attributes of self as constructor

    def __array_finalize__(self, obj):
        if obj is None: return;   # is true for explicit creation of array
        # This stuff is important in case that the "__new__" method isn't called
        # e.g.: z is ndarray and a view of z as image class is created (imz = z.view(image))
        
        # THIS CODE SERVES TO DETECT WHICH AXIS HAVE BEEN ELIMINATED OR CHANGED IN THE CASTNG PROCESS 
        # E.G. DUE TO INDEXING OR SWAPPING!
        
#        if  len(self.shape) != len(obj.shape):
#            if obj.__array_interface__['strides'] is None:
#                s = (np.cumprod(obj.shape[::-1]))[::-1];
#                s = list(np.append(s[1:],1)*obj.dtype.itemsize);
#            else:
#                s = list(obj.__array_interface__['strides']);
#            if self.__array_interface__['strides'] is None:
#                s1 = (np.cumprod(self.shape[::-1]))[::-1];
#                s1 = list(np.append(s1[1:],1)*self.dtype.itemsize);
#            else:
#                s1 = list(self.__array_interface__['strides']);
#            new_axes = [s.index(el) for el in s1 if el in s];
#        else:
#            new_axes = [s for s in range(self.ndim)];

#           TODO: DEPRICATE THIS PART!!!
#           TODO: Handle what happens in case of transposing and ax swapping
#
#           Some problems:
#               1) the matplotlib widgets creates a lot of views which alway cause problemse (thats the reason for the try below)
#               2) for ax swaping (also rolling, transposing changes in strides can't be identified in __array_finalize__)
        
        # p = __DEFAULTS__['IMG_PIXELSIZES']
        # if type(obj) == type(self):
        #     pxs = self.ndim*[1.0] # [i*0+1.0 for i in self.shape];
        #     for i in range(len(self.shape)):
        #         try:
        #             pxs[-i-1] = obj.pixelsize[-i-1]  # new_axes[-i-1]
        #         except:
        #             if i >= len(p):
        #                 pxs[-i-1] = p[0]
        #             else:
        #                 pxs[-i-1] = p[-i-1]
        # else:
        #     pxs = self.ndim*[1.0] # [i*0+1.0 for i in self.shape];
        #     for i in range(len(self.shape)):
        #         if i >= len(p):
        #             pxs[-i-1] = p[0]
        #         else:
        #             pxs[-i-1] = p[-i-1]
        # self.pixelsize = pxs
        if isinstance(obj, image) and obj.pixelsize is not None:
            self.pixelsize = obj.pixelsize.copy()
        else:
            self.pixelsize = __DEFAULTS__['IMG_PIXELSIZES'] # which is None
#        self.dim_description = getattr(obj,'dim_description', {'d0': [],'d1': [],'d2': [],'d3': [],'d4': [],'d5': []})
        self.dim_description = getattr(obj, 'dim_description', None)
        self.metadata = getattr(obj,'metadata',[])
        self.spectral_axes = getattr(obj, 'spectral_axes', [])
        self.ax_shifted = getattr(obj, 'ax_shifted', [])
        #self.pixelsize = getattr(obj, 'pixelsize',  pxs);
        self.info = getattr(obj, 'info',  '')
        self.unit = getattr(obj, 'unit',  __DEFAULTS__['IMG_PIXEL_UNITS'])
        self.name = getattr(obj, 'name', '')
        self.im_number = getattr(obj, 'im_number', 0)
        max_im_number = 0
        if __DEFAULTS__['IMG_NUMBERING']:
            for l in locals().values():
                if isinstance(l,image):
                    if l.im_number >= max_im_number: max_im_number = l.im_number+1;
            self.im_number = max_im_number
        else:
            self.im_number = 0
        # self.name = None # 'Img Nr'+str(max_im_number)


def shiftby(img, avec):
    return np.real(ift(coordinates.applyPhaseRamp(ft(img), avec)))


def shift2Dby(img, avec):
    return np.real(ift2d(coordinates.applyPhaseRamp(ft2d(img), avec)))
