#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:21:21 2017

@author: root

 All kind of image handling
 
 -> read
 -> write
 -> Dampedge
 -> correlations -> create correlations etc.
 -> algin (modes:2D, 3D)
 -> rotate
 -> noise

"""

import tifffile as tif;
import numpy as np;
import NanoImagingPack as nip
from pkg_resources import resource_filename
from .config import DBG_MSG, __DEFAULTS__;
from IPython.lib.pretty import pretty;
from .functions import cossqr, gaussian, coshalf, linear;
from .util import make_damp_ramp,get_type,subslice,expanddim, __cast__;
from .view5d import v5 # for debugging
from .FileUtils import list_files;

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
class image(np.ndarray):
    '''
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

    '''
    
    def __new__(cls, MyArray = None, pixelsize = None, unit = '', info = '', name = None):
        import numbers;
        if MyArray is None:
            MyArray = np.zeros((128,128));
        if type(MyArray) is list or type(MyArray) is tuple:
            res = 1;
            for k in MyArray: res*= isinstance(k, numbers.Integral);
            if res == 0:
                raise ValueError('Only integers are allowed in lists or tuples for creating and image');    
            else:
                MyArray = np.zeros(MyArray);
        obj = np.asarray(MyArray).view(cls);
        # Here all extradata goes in:
        obj.info = info;
        if unit == '':
            obj.unit = __DEFAULTS__['IMG_PIXEL_UNITS'];
        else:
            obj.unit = unit;
        obj.im_number = 0;
        
        if __DEFAULTS__['IMG_NUMBERING']:
            max_im_number = 0;
            for l in locals().values():
                if type(l) == image:
                    if l.im_number >= max_im_number: max_im_number = l.im_number+1;
            obj.im_number = max_im_number;
        else:
            obj.im_number = 0;
            
            
        if name is None:
            name = 'Img Nr'+str(obj.im_number);
       
        obj.spectral_axes =[];
        obj.ax_shifted = [];
        obj.metadata = None;
        
        obj.dim_description = {'d0': [],'d1': [],'d2': [],'d3': [],'d4': [],'d5': []}
        
        
        obj.name = name;
        if pixelsize is None:
            obj.pixelsize = [i*0+1.0 for i in MyArray.shape];
            for i in range(len(MyArray.shape)):
                p = __DEFAULTS__['IMG_PIXELSIZES'];
                if i >= len(p):
                    obj.pixelsize[i] = p[len(p)-1];
                else:
                    obj.pixelsize[i] = p[i];
                    
        elif type(pixelsize) == list or type(pixelsize) == tuple:
            if type(pixelsize) == tuple: pixelsize = list(pixelsize);
            if len(pixelsize) > MyArray.ndim: pixelsize = pixelsize[:MyArray.ndim];
            if len(pixelsize) < MyArray.ndim: pixelsize += [i*0+1.0 for i in range(MyArray.ndim-len(pixelsize))];
            obj.pixelsize = pixelsize
        elif isinstance(pixelsize, numbers.Number) :
            obj.pixelsize = [i*0+pixelsize for i in MyArray.shape];
        else:
            raise ValueError('Pixelsize must be list, tuple or number')
        return(obj);

    def _repr_pretty_(self, p, cycle):
#        print("Here is _repr_pretty_  !!! "+__DEFAULTS__['IMG_VIEWER']);
        if cycle:
            p.text('An image');
        else:
            if len(self.shape) == 0:
                print(self);
            else:
                if __DEFAULTS__['IMG_VIEWER'] == 'NIP_VIEW':
                    if len(self.shape) == 1:
                        from .view import graph;
                        try:
                            graph(self, title = self.name);
                        except:
                            graph(self);

                    else:
                        from .view import view;
                        self.v = view(self);
                elif __DEFAULTS__['IMG_VIEWER'] == 'VIEW5D':  # RH 3.2.19
                    from .view5d import v5;
                    self.v = v5(self);
                elif __DEFAULTS__['IMG_VIEWER'] == 'INFO':
                    print('Image :'+self.name);
                    print('Shape: '+str(self.shape));
                    print('Pixelsize: '+str(self.pixelsize));
                    print('Units: '+self.unit);
                    print('Info: '+self.info);
                    print('');
                    print('Array-Info:')
                    print(self.__array_interface__);
                    print('');
                    print('Array flags:')
                    print(self.flags);
                else:
                    print(self);
    def set_pixelsize(self, pixelsize):
        import numbers;
        if pixelsize is None:
            p = __DEFAULTS__['IMG_PIXELSIZES'];
            self.pixelsize = [i*0+1.0 for i in self.shape];    
            self.pixelsize[-len(self.shape)::] = p[-len(self.shape)::]
        elif type(pixelsize) == list or type(pixelsize) == tuple:
            if type(pixelsize) == tuple: pixelsize = list(pixelsize);
            if len(pixelsize) > self.ndim: pixelsize = pixelsize[-self.ndim:];
            if len(pixelsize) < self.ndim: pixelsize = [i*0+1.0 for i in range(self.ndim-len(pixelsize))] + pixelsize;
            self.pixelsize = pixelsize
        elif isinstance(pixelsize, numbers.Number) :
            self.pixelsize = [i*0+pixelsize for i in self.shape];
        else:
            raise ValueError('Pixelsize must be list, tuple or number')

    def __compare_pixel_sizes__(self, im2):
        '''
            compares the pixelsize setting of the present image with a second image im2
        '''
        if type(im2) == image:
                    for pxs1, pxs2 in zip(self.pixelsize, im2.pixelsize):
                        if pxs1 != pxs2:
                            print('Warning: images have different pixelsizes! Computing FRC based of pixelsize of image 1');

    def __get_img_coord_from_roi__(self,roi_coord,roi = None, axes = None):
        
        if type(roi_coord) != tuple and type(roi_coord) != list:
            raise TypeError('Coords must be a tuple or a list of the coordinates');
        else:
            if len(roi_coord) > self.ndim:
                print('Waring: More coordinates than dimensions');
            if axes is None:
                if len(roi_coord) > self.ndim:
                    roi_coord = roi_coord[:self.ndim];
            else:
                if max(axes)>self.ndim:
                    print('Maximum dimension is '+str(self.ndim)+' -> Higher axis will be ignored')
                pos = list(-1*np.ones((min((max(axes)+1, self.ndim)))));    
                for a in enumerate(axes):

                    if a[1] < self.ndim:
                        print(pos[a[1]])
                        pos[a[1]] = roi[a[0]];
                roi = pos;
            
            # GGF hier die rois nach axesn ordnen
            glob_coords = [];
            for c in enumerate(roi_coord):
                try:
                    glob_coords += [c[1]+roi[c[0]][0]]
                except:
                    glob_coords += [c[1]];
            return(tuple(glob_coords));
 
                        
    def imsave(self, path = None, form = 'tif', rescale = True, BitDepth = 16, Floating = False, truncate = True):
        '''
            Like the imsave method, but:
                    - If no path given the default directory (as stated in config) will be used for directory, and the image name for the file name
                    - Image information (e.g. pixelsize) will be saved in the metadata of the file
        '''
        if path is None:
            from os.path import join;
            path = join(__DEFAULTS__['DIRECTORY'],self.name);
        imsave(self, path = path, form = form, rescale = rescale, BitDepth = BitDepth, Floating = Floating, truncate = truncate);
    
    def mid(self, ax = None):
        '''
            returns the midpos of the given axis (as tuple) as seen for ft, i.e. im.shape//2
            ----
            ax : which axes (list of all axes)
            If nothing given, it returns the mid pos for all axis
        '''
        import numbers;
        if ax is None:
            ax = [s for s in range(self.ndim)]
        elif isinstance(ax, numbers.Integral):
            ax = [ax];
        elif type(ax) == list or type(ax) == tuple:
            ax= list(ax);
        else:
            raise TypeError('Wrong data type for axis');
        pos = [];
        for i in range(self.ndim):
            if i in ax:
                pos += [self.shape[i]//2];
            else:
                pos+=[slice(0,self.shape[i])];
        return(tuple(pos));
    
    def ft(self, shift = True ,shift_before = False, ret = 'complex', axes = None,  s = None, norm = None):
        from .transformations import ft;
        #im = ft(self, shift = shift, shift_before= shift_before,ret = ret, axes = axes,  s = s, norm = norm);
        return(ft(self, shift = shift, shift_before= shift_before,ret = ret, axes = axes,  s = s, norm = norm));

    def ift(self, shift = False,shift_before = True, ret ='complex', axes = None, s = None, norm = None):
        from .transformations import ift;
        return(ift(self, shift = shift,shift_before =shift_before, ret = ret, axes = axes, s = s, norm = norm));
    def ift2d(self, shift = False,shift_before = True, ret ='complex', s = None, norm = None):
        from .transformations import ift2d;
        return(ift2d(self, shift = shift,shift_before =shift_before, ret = ret, s = s, norm = norm));
    def ift3d(self, shift = False,shift_before = True, ret ='complex', s = None, norm = None):
        from .transformations import ift3d;
        return(ift3d(self, shift = shift,shift_before =shift_before, ret = ret, s = s, norm = norm));
    def ft2d(self, shift = True ,shift_before = False, ret = 'complex',  s = None, norm = None):
        from .transformations import ft2d;
        return(ft2d(self, shift = shift, shift_before= shift_before,ret = ret,  s = s, norm = norm));
    def ft3d(self, shift = True ,shift_before = False, ret = 'complex',  s = None, norm = None):
        from .transformations import ft3d;
        return(ft3d(self, shift = shift, shift_before= shift_before,ret = ret,  s = s, norm = norm));

    def rft(self, shift = False, shift_before = False, ret = 'complex', axes = None,  s = None, norm = None, real_return = None, real_axis = None):
        from .transformations import rft;
        return(rft(self, shift = shift, shift_before = shift_before, ret = ret, axes = axes,  s = s, norm = norm, real_return = real_return, real_axis = real_axis))
    def irft(self, shift = False,shift_before = False, ret ='complex', axes = None, s = None, norm = None, real_axis = None):
        from .transformations import irft;
        return(irft(self, shift = shift,shift_before = shift_before, ret =ret, axes = axes, s = s, norm = norm, real_axis = real_axis))
    def poisson(self, NPhot = 100):
        from .noise import poisson;
        im = poisson(self, NPhot).view(image);
        im.info = self.info+ '\n Poission noise, Maximum Photon number = '+str(NPhot)+'\n';
        im.name = self.name+ ' (Poisson, NPhot = '+str(NPhot)+')';
        return(im);
    def DampEdge(self, width = 10, axes =(0,1),func = cossqr):
        im = DampEdge(self, width = width, axes =axes, func = func);
        im.info += 'Damp Edged, width: '+str(width)+', method = '+func.__name__+'\n';
        return(im);
    def check_complex(self):
        return(__check_complex__(self));
    def make_odd(self,ax):
        return(make_odd(self, ax));
    def match_size(self,M2,ax = 0, padmode ='constant', odd = True):
        ret_im1, ret_im2 = match_size(self,M2,ax = ax, padmode =padmode, odd = odd);
        if type(M2) == image:
            ret_im2.__array_finalize__(M2);
        return(ret_im1, ret_im2);
    def FRC(self,im2, num_rings = 10, correct_shift = True):
        self.__compare_pixel_sizes__(im2);
        if (self.shape[0] != self.shape[1]) or self.pixelsize[0] != self.pixelsize[1]:
            print('Warning: Image is not quadratic! Be aware of numerical errors!')
        return(FRC(self, im2, pixel_size = self.pixelsize, num_rings = num_rings, correct_shift = True));
#    def __array_wrap__(self, arr, context = None):
#        return('bam')
    def px_freq_step(self):
        '''
            returns the frequency step in of one pixel in the fourier space for a given image as a list for the different coordinates
        '''
        from .coordinates import px_freq_step;
        return(px_freq_step(self.shape, self.pixelsize));
    
    def max_freq(self):
        '''
            Returns the maximum frequency for each dimension which can be transferred
        '''
        return([fs*s/2-fs for fs, s in zip(self.px_freq_step(),self.shape)]);
    def threshold(self, t1, t2 =None):
        return(threshold(self, t1, t2));
    def histogram(self,name ='', bins=65535, range=None, normed=False, weights=None, density=None):
        return(histogram(self,name ='', bins=65535, range=range, normed=normed, weights=weights, density=density));
    def cat(self, imlist, ax):
        if get_type(imlist) == 'list':
            im =cat(imlist+[self],ax=ax);
        elif get_type(imlist)[0] == 'array':
            im =cat([imlist,self],ax= ax);
            
        else:
            raise TypeError('Imlist is wrong data type')
        im = im.view(image);
        im.__array_finalize__(self);
        return(im);
    def shift(self,delta,direction =0):
        im =shift(self, delta,direction,).view(image);
        im.__array_finalize__(self);
        return(im);
    def shiftx(self,delta):
        return(self.shift(delta, direction = 0));
    def shifty(self,delta):
        return(self.shift(delta, direction = 1));
    def shiftz(self,delta):
        return(self.shift(delta, direction = 2));
    def line_cut(self, coord1 = 0, coord2 = None,thickness = 10):
        return(line_cut(self,coord1 = 0, coord2 = None,thickness = 10));
    
    def __correllator__(self,M2, axes = None, mode = 'convolution', phase_only = True):
        self.__compare_pixel_sizes__(M2);
        im = __correllator__(self,M2, axes = axes, mode = mode, phase_only = phase_only).view(image)
        im.__array_finalize__(self);
        return(im);
        
    def correl_phase(self, M2,  axes = None):
        '''
            Phase correlation with image 2
        '''
        return(self.__correllator__(M2, axes = axes, mode = 'correlation', phase_only = True));
    def correl(self, M2, axes = None,phase_only = False):
        return(self.__correllator__(M2, axes = axes, mode = 'correlation', phase_only = phase_only));
    def convolve(self, M2,  axes = None,phase_only = False):
        return(self.__correllator__(M2, axes = axes, mode = 'convolution', phase_only = phase_only));
    def supersample(self, factor = 2, axis = (0,1)):
        
        return(supersample(self, factor, axis))

    def normalize(self, mode, r = None):    
        from .util import normalize;
        return(normalize(self, mode,r));
    
    def extract_coordinate(self, c):
        from .util import extract_coordinate;
        return(extract_coordinate(self, c))
    
    def bfp_image(self, wavelength, focal_length):
        '''
            This returns the fourier transform of the image but with given coordinates as if you were placing the image in the Front focal plane and observing the pattern in the back focal plane
            Make sure the units of wavelengths and focal length are correct
            
            Only takes first 2 Dimensions
        '''
        from .transformations import ft2d;
        im = ft2d(self);
        im.info = 'BFP image of '+self.name+' using a lens of focal length '+str(focal_length);
        im.unit = 'same as focal length'
        im.pixelsize[0] = im.pixelsize[0]*wavelength*focal_length;
        im.pixelsize[1] = im.pixelsize[1]*wavelength*focal_length;
        return(im);
    
    def rotate(self, angle, axes =(1,0)):
        '''
            rotate round axis in a certain angle in degree
        '''
        from scipy.ndimage.interpolation import rotate;
        return(rotate(self, angle, axes = axes, reshape = False))
    

    def extract(self, roi = [(0,10),(0,10)], axes = None, extend= 'DEFAULT'):
        im = extract(self, roi = roi, axes = axes, extend = extend).view(image);
        im.__array_finalize__(self);
        return(im)
    def extract_c(self, center = None, roi = (100,100), axes_center  = None, axes_roi = None, extend = 'DEFAULT'):
        im = extract_c(self, center = center, roi = roi, axes_center = axes_center, axes_roi = axes_roi, extend = extend).view(image);
        im.__array_finalize__(self);
        return(im);

    
    def max_coord(self, roi = None, axes = None, ret = 'global'):
        '''
            Maximum coordinate in certain roi (list of (min, max))
            ret:
                if global -> returns global coordinates of image, otherwise coords in that roi
        '''
        coord = self.extract(roi, axes).extract_coordinat(np.argmax(self.extract(roi, axes)));
        if ret == 'global':
            return(self.__get_img_coord_from_roi__(coord,roi = roi, axes = axes))
        else:
            return(coord)
    def min_coord(self, roi = None, axes = None, ret = 'global'):
        '''
            Minimum coordinate in certain roi (list of (min, max))
            ret:
                if global -> returns global coordinates of image, otherwise coords in that roi
        '''
        coord = self.extract(roi, axes).extract_coordinat(np.argmin(self.extract(roi, axes)));
        if ret == 'global':
            return(self.__get_img_coord_from_roi__(coord,roi = roi, axes = axes))
        else:
            return(coord)
        
    def centroid(self, roi = None, axes = None, ret = 'global'):
        '''
            Centroid coordinate in certain roi (list of (min, max))
            ret:
                if global -> returns global coordinates of image, otherwise coords in that roi
        '''
        from scipy.ndimage.measurements import center_of_mass as cm;
        coord = cm(np.asarray(self.extract(roi, axes)));
        if ret == 'global':
            return(self.__get_img_coord_from_roi__(coord,roi = roi, axes = axes))
        else:
            return(coord);

    
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
        
        p = __DEFAULTS__['IMG_PIXELSIZES'];
        if type(obj) == type(self):
            pxs = [i*0+1.0 for i in self.shape];
            for i in range(len(self.shape)):
                try:
                    pxs[i] = obj.pixelsize[new_axes[i]];
                except:
                    if i >= len(p):
                        pxs[i] = p[len(p)-1];
                    else:
                        pxs[i] = p[i];
        else:
            pxs = [i*0+1.0 for i in self.shape];
            for i in range(len(self.shape)):
                if i >= len(p):
                    pxs[i] = p[len(p)-1];
                else:
                    pxs[i] = p[i];
        self.pixelsize = pxs;
        self.dim_description = getattr(obj,'dim_description', {'d0': [],'d1': [],'d2': [],'d3': [],'d4': [],'d5': []});
        self.metadata = getattr(obj,'metadata',[]);
        self.spectral_axes = getattr(obj, 'spectral_axes', []);
        self.ax_shifted = getattr(obj, 'ax_shifted', []);
        #self.pixelsize = getattr(obj, 'pixelsize',  pxs);
        self.info = getattr(obj, 'info',  '');
        self.unit = getattr(obj, 'unit',  __DEFAULTS__['IMG_PIXEL_UNITS']);
        self.name = getattr(obj, 'name', '');
        self.im_number = getattr(obj, 'im_number', 0);
        max_im_number = 0;
        if __DEFAULTS__['IMG_NUMBERING']:
            for l in locals().values():
                if type(l) == image:
                    if l.im_number >= max_im_number: max_im_number = l.im_number+1;
            self.im_number = max_im_number;
        else:
            self.im_number = 0;
        self.name = 'Img Nr'+str(max_im_number);
        
        

def save_to_3D_tif(directory, file_prototype, save_name, sort ='date', key = None):
    '''
        load a stack of 2D images and save it as 3D Stack
        
        directory:   directory of the images
        file_prototype: string: all files where the name contains the strings be loaded
        save_name: save name
        sort: how to sort files ('name' or 'date' or 'integer_key')
            integer key: give key character, after which an integer number will be searched
    '''

    from .FileUtils import get_sorted_file_list;
    from os.path import join;
    flist = get_sorted_file_list(directory, file_prototype, sort, key);
    print(flist)
    img = np.asarray([readim(join(directory, file)) for file in flist]);
    img = np.swapaxes(img, 0,1);
    img = np.swapaxes(img, 1,2);
    imsave(img, join(directory, save_name));     

    
'''
    Todo: Stack images!!!
'''                 
          

def imsave(img, path, form = 'tif', rescale = True, BitDepth = 16, Floating = False, truncate = True):
    '''
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
    '''
    from os.path import splitext, split, isdir;
    from os import mkdir;

    folder = split(path)[0];
    if not isdir(folder):
        mkdir(folder);
        print('Creating Folder ... '+ folder);

    ext = splitext(path)[-1][1:];
    if ext == '':
        path+='.'+form;
    
    
    if get_type(img)[1] == 'image':
        metadata = {'name':img.name, 'pixelsize':str(img.pixelsize), 'units':img.unit, 'info': img.info};
    elif get_type(img)[1] == 'ndarray':
        metadata = {}
    else:
        raise TypeError('Wrong type. Can only save ndarrays or image type');
    if Floating == False:
        if truncate:
            img = img*(img>0);

        if BitDepth != 'auto':
            if rescale:
                if BitDepth == 1:
                    img = img/np.max(img)*(255);
                else:
                    img = img/np.max(img)*(2**BitDepth-1);
            if np.max(img) >= 2**BitDepth:
                print('WARNING! Image maximum larger than '+str(2**BitDepth-1)+'! RESCALING!!!');
                img = img/np.max(img)*(2**BitDepth-1);
            if np.min(img) >= 0:
                if BitDepth <= 8:
                    img = np.uint8(img);
                elif BitDepth <= 16:
                    img = np.uint16(img);
                elif BitDepth <= 32:
                    img = np.uint32(img);
                else:
                    img = np.uint64(img);
            else:
                print('ATTENTION: NEGATIVE VALUES IN IMAGE! Using int casting, not uint!')
                if BitDepth <= 8:
                    img = np.int8(img);
                elif BitDepth <= 16:
                    img = np.int16(img);
                elif BitDepth <= 32:
                    img = np.int32(img);
                else:
                    img = np.int64(img);
                        
    if form in __DEFAULTS__['IMG_TIFF_FORMATS']:
        tif.imsave(path,img, metadata = metadata);   # RH 2.2.19 deleted: np.transpose
    else:
        import PIL;
        #img = np;   # RH 2.2.19 deleted: np.transpose   CK commented line
        img = PIL.Image.fromarray(img)
        if BitDepth == 1:
            img=img.convert("1");
        else:
            img= img.convert("L")
        
        img.save(path,form);
    return(img);
          
#def readim(path =resource_filename("NanoImagingPack","resources/todesstern.tif"), which = None, pixelsize = None):
def readim(path =None, which = None, pixelsize = None):
    '''
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
    '''
    from os.path import splitext, isfile, split;
    l = [];
    if path is None:
        path = __DEFAULTS__['IMG_DEFAULT_IMG_NAME'];
    for p in __DEFAULTS__['IMG_DEFAULT_IMG_FOLDERS']:
        l+=(list_files(p, ''));
    default_names = [splitext(split(el)[1])[0] for el in l];
    if path in default_names:
        path = l[default_names.index(path)];

    #from .config import __IMAGE_FORMATS__;
# CK 20.02.19: seems that stuff does the same like mine, but without centralized management via config
#    from os.path import splitext, isfile;
#    if not isfile(path): # Try to find the file either directly or in the resource directory ...
#        path =resource_filename("NanoImagingPack","resources/"+path)
#        if not isfile(path):
#        if not isfile(path):
#            path =resource_filename("NanoImagingPack","resources/"+path+".tif")

    if isfile(path):
        ext = splitext(path)[-1][1:];
        
        if ext.lower() in __DEFAULTS__['IMG_TIFF_FORMATS']:
            
            if which == None:
                img = (tif.imread(path)); # RH 2.2.19 deleted: np.transpose
            else:
                img = (tif.imread(path, key = which));  # RH 2.2.19 deleted: np.transpose
            img = img.view(image);
            #img.pixelsize = pixelsize;
            img.set_pixelsize(pixelsize);
        elif ext.lower() in __DEFAULTS__['IMG_ZEISS_FORMATS']:
            # TODO HERE: READ ONLY SELECTED SLICES OF THE IMAGE 
            from .EXTERNAL.Gohlke_czi_reader import imread as cziread;
            if __DEFAULTS__['IMG_SQUEEZE_ZEISS']:
                img, meta = cziread(path);
                img = img.squeeze().view(image);  # RH 2.2.19 deleted: np.transpose
            else:
                img, meta = cziread(path);
                img = img.view(image);  # RH 2.2.19 deleted: np.transpose
            img = img.view(image, pixelsize);
            img.metadata = meta;
            # TODO: Pixelsizes aus Metadaten fischen
            img.set_pixelsize(pixelsize);
        else:
            try:
                import PIL.Image as IM;
                img = (np.array(IM.open(path)));  # RH 2.2.19 deleted: np.transpose
                img = img.view(image);
                img.set_pixelsize(pixelsize);
            except OSError:
                raise ValueError('No valid image file');
        return(img);
    else:
        raise ValueError('No valid filename')

def readtimeseries(path, filename = '', roi = [-1,-1,-1,-1], channel = 0, ret_old_img_dim = False, axis = -3):
    '''
        This reads a set of 2D Tiff files and creates one 3-D Stack from it.
        Path: The folder, containing the tiff-files
        Filename (optional): Only files containing this filename will be read, can also be a list of filenames which are inteted to be read
        channel: in case of multi channel image: which channel to be read!
         
        
         roi = [x,y, width_x, width_y] -> if nothting given than the whole image will be loaded
        ret_old_img_dim : return old image dimension (maximum dimension of all images read!)
        At which axis you want to concatenate the images (default is -3, that means z-direction)
    '''
    from os import listdir
    from os.path import isfile, join, splitext, split
    from .FileUtils import list_files
    from .config import __DEFAULTS__
    if type(filename)!= list:
        file_list = [split(f)[1] for f in list_files(path = path, file_prototype=filename, subfolders = False)];
        imtypes = __DEFAULTS__['IMG_TIFF_FORMATS']+__DEFAULTS__['IMG_IMG_FORMATS'];
        name_list = [f for f in file_list if splitext(f)[1][1:] in imtypes];
        #file_list = [f for f in listdir(path) if isfile(join(path, f))];
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
        name_list = filename;        

    max_im_dim =0;
    print('Reading images ...')
    number  =0;
    #dim = [0,0];
    #im_list =None;
    final_im =[];
    for name in name_list:
        print(name, end = ' ... ', flush = True);
        if roi == [-1,-1,-1,-1]:
            if (splitext(name)[1][1:]=='png' or splitext(name)[1][1:]=='bmp'):
                from scipy import misc
                im = misc.imread(path+name);
            else:
                im = tif.imread(path+name);
        else:
            if (splitext(name)[1][1:]=='png' or splitext(name)[1][1:] == 'bmp'):
                from scipy import misc
                im = misc.imread(path+name)[roi[1]:roi[3]+roi[1],roi[0]:roi[2]+roi[0]];
            else:
                im = tif.imread(path+name )[roi[1]:roi[3]+roi[1],roi[0]:roi[2]+roi[0]];
        # im = im.transpose();     # CK 20.02.19 transpose outcommented
        print(' Shape: '+str(im.shape), end = ' ; ', flush = True);
        if im.ndim > max_im_dim:
            max_im_dim = im.ndim;
        if np.ndim(im) != 2:
            
            print('Reading channel '+str(channel));
            im = im[:,:,channel]
        if number == 0:
            final_im = im;
        else:
            final_im = cat((final_im, im), axis);
        number +=1;
                
    '''
    DEPRECIATET CODE:                
                dim = [np.size(im, axis =0), np.size(im, axis =1)];
                number +=1;
                if im_list == None:
                    im_list = np.expand_dims(im,2);
                else:
                    im_list = np.concatenate((im_list,np.expand_dims(im,2)),axis =2)
            else:
                if (dim == [np.size(im, axis =0), np.size(im, axis =1)]):
                    number +=1;
                    if im_list == None:
                        im_list = np.expand_dims(im,2);
                    else:
                        im_list = np.concatenate((im_list,np.expand_dims(im,2)),axis =2)
                else:
                    print('Wrong size of image '+ name);
     '''
    print();
    print(str(number)+' images read!');
    
    if  ret_old_img_dim:
        return(image(final_im),max_im_dim)
    else:
        return(image(final_im))
    
    
def DampEdge(img, width = None, rwidth=0.1, axes =None, func = coshalf, method="damp", sigma=4.0):
    '''
        DampEdge function 
        
        im  image to damp edges 
        
        rwidth : relative width (default : 0.1 meaning 10%)
            width in relation to the image size along this dimenions. Can be a single number or a tuple
            
        width (None: rwidht is used, else width takes precedence)
            -> characteristic absolute width for damping
            -> can be integer, than every (given) axis is damped by the same size
            -> can be list or tupel -> than individual damping for given axis
            
        axes-> which axes to be damped (default is (0,1))

        func   - which function shall be used for damping -> some are stated in functions.py, first element should be x, second one the length (Damping length!)
                e.g. cossqr, coshalf, linear
                default: coshalf
        
        method -> which method should be used?
                -> "zero" : dims down to zero
                -> "damp" : blurs to an averaged mean (default)
                -> "moisan" : HF/LF split method according to Moisan, J Math Imaging Vis (2011) 39: 161–179, DOI 10.1007/s10851-010-0227-1
    
        return image with damped edges
        
        TODO in FUTURE: padding of the image before damping
        Example:
            import NanoImagingPack as nip
            nip.DampEdge(nip.readim()[400:])
    '''
    img=img.astype(np.float32)
    res = np.ones(img.shape);    
    if width==None:
        width=tuple(np.round(np.array(img.shape)*np.array(rwidth)).astype("int"))
        
    if axes==None:
        axes=np.arange(0,img.ndim).tolist()
    if type(width) == int:
        width = [width];
    if type(width) == tuple:
        width = list(width);
    if len(width) < len(axes):
        ext = np.ones(len(axes)-len(width))*width[-1];
        width.extend(list(ext.astype(int)));
        
    res=img
    mysum=nip.zeros(img.shape)
    sz=img.shape;
    den=-2*len(set(axes)); # use only the counting dimensions
    for i in range(len(img.shape)):
        if i in axes:
            line = np.arange(0,img.shape[i],1);
            ramp = make_damp_ramp(width[i],func);            
            if method=="zero":
                line = cat((ramp[::-1],np.ones(img.shape[i]-2*width[i]),ramp),-1);
                goal=0.0 # dim down to zero
            elif method=="moisan":
#                for d=1:ndims(img)
                top=nip.subslice(img,i,0)
                bottom=nip.subslice(img,i,-1)
                mysum=nip.subsliceAsg(mysum,i,0,bottom-top + nip.subslice(mysum,i,0));
                mysum=nip.subsliceAsg(mysum,i,-1,top-bottom + nip.subslice(mysum,i,-1));
                den=den+2*np.cos(2*np.pi*nip.ramp(nip.dimVec(i,sz[i],len(sz)),i,freq='freq'))
            elif method=="damp":
                line = nip.cat((ramp[::-1],np.ones(img.shape[i]-2*width[i]+1),ramp[:-1]),0);  # to make it perfectly cyclic
                top=nip.subslice(img,i,0)
                bottom=nip.subslice(img,i,-1)
                goal = (top+bottom)/2.0
                kernel=gaussian(goal.shape,sigma)
                goal = convolve(goal,kernel,norm2nd=True)
            else:
                raise ValueError("DampEdge: Unknown method. Choose: damp, moisan or zero.")
            #res = res.swapaxes(0,i); # The broadcasting works only for Python versions >3.5
#            res = res.swapaxes(len(img.shape)-1,i); # The broadcasting works only for Python versions >3.5
        if method!="moisan":
            line = nip.castdim(line,img.ndim,i) # The broadcasting works only for Python versions >3.5
            try:
                res = res*line + (1.0-line)*goal
            except ValueError:
                print('Broadcasting failed! Maybe the Python version is too old ... - Now we have to use repmat and reshape :(')
                from numpy.matlib import repmat;
                res *= np.reshape(repmat(line, 1, np.prod(res.shape[1:])),res.shape, order = 'F');
    if method=="moisan":
        den=nip.MidValAsg(nip.image(den),1);  # to avoid the division by zero error
        den=nip.ft(mysum)/den;
        den=nip.MidValAsg(den,0);  # kill the zero frequency
        den=np.real(nip.ift(den))
        res=img-den
        
    #return(res)
    return(__cast__(img*res.view(image),img));

def __check_complex__(im):
    '''
        checks the type an image:
            returns True if the image is complex
            otherwise returns False:
    '''
    return(not (np.issubdtype(im.dtype, np.floating) or np.issubdtype(im.dtype, np.integer)))
    
def make_odd(M,ax):
    '''
    Make a image odd in the given axis by removing one pixel line
    '''
    if (np.mod(np.size(M, axis = ax),2)==0):
        M = np.delete(M,0,ax);
    return(__cast__(M,M));

def match_size(M1,M2,ax, padmode ='constant', odd = True):
    '''
        Adjusts the size of the two images 
        The image size will be made odd before adjusting if not given differently 

        
        pad mode 
            clip: clips the larger image to the size of the smaller one, whereas the first part is used!
            constant: fill up with zeros symmetrical (e.g. same size above and below)
            const_below: fill up below
        further allowed pad modes: c.f. numpy.pad -> edge takes the edge values, constant: fills up with zeros
        
        if odd = True (standard) the images will be respective size will be clipped to be odd
    '''
    if np.ndim(M1)==np.ndim(M2):
        if odd:
            M1 = make_odd(M1,ax);
            M2 = make_odd(M2,ax);
        diff = np.size(M1,axis =ax)-np.size(M2,axis =ax);   # Get difference in size
        if padmode == 'clip':   
            M1 = np.swapaxes(M1,0,ax);
            M2 = np.swapaxes(M2,0,ax);
            if np.size(M1, axis =0) > np.size(M2,axis =0):
                M1 = M1[:np.size(M2, axis =0)];
            elif np.size(M1, axis =0) < np.size(M2,axis =0):
                M2 = M2[:np.size(M1, axis =0)];
            M1 = np.swapaxes(M1,0,ax);
            M2 = np.swapaxes(M2,0,ax);
        elif padmode == 'constant':
            padding = [];
            for i in range(np.ndim(M1)):
                if i == ax:
                    padding.append((np.abs(diff//2),np.abs(diff//2+np.mod(diff,2))   ));
                else:
                    padding.append((0,0));
            if diff <0:
                M1 = np.lib.pad(M1,tuple(padding),padmode);
            else:
                M2 = np.lib.pad(M2,tuple(padding),padmode);
        elif padmode == 'const_below':
            padding = [];
            for i in range(np.ndim(M1)):
                if i == ax:
                    padding.append((0,np.abs(diff)   ));
                else:
                    padding.append((0,0));
            if diff <0:
                M1 = np.lib.pad(M1,tuple(padding),'constant');
            else:
                M2 = np.lib.pad(M2,tuple(padding),'constant');

    else:
        print('Cannot match sizes as arrays have different dimensions!');

    return(__cast__(M1,M1), __cast__(M2,M2))


#TODO:
#def optimize_shift(im1,im2, method = 'optimize_max'):
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
#            from .util import max_coord;
#            corr = correl(im1, im2)[0];
#            max_corr = np.max(corr);
#            argsmax_corr = max_coord(np.argmax(corr));
#            im2 = shift(im2, im2.shape//2)
            
 

def FRC(im1,im2, pixel_size = 62, num_rings = 10, correct_shift = True):
    '''
        Compute the Fourier ring correlation (frc) between 2 2D-images
        
        im1, im2: the images
        pixel_size: pixelsize; -> can be tuple or lis or number
        num_rings: number of ringst to compute the FRC
        correct_shift: corrects a possible shift between the images -> until now: no sub-pixel-shift!  
        
        Notice:
            Although the function does adjust the images sizes if the images are not square shaped and have the same dimensions it is recommened to only use square images of the same sizes, since the code is not sufficiently testet!
        
        Returns a tupel:  (FRC strength, spatial frequency [1/unit(pixel_size)])
    '''
    if (im1.ndim == 2) and (im2.ndim == 2):
        if im1.shape[0]>im1.shape[1]:
            print('Adjusting shape of image 1 in y -direction')
            im1 = np.lib.pad(im1, ((0,0),(0,im1.shape[0]-im1.shape[1])),'constant')
        if im1.shape[1]>im1.shape[0]:
            print('Adjusting shape of image 1 in x -direction')
            im1 = np.lib.pad(im1, ((0,im1.shape[0]-im1.shape[1]),(0,0)),'constant')
       
        im1,im2 = match_size(im1,im2, padmode= 'const_below', ax =0, odd = False);
        im1,im2 = match_size(im1,im2, padmode= 'const_below', ax =1, odd = False);
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
        
        if correct_shift ==  True:
            from .util import max_coord;
            corr = np.abs(correl(im1, im2));
            argmax_corr = max_coord(corr);
            print(argmax_corr);
            im2 = shift_center(im2, -im2.shape[0]//2+argmax_corr[0], -im2.shape[1]//2+argmax_corr[1]);
            print(max_coord(np.abs(correl(im1, im2))));
        
        
        from numpy.matlib import repmat;
        from .coordinates import xx, yy;
        from .transformations import ft;
        im1 = ft(im1,shift = True, shift_before= False, ret = 'complex', axes = (0,1));
        im2 = ft(im2,shift = True, shift_before= False, ret = 'complex', axes = (0,1));
        
        
        if type(im1) == image:
            pxs = image.pixelsize;
        else:
            import numbers
            if isinstance(pixel_size, numbers.Number):
                pxs = [pixel_size, pixel_size];
            elif (type(pixel_size) == tuple or type(pixel_size) == list) and len(pixel_size) > 1:
                pxs = pixel_size[:2];
            else:
                raise ValueError('Pixelsize must be list, tuple or number');
        f_step = [1/(p*s) for (p,s) in zip(pxs, im1.shape)];
        max_f = [1/(2*p)-1/(p*s) for (p,s) in zip(pxs, im1.shape)];
        k_max = min(max_f);
        max_pixel = [k_max/f for f in f_step];
        rad_x = np.linspace(0,max_pixel[0],num_rings+1);
        rad_y = np.linspace(0,max_pixel[1],num_rings+1);

        try:
            vol_im1 = np.reshape(repmat(im1,num_rings,1),(num_rings,im1.shape[0], im1.shape[1]));
            vol_im1 = vol_im1.swapaxes(0,1);
            vol_im1 = vol_im1.swapaxes(1,2);
            vol_im2 = np.reshape(repmat(im2,num_rings,1),(num_rings,im2.shape[0], im2.shape[1]));
            vol_im2 = vol_im2.swapaxes(0,1);
            vol_im2 = vol_im2.swapaxes(1,2);    
            mask = ((xx((im1.shape[0],im1.shape[1],num_rings)))**2/rad_x[1:]**2+(yy((im1.shape[0],im1.shape[1],num_rings)))**2/rad_y[1:]**2<1)*1-((xx((im1.shape[0],im1.shape[1],num_rings)))**2/rad_x[:rad_x.size-1]**2+(yy((im1.shape[0],im1.shape[1],num_rings)))**2/rad_y[:rad_x.size-1]**2<1)*1;
            

            fcr = np.sum(vol_im1*np.conjugate(vol_im2)*mask,axis = (0,1))/(np.sqrt(np.sum(np.abs(vol_im1)**2*mask,axis =(0,1)  )  *np.sum(np.abs(vol_im2)**2*mask,axis =(0,1)  )   ));
        except MemoryError:
            print('Memory problem! Analyzing rings sequentially!');
            for i in range(num_rings):
                print('Ringnumber '+str(i));
                mask = ((xx((im1.shape[0],im1.shape[1]))**2/rad_x[i+1]**2+yy((im1.shape[0],im1.shape[1]))**2/rad_x[i+1]**2)<1)*1-((xx((im1.shape[0],im1.shape[1]))**2/rad_x[i]**2+yy((im1.shape[0],im1.shape[1]))**2/rad_y[i]**2)<1)*1;
                el = np.sum(im1*np.conjugate(im2)*mask)/(np.sqrt(np.sum(np.abs(im1)**2*mask) * np.sum(np.abs(im2)**2*mask) ));
                if i == 0:
                    fcr = el;
                else:
                    fcr = cat((fcr, el),0)
        return(np.abs(fcr), np.linspace(0,k_max,num_rings));
    else:
        raise TypeError('Wrong image dimension! Only 2 Dimensional images allowed');
        return;


    
#def supersample(M, direction)    

def threshold(im, t1, t2 =None):
    '''
        Threshold image
        returns a binary image -> if pixelvalue >= t1 -> 1, else: 0
        
        if t2 is given than it returns 1 for pixelvalues between t1 and t2
    '''
    if t2 == None:
        return((im>=t1)*1)
    else:
        if t1>t2:
            h = t2; 
            t2 = t1;
            t1 = h;
        return(__cast__((im>=t1)*(im<=t2),im));

def get_max(M,region = [-1,-1,-1,-1]):
    '''
    Get maximum value in a certain region
    
    region = [x,y, width_x, width_y]
    
    !!!only works for 2D right now!!!
    '''
    from .util import max_coord;
    if region[0]!= -1:
        M = M[region[1]-region[3]//2:region[1]+region[3]//2,region[0]-region[2]//2:region[0]+region[2]//2]
    MAX = max_coord(M);
    return(region[0]-region[2]//2+MAX[1],region[1]-region[3]//2+MAX[0])

def adjust_dims(imlist, maxdim = None):
    '''
        This functions takes a tupel of a list of images and adds dimensions in a way that all images in the list (or the tupel) have the same number of dimensions afterwards. T
        Maxdim defines the dimension number of the final images.
        If maxdim is smaller than the dimension size in one image or not given, maxdim will be ignored and the dimension number of that image with the most dimensions will be used instead
        
        Extra dimensions will be added at the end
    '''
    
    def __exp_dims__(im):
        for i in range(im.ndim, maxdim):
            im = np.expand_dims(im, 0);  # i RH 2.2.19
        return(im);    

    err = False;
    if type(imlist) == list or type(imlist) == tuple:
        dimsize_list =[];
        for im in imlist:
            if type(im) == np.ndarray:
                dimsize_list.append(im.ndim);
            else:
                err = True;
        if maxdim == None or maxdim < max(dimsize_list):
            #if maxdim != None: print('Given maximum dimension too small -> adjusting it to ' + str(maxdim));
            maxdim = max(dimsize_list);
        if err == False:
            imlist = [__exp_dims__(im) for im in imlist];
        else:
            raise TypeError('Wrong data input -> give list');
    else:
        raise TypeError('Wrong data input');
    return(imlist)

def toClipboard(im, separator = '\t', decimal_delimiter = '.', transpose = False):
    import win32clipboard as clipboard
    '''
        Save image to clipboard
        only works with 1D or 2D images
    '''
    
    # save to clipboard
    
    # TODO: nD darstellung
    # Put string into clipboard (open, clear, set, close)
    if transpose:
        im = im.transpose;
    s = np.array2string(im);
    s = s.replace(']\n ','\n');
    s = s.replace('\n ','')
    s = s.replace('[','');
    s = s.replace(']','');
    s = s.replace('.',decimal_delimiter);
    pos = 0;
    while pos >= 0:
        pos = s.find(' ');
        if pos != len(s)-1:
            if s[pos +1 ] == ' ':
                s = s[:pos]+s[1+pos:];
            else:
                s = s[:pos]+separator+s[1+pos:];
        else:
            s = s[:pos];
                
        
                
                	
#    for i in im: 
#        s+= str(i)+separator;
    clipboard.OpenClipboard()
    clipboard.EmptyClipboard()
    clipboard.SetClipboardText(s);
    clipboard.CloseClipboard()
    
def cat(imlist, axis=None, destdims=None):
    '''
        This function takes a list or a tuple of images and stacks them at the given axis.
        If the images have different dimensions the dimensions will be adjusted (using adjust_dims)
        
        If the axis is larger than the image dimensions the image dimensions will be expanded
        
        If the images do have different sizes, the sizes will also be adjusted by using the match_size function
        
        Be aware that axis count is according to the numpy convention, e.g. Z,Y,X meaning ax=0 for Z!
        default: axis=-(numdims+1), meaning the unused dimension (before the first). Positive values will adress the axis in the numpy convention and throw an error if over the limit. Negative values will automatically expand.
        destdims (default: None): The number of destination dimensions to expand to
    '''
    imlist = list(imlist);
    imlist = [x for x in imlist if x is not None];
    for i in range(len(imlist)):
        if type(imlist[i]) != np.ndarray:
            imlist[i] = np.asarray(imlist[i]);
    imlist = tuple(imlist);
    shapes = np.asarray([list(im.shape) for im in imlist])    
    
    if axis == None:
        axis = -shapes.shape[1]-1
         
    if destdims==None:
        if axis < 0:
            maxdims=-axis
    #        imlist = adjust_dims(imlist, maxdims);
        else:
            maxdims=axis+1
    else:
        if axis+1>destdims:
            print('WARNING: Axis larger than destdims -> adjusting destims to '+str(axis+1));
            destdims = axis+1; 
        maxdims=destdims

    imlist = adjust_dims(imlist, maxdims);
    shapes = np.asarray([list(im.shape) for im in imlist])    

    
    if axis <0:
        ax = shapes.shape[1]+axis;
    else:
        ax = axis;
    for i in range(shapes.shape[1]):
        if (np.max(shapes[:,i]) != np.min(shapes[:,i])) and (i != ax):
            imlist = [match_size(im, imlist[np.argmax(shapes[:,i])], i, padmode ='constant', odd = False)[0] for im in imlist] 
    #return(np.concatenate((imlist),axis).squeeze());
    return image(np.concatenate((imlist),axis));  #RH  2.2.19
    
    
def histogram(im,name ='', bins=65535, range=None, normed=False, weights=None, density=None):
    from .view import graph;
    h = np.histogram(im, bins = bins)
    graph(y=h[0],x=h[1][:len(h[0])], title = 'Histogram of image '+name, x_label = 'Bins', y_label = 'Counts', legend = [])
    return(h);
    
def shift(M,delta,direction =0):
    '''
        Shifts an image M for a distance of delta pixels in the given direction using the FT shift theorem
        
        Delta can be given as float or int: than M is shifted for delta in the respective direction
        
        Delta can also be given as list (of ints and floats)
                -> In this case M is shifted according for multiple directions. The direction doesn't need to be given than
                
                Examples:
                    shift(M, 30, 2):   shifts the matrix M for 30 pixels in direction 2 (mostly z)
                    shift(M, [20,30,0,8]):  shift the matrix M for 20 pixels in direction 0, 30 pixels in direction 1 and 8 pixels in direction 3
                    
        full_fft: if True -> the full FFT is used for computation, else: the rfft
                                Full FFT might be more accurate but also requires more computation time
    '''
    from .transformations import ft, ift, rft, irft;
    from .coordinates import ramp;
    import numbers;
    
    old_arr = M;

    if type(delta) == tuple:
        delta = list(delta);
    if type(delta) == list:
        axes = list(range(len(delta)))
    elif isinstance(delta, numbers.Real):
        axes = [direction];
        delta = [delta];
    
    t = [(0,0) for i in range(M.ndim)]
    old_shape = M.shape;
    for d,ax in zip(delta, axes):
        t[ax] = (int(np.ceil(np.abs(d))),int(np.ceil(np.abs(d))));
    M = np.lib.pad(M,tuple(t),'constant');                                    # Change Boundaries to avoid ringing
    
    if M.dtype == np.complexfloating:
        FT = ft(M, shift = True,shift_before=False, axes = axes,s= None, norm = None, ret = 'complex');
        real_ax = -1;
    else:
        FT = rft(M, shift = True,shift_before=False, axes = axes,s= None, norm = None, ret = 'complex',real_return = None);
        from .transformations import __REAL_AXIS__;
        real_ax = __REAL_AXIS__;
        
    phaseramp = np.zeros(FT.shape);
    for d, ax in zip(delta, axes):        
        if ax == real_ax:
            
            phaseramp += ramp(FT.shape,ramp_dim = ax, corner = 'positive')*2*np.pi*d/(M.shape[ax]);
        else:
            phaseramp += ramp(FT.shape,ramp_dim = ax, corner = 'center')*2*np.pi*d/(M.shape[ax]);
    phaseramp = np.exp(-1j*phaseramp)
    if M.dtype == np.complexfloating:
        M = ift(FT*phaseramp, shift = False,shift_before=True, axes = axes,s= None, norm = None, ret = 'complex');
    else:
        M = irft(FT*phaseramp, shift = False,shift_before=True, axes = axes,s= None, norm = None, ret = 'complex', real_axis = real_ax);
    for d, ax in zip(delta, axes):
        M = M.swapaxes(0,ax);
        M = M[int(np.ceil(np.abs(d))):old_shape[ax]+int(np.ceil(np.abs(d)))];
        M = M.swapaxes(0,ax);
    return(__cast__(M, old_arr))

def shiftx(M,delta):
    '''
        shift image M for delta pixels in x-direction
    '''
    return(shift(M,delta,0))

def shifty(M,delta):
    '''
        shift image M for delta pixels in y-direction
    '''
    
    return(shift(M,delta,1))

def shiftz(M,delta):
    '''
        shift image M for delta pixels in z-direction
    '''
    return(shift(M,delta,2))
    
def shift_center(M,x,y):
    '''
        Shifts the center of an image by full pixels
        
        !!!only works for 2D right now!!!
        
        use the "shift" method for subpixel shifts
    '''
    non = lambda s: s if s<0 else None
    mom = lambda s: max(0,s)
    New = np.zeros_like(M);
    New[mom(x):non(x), mom(y):non(y)] = M[mom(-x):non(-x), mom(-y):non(-y)]
    return(__cast__(New,M))
    
def __correllator__(M1,M2, axes = None, mode = 'convolution', phase_only = False, norm2nd=False):
    '''
        Correlator for images
        
         If the images have different sizes, zeros will be padded at the rims
         M1,M2: Images
         axes:   along which axes
         mode= 'convolution' or 'correlation'
         phase only: Phase correlation
         
         If inputs are real than it tries to take the rfft
        
    '''
    if np.ndim(M1) == np.ndim(M2):
        
        for axis in range(np.ndim(M1)):
            if M1.shape[axis] != M2.shape[axis]:
                M1,M2 = match_size(M1,M2,axis, padmode ='constant', odd = False);
                print('Matching sizes at axis '+str(axis));
        
        #create axes list
        if axes == None:
                axes = list(range(len(M1.shape)));
        if type(axes) == int:
            axes = [axes];
        try: 
            if np.issubdtype(axes.dtype, np.integer):
                axes = [axes];
        except AttributeError:
            pass;

        from .transformations import ft, ift, rft, irft;
        
        if M1.dtype == np.complexfloating or M2.dtype == np.complexfloating:
            FT1 = ft(M1, shift = False, shift_before = False, norm = None, ret = 'complex', axes = axes);
            FT2 = ft(M2, shift = False, shift_before = False, norm = None, ret = 'complex', axes = axes);
        else:
            FT1 = rft(M1, shift = False, shift_before = False, norm = None, ret = 'complex', axes = axes, real_return = None);
            FT2 = rft(M2, shift = False, shift_before = False, norm = None, ret = 'complex', axes = axes, real_return = None);

        if norm2nd == True:
            FT2 = FT2 / np.abs(FT2.flat[0])
            
        if mode == 'convolution':
            if phase_only:
                cor = np.exp(1j*(np.angle(FT1)+np.angle(FT2)))
            else:
                cor = FT1*FT2;
        elif mode == 'correlation':
            if phase_only:
                cor = np.exp(1j*(np.angle(FT1)-np.angle(FT2)))
            else:
                cor = FT1*FT2.conjugate();
        else:
            raise ValueError('Wrong mode');
            return(M1);
        if M1.dtype == np.complexfloating or M2.dtype == np.complexfloating:
            return(__cast__(ift(cor, shift = True, shift_before = False, norm = None, ret ='complex', axes = axes),old_arr))
        else:
            if __DEFAULTS__['CC_ABS_RETURN']:
                return(__cast__(irft(cor, shift = True, shift_before = False, norm = None,ret = 'abs', axes =axes, real_axis='GLOBAL'), old_arr));
            else:
                return(__cast__(irft(cor, shift = True, shift_before = False, norm = None, axes =axes, real_axis='GLOBAL'), old_arr));
    else:
        raise ValueError('Images have different dimensions')
        return(M1)


def correl(M1,M2,  axes = None,phase_only = False):
    '''
        Correlator for images
        
         If the images have different sizes, zeros will be padded at the rims
         M1,M2: Images
         full_fft: apply full fft? or is real enough
         axes:   along which axes
         phase only: Phase correlation
    '''
    return(__correllator__(M1,M2, axes = axes, mode = 'correlation', phase_only = phase_only))

def convolve(M1,M2, full_fft = False, axes = None,phase_only = False, norm2nd=False):  
    '''
        Convolve two images for images
        
         If the images have different sizes, zeros will be padded at the rims
         M1,M2: Images
         full_fft: apply full fft? or is real enough
         axes:   along which axes
         phase only: Phase correlation (default: False)
         norm2nd : normalizes the second argument to one (keeps the mean of the first argument after convolution), (default: False)
    '''
    return(__correllator__(M1,M2, axes = axes, mode = 'convolution', phase_only = phase_only, norm2nd=norm2nd))

    
def rot_2D_im(M, angle):
    '''
        Rotates 2D image 
        Maintains the size of the image and fills gaps due to rotation up with zeros
        Parts of the image that will be outside of the boarders (aka the edges of the old image) will be clipped!
    '''
    import scipy.ndimage as image;
    return(image.interpolation.rotate(M, angle, reshape = False));

def centroid(im):
    '''
        returns tupel with the center of mass of the image
    '''
    from scipy.ndimage.measurements import center_of_mass as cm;
    return(cm(np.asarray(im)))

def extract_c(im, center = None, roi = (100,100), axes_center  = None, axes_roi = None, extend ='DEFAULT'):
    '''
        Extract a roi around a center coordinate
        if center is none (default), the roi will be in the center of the image
        otherwise center should be a tuple
        
        roi -> edge length of the cube to be extracted -> can be tuple or list or number -> if its a number it will be squareshaped for each direction
        
        axex_center: to which axes (tupel) do the center cooridnates refere
        axes_roi: to which axes do the roi cooridnates refere:
    '''
    if roi is None:
        return(im);
    else:
        if center == None:
            center = (im.shape[0]//2, im.shape[1]//2);
            
        if len(center) > np.ndim(im):
            print('Too much dimensions for center coordinates');
        else:
            import numbers
            if isinstance(roi, numbers.Number):
               roi = tuple(int(roi) for i in im.shape);
            if axes_center == None:
                axes_center = tuple(range(len(center)));
            if axes_roi == None:
                axes_roi = tuple(range(len(roi)))
            if len(axes_center) != len(center): 
                print('axes_center and center must have same length');
            else:
                roi_list = [];
                new_ax = [];
                for i in range(im.ndim):
                    if i in axes_center:
                        if i in axes_roi:
                            new_ax.append(i);
                            roi_list.append((center[axes_center.index(i)]-roi[axes_roi.index(i)]//2,center[axes_center.index(i)]+roi[axes_roi.index(i)]//2))
                return(extract(im, roi_list, tuple(new_ax), extend))


def line_cut(im,coord1 = 0, coord2 = None,thickness = 10):
    '''
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
    '''
    through_all = False;
    if coord2 == None:
        coord2 = (np.array(im.shape[:2][::-1])-1)/2.;
        through_all = True;

    if type(coord1) == tuple or type(coord1) == list:
        try:
            m = (coord2[1]-coord1[1])/(coord2[0]-coord1[0])
            alpha = np.arctan(m)*180/np.pi
        except ZeroDivisionError:
            alpha = 90.0; 
        
    else:
        alpha = coord1;
        coord1 = coord2;
        through_all = True
        
    def __get_line__(img):           
        from scipy.ndimage import rotate;
        im_rot = rotate(img,-alpha)         # rotate image so, that the line is parallel to the x axis
        org_center = (np.array(img.shape[:2][::-1])-1)/2.# old center
        rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.    #center of roted image
        org1 = coord1-org_center;
        org2 = coord2-org_center;
        a = np.deg2rad(-alpha);
        new1 = tuple(np.array([org1[0]*np.cos(a) - org1[1]*np.sin(a),org1[0]*np.sin(a) + org1[1]*np.cos(a) ])+rot_center);
        new2 = tuple(np.array([org2[0]*np.cos(a) - org2[1]*np.sin(a),org2[0]*np.sin(a) + org2[1]*np.cos(a) ])+rot_center);

# THIS IS FOR DEBUGGING TO SHOW THE ROTATED IMGE AND THE POINTS
#        v = view(im_rot)   
#        v.set_mark(new1);
#        v.set_mark(new2);
#        
        if through_all:
            lower_thresh = 0;
            upper_thresh = im_rot.shape[0];
        else:
            lower_thresh = np.trunc(np.min([new1[0], new2[0]])).astype(int);
            upper_thresh = np.ceil(np.max([new1[0], new2[0]])).astype(int);
        
        if thickness == 1:
             # this is in case that the coordinate to we want to extract is not-integer -> than we take tha neighboring elements and weight them!
            gewicht = 1-(new1[1]-np.trunc(new1[1]));  
            line = gewicht*im_rot[lower_thresh:upper_thresh, np.floor(new1[1]).astype(int)]+(1-gewicht)*im_rot[lower_thresh:upper_thresh, np.ceil(new1[1]).astype(int)];    
        else:
            # if we average over several we don't do any weighting! 
            line = np.mean(im_rot[lower_thresh:upper_thresh, np.round(new1[1]-thickness/2).astype(int):np.round(new1[1]+thickness/2).astype(int)], axis =1);
        return(line)            
        
    
    if im.ndim == 2:
        return(__get_line__(im));
    else:
        im = np.reshape(im, (im.shape[0], im.shape[1], np.prod(np.asarray(im.shape)[2:])));
        return([__get_line__(im[:,:,i]) for i in range(im.shape[2])])      


def centered_extract(img,ROIsize,centerpos=None,PadValue=0.0):
    '''
        extracts a part in an n-dimensional array based on stating the center and ROI size
        
        ROIsize: size of the ROI to extract. Will automatically be limited by the array sizes when applied. If ROIsize==None the original size is used
        centerpos: center of the ROI in source image to exatract
        PadValue (default=0) : Value to assign to the padded area. If PadValue==None, no padding is performed and the non-existing regions are pruned.
       
        Example:
            nip.centered_extract(nip.readim(),[799,799],[-1,-1],100) # extracts the right bottom quarter of the image

   '''
    mysize=img.shape
    if ROIsize==None:
        ROIsize=mysize
    if centerpos==None:
        centerpos=[sd//2 for sd in mysize]
    else:
        centerpos=nip.coordsToPos(centerpos,mysize)
#    print(nip.ROIcoords(centerpos,ROIsize,img.ndim))
    res=img[nip.ROIcoords(centerpos,ROIsize,img.ndim)]
    if PadValue is None:
        return res
    else: # perform padding
        pads=[(max(0,ROIsize[d]//2-centerpos[d]),max(0,centerpos[d]+ROIsize[d]-mysize[d]-ROIsize[d]//2)) for d in range(img.ndim)]
#        print(pads)
        res=nip.image(np.pad(res,tuple(pads),'constant',constant_values=PadValue)) #,PadValue
        return res
    
def extract(im, roi = [(0,10),(0,10)], axes = None, extend ='DEFAULT'):
    '''
        returns sub image
        
            im is the image
            roi is the region of interests  -> this must be a list of tupels
                                                     each tupel gives the minium and the maximum to clip for the given axis!
                                                     
                                                     
            axes is a list of axis, if not given the first axes will be used, if given its length has to have the length of the roi list
            
            extend: True or False, if true, the image will be padded with zeros if the roi exceeds boarders
            
            example:
                extract(im, roi = [(20,30), (100,300)], axes = (0,2));
                   will clip the image im between 20 and 30 in the x-axis and 100 and 300 in the z-axis
            
    '''
    old_arr = im;
    if extend ==  'DEFAULT':
        extend = __DEFAULTS__['EXTRACT_EXTEND'];
   
    if type(roi) != list and type(roi) != tuple:
        raise ValueError('roi must be a list or a tuple');
    import numbers;
    if isinstance(roi[0], numbers.Number):
        if len(roi)>1:
            roi = [roi];
            print('Warning: Roi is list of numbers -> taking first two as roi for axis 0');
        else:
            raise ValueError('Lacking roi information')
    if roi is None:
        return(__cast__(im, old_arr));
    else:
        padding = [[0,0] for i in range(im.ndim)]
        if (axes == None):
            if len(roi) > np.ndim(im):
                print('Error: to much Rois in list or image dimension too small');
            else:
                for r in enumerate(roi):
                    im = im.swapaxes(0,r[0]);
                    if r[1][0] < 0:
                        low = 0;
                        padding[r[0]][0] = np.abs(r[1][0]);
                    else: low = r[1][0];
                    if r[1][1] > im.shape[0]:
                        up = im.shape[0];
                        padding[r[0]][1] = np.abs(r[1][1]-im.shape[0]);
                    else:
                        up = r[1][1]
                    im = im[low:up]; 
                    im = im.swapaxes(0,r[0]);
        else:
            if (type(axes) != tuple) or (max(axes) > np.ndim(im)):
                raise ValueError('Either axes is not a tupel or has too much elements');
            else:
                for r in enumerate(roi):
                    im = im.swapaxes(0,axes[r[0]]);
                    if r[1][0] < 0:
                        low = 0;
                        padding[r[0]][0] = np.abs(r[1][0]);
                    else: low = r[1][0];
                    if r[1][1] > im.shape[0]:
                        up = im.shape[0];
                        padding[r[0]][1] = np.abs(r[1][1]-im.shape[0]);
                    else:
                        up = r[1][1]
                    im = im[low:up];
                    im = im.swapaxes(0,axes[r[0]]);
        if extend == False:
            return(__cast__(im, old_arr))
        else:
            padding = tuple([tuple(i) for i in padding]);
            return(__cast__(np.lib.pad(im,padding,'constant'), old_arr))

def supersample(im, factor = 2, axis = (0,1), full_fft = False):
    '''
        supersample (resample, rescale, zoom) an imgage by a given factor (default = 2) along the given axis(default is 0 and 1)

        im: image to supersample         
        factor: a scalar value (for all axes) or a list of values for individual scales for each axes (...,Z,Y,X)
                                
    '''
    orig = im;
    if type(axis) == int:
        axis = tuple([axis]);
    
    from .transformations import ft , ift, rft, irft;
    if im.dtype == np.complexfloating or full_fft:
        FT = ft(im, shift = True,shift_before=False, axes = axis, ret = 'complex');
        real_ax = -1;
        r = [[float(im.real.max()), float(im.real.min())],[float(im.imag.max()), float(im.imag.min())]]
    else:
        FT = rft(im, shift = True,shift_before=False, axes = axis, ret = 'complex',real_return = None);
        from .transformations import __REAL_AXIS__;
        real_ax = __REAL_AXIS__;
        r = [float(im.max()),float(im.min())];
    
    padding =[];
    
    if factor >1:
        padding =[];
        for i in range(np.ndim(im)):
            if i in axis:
                if i == real_ax:
                    
                    padding.append((0,int((FT.shape[i]-1)*(factor-1))));
                else:
                    padding.append((np.floor(FT.shape[i]/2*(factor-1)).astype(int),np.ceil(FT.shape[i]/2*(factor-1)).astype(int)));
            else:
                padding.append((0,0));
        FT = np.lib.pad(FT,tuple(padding),'constant');
    elif factor < 1:
        roi = [];
        center = [];
        for i in range(np.ndim(im)):
            if i in axis:
                if i == real_ax:
                    roi += [int(2*np.ceil(FT.shape[i]*factor))];
                    center += [0];
                else:
                    roi += [int(np.ceil(FT.shape[i]*factor))];
                    center += [int(np.ceil(FT.shape[i]//2))];
        FT = extract_c(FT, center = center, roi = roi, axes_roi = axis, extend = False);
    
    if im.dtype == np.complexfloating:
        im = ift(FT, shift = False,shift_before=True, axes = axis,s= None, norm = None, ret = 'complex');
    elif full_fft:
        im = ift(FT, shift = False,shift_before=True, axes = axis,s= None, norm = None, ret = 'real');
        r = r[0];
    else:
        im = irft(FT, shift = False,shift_before=True, axes = axis,s = None, norm = None, ret = 'complex', real_axis = real_ax);
    if type(orig) == image:
        im = image(im)
        im.__array_finalize__(orig);
        im.pixelsize = [i*factor for i in orig.pixelsize];
        from .util import normalize;        
        im = normalize(im, 3, r);
        return(__cast__(im.astype(im.dtype),orig));
    else:
        from .util import normalize;        
        return(__cast__(normalize(im.astype(im.dtype),3,r), orig));
            
