#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:58:25 2017

@author: root

all kind of utils;

"""


import numpy as np;
import time;
import functools;

class timer():
    '''
        A simple class for measuring times:
            
            initialize object:
                t = nip.timer(units = 's');
            
                units can be "s", "ms", "us"
            
            append new time point:
                t.add(comm)
                    comm is optionally a string with a commetn
            
            get timepoints:
                t.get(mode = 'mutual', comm= False)
                    get timepoints with respect to the time when timer was initialized:
                        
                mode:  'abs' -> absolute (e.g. distance to time when timer was initialized)
                       'mut' -> mutually (e.g. pairwise distance between times)
                comm:   show comments
    '''
    
    def __init__(self, units = 's'):
        self.times = np.zeros(1);
        self.comments = [];
        from time import time;
        self.t_offset = time();
        self.mut_times = np.ndarray((0))
        self.units = units;
        if self.units == 's':
            self.factor = 1;
        elif self.units == 'ms':
            self.factor = 1000;
        elif self.units == 'us':
            self.factor = 1000000;
    def add(self, comm = None):
        from time import time;
        self.times = np.append(self.times,time()-self.t_offset);
        self.comments += [comm];
        if self.times.size >1:
            self.mut_times = pairwise_arith(self.times, 'diff');
    
    def get(self, mode = 'mut', comm = True):
        if mode == 'mut':
            print();
            print('Mutual timings ['+self.units+']');
            print('==================');
            print();
            if comm == False:
                for i in range(self.mut_times.size):
                    print('dT%d: %f3 \n' % (i, self.mut_times[i]*self.factor));
            else:
                for i in range(self.mut_times.size):
                    print('dT%d: %f3 %s \n' % (i, self.mut_times[i]*self.factor, self.comments[i]));    
        if mode == 'abs':
            print();
            print('Absolute timings ['+self.units+']');
            print('==================');
            print();
            if comm == False:
                for i in range(self.times.size-1):
                    print('T%d: %f3 \n' % (i, self.times[i+1]*self.factor));
            else:
                for i in range(self.times.size-1):
                    print('T%d: %f3 %s \n' % (i, self.times[i+1]*self.factor, self.comments[i]));


def ftimer(func):
        '''
            Print the runtime of a function using a decorator
        '''
        @functools.wraps(func)
        def wrap_tmr(*args, **kwargs):
            start = time.perf_counter();
            value = func(*args, **kwargs);
            end = time.perf_counter();
            rt = end-start;
            print(f'Runtime of {func.__name__!r} is {rt:.3f} secs ')
        return(wrap_tmr);

def inrange(arr, ran):
    '''
        check if the elements of the array (or number) arr are inside the range of ran (tuple or list)
    '''
    if ran[0]>ran[1]:
        ran = ran[::-1];
    return((arr>ran[0])&(arr<ran[1]))

def zernike(r,m,n, radial = False):
    
    '''
        compute the zerincke polynomial Z^m_n(r, phi)
        
        r: normalized radial coordinate
        n,m  orders of the polynomial 
        radial: if true, only the radial component of the zernike will be returned
    '''
    if n<0 or np.abs(m)>n:
        raise ValueError('n must not be negative and |m| <= n')
    if radial == False:
        from .coordinates import phiphi
        if m>=0:
            fact = np.cos(np.abs(m)*phiphi(r, angle_range = 2));
        else:
            fact = np.sin(np.abs(m)*phiphi(r, angle_range = 2));
    else:
        fact = 1;
    
    m = np.abs(m);
    n = np.abs(n);
    if np.mod((n-m),2) ==1:
        return(np.zeros(r.shape));
    else:
        from scipy.special import factorial;
        k = np.arange(0, (n-m)//2+1,1)
        zer_coeff = (-1)**k*factorial(n-k)/(factorial(k)*factorial((n+m)/2-k)*factorial((n-m)/2-k))
        zer = np.zeros(r.shape);
        for c, k_el in zip(zer_coeff, k):       # on purpose used a for loop here, as it is indeed faster
            zer += c*r**(n-2*k_el);
    return(zer*fact)
        
def scale_16bit(M):
    '''
        scale an image between 0 and 65536 and cast it into 'int'
    '''
    M = (M/np.max(M)*65536)
    return M.astype(int);

def parse_string_for_int(s, key):
    '''
        search a string s for int numbers after the key value
    '''
    pos = s.find(key);
    i = pos+1;
    end = False;
    val =''
    while (i<=len(s)) and end == False:
        try:
            val += str(int(s[i]));
        except:
            end = True;
        
        i+=1;
    if val != '':
        val = int(val);
    return(val)


#
#def __adjust_imtype__(im, ft_axes, orig):
#    '''
#        Check the data dtype and potentially change pixelsizes
#    '''
#    from .image import image;
#    if type(orig) == image:
#        im = im.view(image);            # note: view casting -> this is not the viewer!
#        if type(orig.name) is str:
#            im.name = 'FT of '+orig.name;
#            im.
#        return(im);
#    else:
#        return(im);
#      

def scale_log(M, c =1):
    '''
        Scale an image logarithmic as:
            
            c*log(1+|M|)
    '''
    return(c*np.log(1+np.abs(M.astype(np.float64))));

def expand(vector, size, transpose = False):
    '''
        expands a vector in the second dimension for unit size
        Result can be transposed
        
        TODO: FOR NEXT DIMENSION!!! -> Hence it won't only apply for 1D vectors
    '''
    if transpose:
        return(np.transpose(np.reshape(np.repeat(vector,size,axis=0),(np.size(vector),size) )));
    else:
        return(np.reshape(np.repeat(vector,size,axis=0),(np.size(vector),size) ))

def castdim(img,ndims,wanteddim):
    '''
        expands a 1D image to the necessary number of dimension casting the dimension to a wanted one
        ----------
        img: input image to expand
        ndims: number of dimensions to expand to
        wanteddim: number that the one-D axis should end up in
    '''
    mysize=img.shape
    if wanteddim<0:
        wanteddim=ndims+wanteddim
    if wanteddim+img.ndim > ndims:
        raise ValueError("castdim: ndims is smaller than requested total size including the object to place.")
    newshape=wanteddim*(1,)+mysize+(ndims-wanteddim-img.ndim)*(1,)
    return np.reshape(img,newshape)

def expanddim(img,ndims,trailing=False):
    '''
        expands an nd image to the necessary number of dimension by inserting leading dimensions
        ----------
        img: input image to expand
        ndims: number of dimensions to expand to
        trailing (default:False) : append trailing dimensions rather than dimensions at the front of the size vector
    '''
    if trailing:
        return np.reshape(img,img.shape+(ndims-len(img.shape))*(1,))   # RH 2.2.19
    else:
        return np.reshape(img,(ndims-len(img.shape))*(1,)+img.shape)   # RH 2.2.19

def dimVec(d,mysize,ndims):
    '''
        creates a vector of ndims dimensions with all entries equalt to one except the one at d which is mysize
        ----------
        d: dimension to specify entry for
        mysize : entry for res[d]        
        ndims: length of the result vector
    '''
    res=ndims*[1]
    res[d]=mysize
    return tuple(res)

def slicecoords(mydim,ndim,start):
    '''
        constructs a coordinate vector reducing dimension mydim, such that numpy can extract the slice via an array access 
    '''
    if start!=-1:
        end=start+1
    else:
        end=None
    return tuple((mydim)*[slice(None)]+[slice(start,end)]+(ndim-mydim-1)*[slice(None)])

def subslice(img,mydim,start):
    '''
        extracts an N-1 dimensional subslice at dimension dim and position start        
        It keeps empty slices as singleton dimensions
    '''
        
    return img[slicecoords(mydim,img.ndim,start)]

def subsliceAsg(img,mydim,start,val):
    '''
        assigns val to an N-1 dimensional subslice at dimension dim and position start
        -----
        img : input image to assign to
        mydim : dimension into which the subslice is chosen
        start : offset along this dimension
        val : value(s) to assign into the array
        It keeps empty slices as singleton dimensions
    '''
    img[slicecoords(mydim,img.ndim,start)]=val
    return img

def MidValAsg(img,val):
    '''
        assigns val to the middle position of an image (where ft has its zero frequency)
        -----
        img : input image to assign to
        mydim : dimension into which the subslice is chosen
        start : offset along this dimension
    '''
    img[img.mid()]=val
    return img

def abssqr(img):
    return img*np.conjugate(img)

def pairwise_arith(a, mode = 'sum'):
    '''
        returns the pairwise arithmetics of an 1D array
        mode:
            'sum', 'prod', 'diff', 'div'
    '''
    if type(a) != np.ndarray:
        raise TypeError('a has to be 1D ndarray');
    elif a.ndim > 1:
        raise TypeError('Only 1D arrays');
    else:
        b =np.append(a,0);
        a = np.insert(a,0,0);
    if mode == 'sum':
        return((a+b)[1:-1]);
    elif mode == 'prod':
        return((a*b)[1:-1]);
    elif mode == 'diff':
        return((b-a)[1:-1]);
    elif mode == 'div':
        return((b/a)[1:-1]);
    else:
        raise ValueError('Wrong mode: give  "sum", "prod", "diff", "div"');    

def make_damp_ramp(length, function):
    '''
        creates a damp ramp:
            length - length of the damp ramp in pixes
            function - function of the damp ramp
                        Generally implemented in nip.functions
                        Make sure that first element is x and second element is characteristica lengths of the function
    '''
    x = np.arange(0, length,1);
    return(function(x, length-1));



def normalize(M, mode, r = None):
    '''
        Normalize an image between 0 and 1
        
        mode
            0:      between 0 and 1 (amplitude normalization)
            1:      divided by maximum 
            2:      divided by sum (Normalized to Photon number)
            3:      in range given by r (list or tuple) -> for complex values: give one list of 2 tuples, containing the range for real and imaginary part
            
    '''
    
    if M.dtype == np.complexfloating:
        if mode == 0:
            return((M.real-M.real.min())/(M.real.max()-M.real.min())+1j*((M.imag-M.imag.min())/(M.imag.max()-M.imag.min()))); 
        if mode == 1:
            return(M.real/M.real.max()+1j*M.imag/M.imag.max());
        if mode == 2:
            return(M.real/np.sum(M.real)+1j*M.imag/np.sum(M.imag));
        if mode == 3:
            if type(r) != list and type(r) != tuple:
                if type(r[0]) != list and type(r[0]) != tuple:
                    raise ValueError('Give proper range r [tuple or list of tuple or list of ranges]')
            else:
                return((M.real-M.real.min())/(M.real.max()-M.real.min())*(max(r[0])-min(r[0]))+min(r[0])+1j *(M.real-M.imag.min())/(M.imag.max()-M.imag.min())*(max(r[1])-min(r[1]))+min(r[1]));
    else:
        if mode == 0:
            return((M-M.min())/(M.max()-M.min()));
        if mode == 1:
            return(M/np.max(M));
        if mode == 2:
            return(M/np.sum(M));
        if mode == 3:
            if type(r) != list and type(r) != tuple:
                raise ValueError('Give proper range r [tuple or list]')
            else:
                return((M-M.min())/(M.max()-M.min())*(max(r)-min(r))+min(r));

def extract_coordinate(M,c):
    '''
        Returns the coordinate of position c in an n-dimensional matrix as tupel
        
        Example:
            M is a matrix of shape (3,4,5)  (numpy nd array)
            c= 31 is the postion in the matrix (e.g. given by the argmax function of numpy)
            
            This script returns (1,2,1).
            
 
    '''
    coord =[];
    def get_c(M,c,l):
        if M.ndim == 1:
            l.append(c);
        else:
            coo = c//np.prod(M.shape[1:]);
            l.append(coo);
            get_c(M[coo], c-np.prod(M.shape[1:])*coo,l)
        return(l)
    get_c(M,c,coord);
    return(tuple(coord));


def max_coord(M):
    '''
        Returns tuple of coordinates of the maximum of M
        
    '''
    return(extract_coordinate(M,np.argmax(M)))


def min_coord(M):
    '''
        Returns tuple of coordinates: (line, colum) from the minimum
        Rigth now only for 2 Dimension
    '''
    return(extract_coordinate(M,np.argmin(M)))

def roi_from_poslist(poslist):
    '''
        get the roi from a given positon list
        the roi is [(min_dim0, max_dim0),(min_dim1, max_dim1),...]
    '''
    if type(poslist) == list or type(poslist) == list:
        poslist = np.asarray(poslist);
        if poslist.ndim != 2:
            raise TypeError('poslist has to be list or tuple of position tuples ')
        else:
            mins = np.round(np.min(poslist,0)).astype(int);
            maxs = np.round(np.max(poslist,0)).astype(int);
            return([(min(mi, ma), max(mi,ma)) for mi, ma in zip(mins, maxs)]);
    else:
        raise TypeError('poslist has to be list or tuple of position tuples ')

def get_max(M,region = [-1,-1,-1,-1]):
    '''
    Get coordinates maximum value in a certain region
    
    region = [x,y, width_x, width_y]
    
    !!!only works for 2D right now!!!
    '''
    if region[0]!= -1:
        M = M[region[1]-region[3]//2:region[1]+region[3]//2,region[0]-region[2]//2:region[0]+region[2]//2]
    MAX = max_coord(M);
    return(region[0]-region[2]//2+MAX[1],region[1]-region[3]//2+MAX[0])

def adjust_lists(l1,l2, el =0):
    '''
        adjust the length of two lists by appending zeros to the shorter ones
    '''
    if get_type(l1)[0] != 'list' and get_type(l2)[0] != 'list':
        raise TypeError('Wrong input type');
    l1 = list(l1);
    l2 = list(l2);
    
    if len(l1) == len(l2):
        return(l1,l2);
    elif len(l1) > len(l2):
        l2+=[el for i in range(len(l1)-len(l2))];
        return(l1,l2)
    elif len(l1) < len(l2):
        l1+=[el for i in range(len(l2)-len(l1))];
        return(l1,l2)

def get_type(var):
    '''

        TODO: DEPRICATE
        get the type of the variable:
            
            parameter         var -> some object
            
            returns list of strings: first is topic, second is explanation :
                                
                        ['number', 'integer' or 'float' or 'complex']
                        ['string']
                        ['array', 'nparray, image, otf2d, psf2d, otf3d, psf3 etc']
                        ['list', 'list' or 'tuple']
    '''
    import numbers;
    if var is None:
        return(['none']);
    
    elif isinstance(var, numbers.Number):
        if isinstance(var, numbers.Integral):
            return(['number', 'integer']);
        elif isinstance(var, numbers.Real):
            return(['number', 'float']);
        elif isinstance(var, numbers.Complex):
            return(['number', 'complex']);
        else:
            return(['number', 'unknown']);
    elif isinstance(var, str):
        return(['string']);
    elif isinstance(var, bool):
        return(['boolean']);
    elif type(var) == list:
        return(['list','list']);
    elif type(var) == tuple:
        return(['list','tuple']);
    else:
        if type(var) != type:
            t = type(var);
    
        def __recover_parents__(v, classes = []):
            #if v.__base__ is not None:
            if v.__base__ is not None:
                
                for b in v.__bases__:
                    classes.append(b.__name__);
                for c in v.__bases__:
                    return(__recover_parents__(c, classes));
            return(classes)    
        
        classes = [t.__name__]+__recover_parents__(t);
        if 'ndarray' in classes:
            if 'image' in classes:
                return(['array', 'image', classes[0]]);
            else:
                return(['array', classes[0]]);
        else:
            return(['unknown']+classes)
        
def get_min(M,region = [-1,-1,-1,-1]):
    '''
    Get coordinate of mininum value in a certain region
    
    region = [x,y, width_x, width_y] where x and y are the center coordinates and width_x and width_y are the edge lengths around that center
    
    !!!only works for 2D right now!!!
    '''
    if region[0]!= -1:
        M = M[region[1]-region[3]//2:region[1]+region[3]//2,region[0]-region[2]//2:region[0]+region[2]//2]
    MIN = min_coord(M);
    return(region[0]-region[2]//2+MIN[1],region[1]-region[3]//2+MIN[0])
    
    
def isnp(animg):
    return isinstance(animg,np.ndarray)

def ones(s,dtype=None,order='C'):
    from .image import image
    if isnp(s):
        s=s.shape
    return image(np.ones(s,dtype,order))

def zeros(s,dtype=None,order='C'):
    from .image import image
    if isnp(s):
        s=s.shape
    return image(np.zeros(s,dtype,order))


def __cast__(arr, orig_arr=None):
    '''
        this function should cast the array returned by functions correctly dendent on what is the desired format

        input:
                arr: the array to return
                orig_arr: original array -> from here the image features might be deduced!
                            -> can also be some list etc. -> than it is treated like nparray

        The way how to cast is set up in the config __DEFAULTS__['ARRAY_RETURN_TYPE']
        Can be
            'image'   returns image type -> properties are adapted from original image (if given) or as stated in Defaults , also: If input is a OTF or equal it returns a OTF
            'IMAGE' like 'image' but OTFs etc will be casted to image!
            'ndarray' returns numpy array
            'asInput' same type as input image

    '''
    from .image import image;
    from .config import __DEFAULTS__;
    # from .util import get_type

    if not isinstance(orig_arr, np.ndarray):
        orig_arr = np.zeros(0)
    if __DEFAULTS__['ARRAY_RETURN_TYPE'] == 'IMAGE':
        if type(arr) is not image:
            return (arr.view(type=image));
        else:
            return (arr);

    elif __DEFAULTS__['ARRAY_RETURN_TYPE'] == 'image':
        if isinstance(arr, image):
            return (arr);
        else:
            arr = image(arr);  # Cast as image
            if isinstance(orig_arr, image):
                arr.__array_finalize__(orig_arr);  # if original array was image -> copy members!
            return (arr.view(type=image));
    elif __DEFAULTS__['ARRAY_RETURN_TYPE'] == 'ndarray':
        if type(arr) is not np.ndarray:
            return (np.asarray(arr));
        else:
            return (arr);
    elif __DEFAULTS__['ARRAY_RETURN_TYPE'] == 'asInput':
        if type(arr) == type(orig_arr):
            return (arr);
        else:
            arr = arr.view(type=type(
                orig_arr));  # here it important that we checked if orig_arr was at least array type before -> if not -> make it array
            if isinstance(orig_arr, image):
                arr.__array_finalize__(orig_arr);  # if original array was image -> copy members!
            return (arr)
    else:
        raise ValueError(
            'ARRAY_RETURN_TYPE msut be "asInput", "image", "IMAGE", or "ndarray". Set it correctly in __DEFAULTS__');
#