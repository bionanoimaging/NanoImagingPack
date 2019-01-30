import numpy as np;
from .util import get_type;

def ramp(mysize=(256,256), ramp_dim=0, corner='center', freq=None):
    '''
    creates a ramp in the given direction direction 
    standart size is 256 X 256
    mode: 
        center: 0 is at cetner
                if x is even, it has one more value in the positve:
                    e.g. size_x = 100 -> goes from -50 to 49
                         size_x = 101 -> goes from -50 to 50
        negative : goes from negative size_x to 0
        positive : goes from 0 size_x to positive
        freq : if "freq" is given, the Fourier-space frequency scale (roughly -0.5 to 0.5) is used.
        int number: is the index where the center is!
    '''
    from .image import image;
    if type(mysize)== np.ndarray or type(mysize) == image:
        mysize = mysize.shape;
    
    res = np.ones(mysize);
   
    if corner == 'negative':
        miniramp = np.arange(-mysize[ramp_dim]+1,1,1);
    elif corner == 'positive' or corner == 'corner':
        miniramp = np.arange(0,mysize[ramp_dim],1);
    elif corner == 'center':
        miniramp = np.arange(-mysize[ramp_dim]//2+np.mod(mysize[ramp_dim],2),mysize[ramp_dim]//2+np.mod(mysize[ramp_dim],2),1);
    elif (type(corner) == int or type(corner) == float):
        miniramp = np.arange(0,mysize[ramp_dim],1)-corner;
    else:
        try: 
            if np.issubdtype(corner.dtype, np.number):
                miniramp = np.arange(0,mysize[ramp_dim],1)-corner;
        except AttributeError:
            pass;
    minisize = list(np.ones(len(mysize)).astype(int));
    minisize[ramp_dim] = mysize[ramp_dim];
    #np.seterr(divide ='ignore');
    miniramp = np.reshape(miniramp,minisize)
    if freq=="freq":
        res*=(miniramp/(mysize[ramp_dim]//2)/2.0)
    elif freq=="radfreq":
        res*=(miniramp/(mysize[ramp_dim]//2/2.0)*2*np.pi)
    else:
        res*=miniramp
    return(image(res));  # RH casted to image

def xx(mysize = (256,256), mode = 'center', freq=None):
    '''
    creates a ramp in x direction 
    standart size is 256 X 256
    mode: 
        center: 0 is at cetner
                if x is even, it has one more value in the positve:
                    e.g. size_x = 100 -> goes from -50 to 49
        negative : goes from negative size_x to 0
        positvie : goes from 0 size_x to positive
    '''
    return(ramp(mysize,0,mode,freq))

def yy(mysize = (256,256), mode = 'center', freq=None):
    '''
    creates a ramp in y direction 
    standart size is 256 X 256
    mode: 
        center: 0 is at cetner
                if y is even, it has one more value in the positve:
                    e.g. size_y = 100 -> goes from -50 to 49
        negative : goes from negative size_y to 0
        positvie : goes from 0 size_y to positive
    '''
    return(ramp(mysize,1,mode,freq))
 
def zz(mysize = (256,256), mode = 'center', freq=None):
    '''
    creates a ramp in z direction 
    standart size is 256 X 256
    mode: 
        center: 0 is at cetner
                if y is even, it has one more value in the positve:
                    e.g. size_y = 100 -> goes from -50 to 49
        negative : goes from negative size_y to 0
        positvie : goes from 0 size_y to positive
    '''
    return(ramp(mysize,2,mode))

def rr2(mysize=(256,256), offset = (0,0),scale = None, freq=None):
    '''
    creates a square of a ramp in r direction 
    standart size is 256 X 256
    mode is always "center"
    offset -> x/y offset in pixels (number, list or tuple)
    scale is tuple, list, none or number of axis scale
    '''
    import numbers;
    if offset is None:
        scale = [0,0];
    elif isinstance(offset, numbers.Number):
        offset = [offset, 0];
    elif type(offset)  == list or type(offset) == tuple:
        offset = list(offset[0:2]);
    else:
        raise TypeError('Wrong data type for offset -> must be Number, list, tuple or none')
        
    if scale is None:
        scale = [1,1];
    elif isinstance(scale, numbers.Integral):
        scale = [scale, scale];
    elif type(scale)  == list or type(scale) == tuple:
        scale = list(scale[0:2]);
    else:
        raise TypeError('Wrong data type for scale -> must be integer, list, tuple or none')
    res=((ramp(mysize,0,'center',freq)-offset[0])*scale[0])**2
    for d in range(1,len(mysize)):
        res+=((ramp(mysize,d,'center',freq)-offset[d])*scale[d])**2
    return res

def rr(mysize=(256,256), offset = (0,0),scale = None, freq=None):
    '''
    creates a ramp in r direction 
    standart size is 256 X 256
    mode is always "center"
    offset -> x/y offset in pixels (number, list or tuple)
    scale is tuple, list, none or number of axis scale
    '''
    return np.sqrt(rr2(mysize,offset,scale,freq))
   
def phiphi(mysize=(256,256), offset = 0, angle_range = 1):
    '''
    creates a ramp in phi direction 
    standart size is 256 X 256
    mode is always center
    offset: angle offset in rad
    angle_range:
            1:   0 - pi for positive y, 0 - -pi for negative y
            2:   0 - 2pi for around
        
    '''
    np.seterr(divide ='ignore', invalid = 'ignore');
    x = ramp(mysize,0,'center');
    y = ramp(mysize,1,'center');
    #phi = np.arctan(y/x)+(x<0)*np.pi*((y>0)*2-1);
    if angle_range == 1:
        phi = np.mod((np.arctan(y/x)+(x<0)*np.pi*((y>0)*2-1)+offset)+np.pi, 2*np.pi) -np.pi;
    elif angle_range == 2:
        phi = np.mod((np.arctan(y/x)+(x<0)*np.pi*((y>0)*2-1)+offset), 2*np.pi);
    phi[phi.shape[0]//2,phi.shape[1]//2]=0;
    np.seterr(divide='warn', invalid = 'warn');
    return(phi)

def VolumeList(MyShape = (256,256), MyCenter = 'center', MyStretch = 1, polar_axes = None, return_axes = 'all'):
        
    '''
        Returns a list of coordinate ramps
        
        !!! DIFFERENCE COMPARED TO XX, YY, ZZ, ramp etc: this function returns a list of float arrays!!!
        !!! THE PURPOSE IS FOR EASILY GENERATE VOLUME SHAPES !!!        
        
        MyShape:   The shape of the ramps to return. This has to be a tuple or a list, or nd_array. E.g. 3D volume: MyShape = (256,256,256);
        
        MyCenter:  Where is the center coordinate for a respective axis:
                    tuple or list with coordinates for the respective axis, can contain string 'center', 'positive' or 'negative' than the coordinate zero will be placed like defined in the "ramp" function
                    int: all axis will have the same zero
                    string: 'center', 'positive', 'negative' all axis same center, like defined in 'ramp' function
        
        MyStretch: Stretching factor of the different axes
                    E.g. for a 3D Volume with MyShape = (256,256,256) and MyCenter= 'Center':
                          With MyStretch = (1,2,0.5): first dimension goes from -127 to 128, second from -254 to 256 and third from -63.5 to 64
        
        polar_axes:  Which axes should be in polar coordinates?
                        -> None - only Cartesian axes
                        -> Tuple or list:  tuple or list, including the axes which should be polar
                        -> 'all', all axes will be in polar coordinates
                        
                                          
                        -> either None (default), or more than two, becasue one polar axes is senseless!
                        
                        -> First polar axes is r, second is phi, third is theta_1, every next one is higher theta
             
                Note: Using this option wiht Mystrech makes it possible to easily create an n-dimensional hyperellipsoid!
                        
                Relationship between polar and cartesian coordinates are computed as conventionally known:
                       
                        E.g. 4 axes:
                            
                            x1 = r * cos(phi) * sin(theta_1) * sin(theta_2)   (= x -Axes)
                            x2 = r * sin(phi) * sin(theta_1) * sin(theta_2)   (= y -Axes)
                            x3 = r            * cos(theta_1) * sin(theta_2)   (= z -Axes)
                            x4 = r                           * cos(theta_2)   
            
        return_axes: Which axes should be returned by the function?
                    -> None - empty list is returned
                    -> 'all' - all axes are returned (i.e. a list conating as much volumes as length of shape)
                    -> list or tuple: choose the volumes along axis to be returned (recommended, since less memory and possibly faster)
                    -> Can be int -> Only this axis is returend
                    -> IF ONLY ONE AXIS IS RETURNED the Returnvalue is a nd-array (shape as myshape). Otherwise it is a list of nd-arrays
                    
        Examples:
            
            Simple ramp in X-direction (same as xx(), but with shifted center and streched by a factor of 10):
                
                VolumeList((256,256), MyCenter = (30, 20), MyStretch = 10, polar_axes = None, return_axes = 0)
                
            
            Cylindrical coordinates with x -Direction being the cylinder axes, only the phi - axis returned    
                VolumeList((64,128,128), MyCenter = 'center', MyStretch = 1, polar_axes = (1,2), return_axes = 2)
                
            Cylindrical coordinates with x -Direction being the cylinder axes, only the whole volume returned
                VolumeList((64,128,128), MyCenter = 'center', MyStretch = 1, polar_axes = (1,2), return_axes = 'all')
    
     
            4 Dimensional hyper ellipsoid whit stretching factors of 0.1, 1, 10, 2;
                VolumeList((64,64,64,64), MyCenter = 'center', MyStretch = (0.1,1,10,2), polar_axes = 'all', return_axes = 'all')
     '''


    
    # Boiler plate for checking input and defining the axes
    def __check_para__(MyPara, Name, default):
        if (type(MyPara) == list or type(MyPara) == tuple):
            if len(MyPara) > len(MyShape):
                print('Waringing: Too many '+Name+'. Only taking the first '+str(len(MyShape)));
                MyPara = MyPara[:len(MyShape)];
            if len(MyPara) < len(MyShape):
                MyPara += tuple(np.ones(len(MyShape)-len(MyPara)).astype(int)*default);
        elif (type(MyPara) == int or type(MyPara) == float):
            MyPara = tuple(np.ones(len(MyShape)).astype(int)*MyPara);
        elif type(MyPara) == str:
            MyPara = tuple([MyPara]*len(MyShape));
        else:
            try: 
                if np.issubdtype(MyPara.dtype, np.number):
                    MyPara = tuple(np.ones(len(MyShape)).astype(int)*MyPara);
            except AttributeError:
                pass;
        return(MyPara);
    if type(MyShape)== np.ndarray:
        MyShape = MyShape.shape;
        
    MyCenter = __check_para__(MyCenter, 'center coordinates', 0);
    MyStretch = __check_para__(MyStretch, 'stretch coordinates', 1);
    
    if type(polar_axes) == tuple or type(polar_axes) == list:
        polar_axes = tuple(filter(lambda x: x<len(MyShape), polar_axes));
        cartesian_axes = tuple(filter(lambda x: x not in polar_axes, range(len(MyShape))));
        if len(polar_axes) == 1:
            print('Warning: Single polar axsis doesn\'t make any sense!');
            print('Either 2 (Cylindrical coordinates in these directions), 3 (Spherical coordinates in these direcection) or more (Hyperspherical coordiantes in this directions)')
            print('');
            print('I don\'t take any Polar coordinate');
            print('');
            print('HINT: Also make sure that given axes don\'t exceed volume dimension! (Counting starts at 0!)')
            polar_axes = None;

    if return_axes == 'all':
        return_axes = tuple(range(len(MyShape)));    
    if return_axes == None:
        return_axes = [];    
    if type(return_axes) == int:
        return_axes =tuple([return_axes]);
    try: 
        if np.issubdtype(return_axes.dtype, np.integer):
            return_axes =tuple([return_axes]);
    except AttributeError:
        pass;
    if polar_axes == None:
        cartesian_axes = tuple(range(len(MyShape)))
        polar_axes = ();
    if polar_axes == 'all':
        cartesian_axes = ();
        polar_axes = tuple(range(len(MyShape)));
    polar_shape = [MyShape[i] for i in polar_axes];    


    if set(return_axes).intersection(set(polar_axes)) != set([]):
        # only if I want to return at least one polar axes!
        
        # Creates set of ramps (as many as needed for the polar coordinates)
        # Later the cartesians have to be included here! -> note now: all the polar axes are the first ones!
        polar_ramps = [1/MyStretch[i]*ramp(MyShape, ramp_dim = i, corner = MyCenter[i]) for i in polar_axes];
        r = np.sqrt(np.sum(np.asarray(np.square(polar_ramps)), axis =0))
        if set(return_axes).intersection(set(polar_axes[1:])) != set([]):
            np.seterr(divide ='ignore');
            sin_factor = 1;
            if len(polar_shape) > 2:
                # Spherical coordinates
                l = [];
                for i in range(len(polar_shape)-1,1,-1): 
                    t = np.arccos(polar_ramps[i]/(r*sin_factor));
                    if polar_axes[i] in return_axes:
                        l.append(t);
                    sin_factor *= np.sin(t);
                l.reverse();
            # cylindrical only
            if polar_axes[1] in return_axes:
                phi = np.arccos(polar_ramps[0]/(r*sin_factor));
            np.seterr(divide='warn');
    
    ret_list = [];
    for i in return_axes:
        if i in cartesian_axes:
            ret_list.append(1/MyStretch[i]*ramp(MyShape, ramp_dim = i, corner = MyCenter[i]));
        if len(polar_axes) >0:
            if i == polar_axes[0]:
                ret_list.append(r);
        if len(polar_axes) >1:
            if i == polar_axes[1]:
                ret_list.append(phi)
        if len(polar_axes) >2:
            if i in polar_axes[2:]:
                ret_list.append(l[polar_axes[2:].index(i)])
    if len(ret_list) == 1:
        return(ret_list[0]);
    else:
        return(ret_list);


    
def freq_ramp(M, pxs = 50,  shift = True, real = False, ax =0):
    '''
        This function returns the frequency ramp along a given axes of the image M
        
        M is the image (or the function)
        pxs is the pixelsize in the given direction
            Note: the unit of the frequency ramp is 1/unit of pxs
            
            Example:
                if you have an image with a pixelsize of 80 nm, which is 100 pixel along the axis you wanna create the ramp
                you will get a ramp runnig up to 0.006125 1/nm in steps of 0.0001251  1/nm
        
        ax is the axes in which the ramp points
        shift: use true if the ft is shifted (default setup)
        real: use true if it is a real ft (default is false)
    '''
    from .image import image;
    if type(M) == tuple:
        mysize =M;
    elif type(M) == list:
        mysize =tuple(M);
    elif type(M) == np.ndarray:
        mysize = M.shape;
    elif type(M) == image:
        mysize = M.shape;
        pxs = M.pixelsize[ax];
    else:
        print('Wrong data type for M');
        return(M);
    res = np.ones(mysize);
    if shift:
        if real:
            miniramp =  np.fft.fftshift(np.fft.rfftfreq(mysize[ax],pxs));
        else:
            miniramp =  np.fft.fftshift(np.fft.fftfreq(mysize[ax],pxs));
    else:
        if real:
            miniramp =  np.fft.rfftfreq(mysize[ax],pxs);
        else:
            miniramp =  np.fft.fftfreq(mysize[ax],pxs);
    minisize = list(np.ones(len(mysize)).astype(int));
    minisize[ax] = mysize[ax];
    
    #np.seterr(divide ='ignore');
    miniramp = np.reshape(miniramp,minisize)
    res*=miniramp;
    return(res);

def px_freq_step(im = (256,256), pxs = 62.5):
    '''
        returns the frequency step in of one pixel in the fourier space for a given image as a list for the different coordinates
        The unit is 1/[unit pxs]
        
        pixelsize can be a number or a tupel:
            if number: pixelsize will be used for all dimensions
            if tupel: individual pixelsize for individual dimension, but: coord list and pixelsize list have to have the same dimension
        im: image or Tuple 
        pxs: pixelsize
    '''
    from .image import image;
    if type(im) == image:
        pxs = im.pixelsize;
        im = im.shape;
    else:
        if type(im) == np.ndarray:
            im = im.shape;
        
        if type(im) == list:
            im = tuple(im);
    
        import numbers;
        if isinstance(pxs, numbers.Number):
            pxs = [pxs for i in im];
        if type(pxs) == tuple or type(pxs) == np.ndarray:
            pxs = list(pxs);
        if type(pxs) != list:
            print('Wrong data type for pixelsize!!!');
            return;
    
    return([1/(p*s) for p,s in zip(pxs, im)]);
    
     

def get_freq_of_pixel(im = (256,256),coord = (0.,0.),pxs = 62.5,shift = True, real =False):
    '''
        get the frequency value of a pixel value
        
        pixelsize can be a number or a tupel:
            if number: pixelsize will be used for all dimensions
            if tupel: individual pixelsize for individual dimension, but: coord list and pixelsize list have to have the same dimension
        im: image or Tuple (if more than 2 Dimension, only the first 2 Dimensions will be used)
        coord: the pixel coordinate
        pxs: pixelsize
        shift: was there a shift of the Fourier transform
        real: real valued ft?
             -> c.f. help of freq_ramp
            
    '''
    from scipy.interpolate import interp1d;
    if type(im) == list:
        im = tuple(im);
    if type(im) == np.ndarray:
        im = im.shape;
    new_coord = [];
    if type(pxs) == float or type(pxs) == int: 
        pxs = tuple(pxs for i in range(len(coord)))
    elif type(pxs)  == tuple:
        if len(pxs)< len(coord):
            print('to view pixelsizes! C.F. help file')
        return(pxs);
    else:
        print('wrong pixel size fromat')
    for i in range(len(im)):
        if i < len(coord):
            f = freq_ramp(im, pxs[i], shift, real, i);
            f = np.swapaxes(f, len(im)-1, i);
            f = np.ravel(f, 'C');
            f = f[:im[i]];
            fx = np.arange(0,np.size(f),1)
            inter = interp1d(fx,f);
            new_coord.append(float(inter(coord[i])));
    return(tuple(new_coord));

def k_to_coords(im = (256,256) ,pxs = 62.5, parameters = (0,1)):
    '''
     Converts a tuple of (angle, k_length) to pixel coordinates (x,y)
     
         im: image or Tuple 
         pxs       - pixelsize (can be number or vector if the pixelsize is differnt for x and y)
         parameters:
                 tuple or list consisting of:
                     angle[Degree] -> direction like angle is commonly defined in complex plane
                     lenght of the k-vector -> in inverse units of pxs:
        
        
    '''
  
    if isinstance(parameters, list) or isinstance(parameters, tuple):
        import numbers;
        if isinstance(parameters[0], numbers.Real) and isinstance(parameters[1], numbers.Real):
            px_converter = px_freq_step(im, pxs);
            return((np.cos(parameters[0]*np.pi/180)*parameters[1]/px_converter[0]+im.shape[0]/2, np.sin(parameters[0]*np.pi/180)*parameters[1]/px_converter[1]+im.shape[1]/2));
        elif isinstance(parameters[0], list) or isinstance(parameters[0], tuple):
            ret_list = [];
            for p in parameters:
                if isinstance(p[0], numbers.Real) and isinstance(p[1], numbers.Real):
                    px_converter = px_freq_step(im, pxs);
                    ret_list += [(np.cos(p[0]*np.pi/180)*p[1]/px_converter[0]+im.shape[0]/2, np.sin(p[0]*np.pi/180)*p[1]/px_converter[1]+im.shape[1]/2)];
            return ret_list;
    else:
        print('Wrong paramter format!');
        return;
    
    


    
def bfp_coords(M, pxs, wavelength, focal_length,  shift = True, real = False, ax =0):
    '''
        This function returns the coordinate map of the fourier image in the back focal plane
        
        i.e.: If the object is placed in the front focal plane, the image in the back focal plane is the fourier transform of the image
              The each object coorinate (unit [lenght]) maps to an angle/frequency in the back focal plane (unit [1/length])
              This function transfroms the angles into length distances in the fourier plane 
              
         M is the image (or the function)
         pxs is the pixelsize in the given direction
         wavelength is the wavelenghts of the light

            Note: the units of pxs and wavelength have to be identical!!! (e.g. um)
        
        focal_length  - The focal length of the imageing lens
        
            Note: the result will have the same unit as the focal_length (e.g. cm)
            
            
        ax is the axes in which the ramp points
        shift: use true if the ft is shifted (default setup)
        real: use true if it is a real ft (default is false)    
        
        Example:
              if the image is a image of 20 um pixelsize of alternating black and white 
              (e.g. a grating pattern of grating period 40 um), the wavelengths is 488 nm 
              and the focal length is 500 mm, the peaks are located at +-6.1 mm around 
              the center. This distance is givne here!  
        
        
    '''
    return(freq_ramp(M, pxs, shift, real, ax)*wavelength*focal_length);