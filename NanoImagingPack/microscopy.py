#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:22:51 2017

@author: root

This Package should contain funcitons in order to create psfs and otfs both 2-Dimensional and 3-Dimensional
 
Also SIM stuff
"""

import numpy as np;
import NanoImagingPack as nip;
from .image import image;
from .util import get_type,abssqr;
from .transformations import rft,irft,ft2d,ift2d;
from .coordinates import rr, ramp
from .view5d import v5 # for debugging

class transfer():
    '''
        A class to create transfer functions
        
        Parameters:
            im                   The image for which to compute the NA -> if None -> An image of standard dimensions (according to NA, n, pixelsize is created)
            NA                   Numerical aperture
            n                    Refractive index
            wavelength           wavelength
            pixelsize            pixelsize (if not given taken from image or config file)
            
            resample             Resample transfere function at computation to speed up the computation time
            vectorial            Treat computation vectorial or not? if not -> polarization can be ignored
            pol                  Light polarization (give: angle in degree for linear polarization, or 'lin', 'lin_x', 'lin_y', 'circular','eliptical' ,'arb', 'azimuthal', 'radial', for eliptical you have to give pol_xy_phase_shift)
            pol_xy_phase_shift   Only for elliptic polarization: Enter the phase shift (in multiples of pi)
            aplanar              Aplanatic behaviour:    'illumination' for inbound, 'detection' for outbound, None if you want to ignore it
            r                    normalized radial vector
            z_sampling           optionally z sampling, (in units) -> only used if image is 2D
            

    '''
    def __init__(self, im, NA = 'DEFAULT', n = 'DEFAULT', wavelength = 'DEFAULT', pixelsize = 'DEFAULT', resample = 'DEFAULT', vectorial = 'DEFAULT', pol = 'DEFAULT', pol_xy_phase_shift = None ,aplanar = 'DEFAULT', unit = 'DEFAULT'):
        from .config import __DEFAULTS__;    
        from .image import image;
        from .coordinates import rr;
        
        self.__TEST__ = None;
        # Prepare settings
        if type(im) == list or type(im) == tuple:
            self.__orig_shape__ = im;
        elif type(im) == np.ndarray or type(im) == image:
            self.__orig_shape__ = im.shape;
        
        self.set_shape(im);  
        
        if NA == 'DEFAULT': self.NA = __DEFAULTS__['TRANSFER_NA']; 
        else: self.NA = NA;
        if n == 'DEFAULT': self.n = __DEFAULTS__['TRANSFER_n']; 
        else: self.n = n;
        if wavelength == 'DEFAULT': self.wavelength = __DEFAULTS__['TRANSFER_wavelength']; 
        else: self.wavelength = wavelength;
        self.foc_length = __DEFAULTS__['TRANSFER_FOCAL_LENGTH']; 
        self.pol_xy_phase_shift = pol_xy_phase_shift;
        self.set_pol(pol, pol_xy_phase_shift)
        
        if vectorial == 'DEFAULT': self.vectorial = __DEFAULTS__['TRANSFER_VECTORIAL']; 
        else: self.vectorial = vectorial;
        if resample == 'DEFAULT': self.resample = __DEFAULTS__['TRANSFER_RESAMPLE']; 
        else: self.resample = resample;
        if aplanar == 'DEFAULT': self.aplanar = __DEFAULTS__['TRANSFER_APLANATIC']; 
        else: self.aplanar = aplanar;
        self.foc_field_mode =  __DEFAULTS__['TRANSFER_FOC_ FIELD_MODE'];
        
        if unit == 'DEFAULT': self.unit = __DEFAULTS__['IMG_PIXEL_UNITS'];
        else: self.unit = unit;
        
        if type(im) == image or type(im) == otf2d:
            self.set_pxs(im.pixelsize);
            self.name = im.name+' - Transfer';
        else:
            self.set_pxs(pixelsize);
            self.name = 'Transfer fct'
        self.check_sampling();
        
     
        # rr are lateral abbe coordinates!
        r = rr(self.shape, scale = self.px_freq_step)*self.wavelength/self.NA # [:2]
        #self.r = (r<=1)*r;
        self.r = r;
        
        self.s = self.NA/self.n; 
        self.aberration_map = nip.zeros(self.shape[-2:]);

    def set_vals(self, NA = None, n = None, wavelength = None, pixelsize = None, resample = None, vectorial = None, pol = None, pol_xy_phase_shift = None ,aplanar = None, foc_field_mode = None):
        from .coordinates import rr;
        if NA is not None: self.NA = NA;
        if n is not None:  self.n = n;
        if wavelength is not None: self.wavelength = wavelength;
        if pixelsize is not None: self.set_pxs(pixelsize);
        if resample is not None: self.resample = resample;
        if vectorial is not None: self.vectorial = vectorial; 
        if pol is not None: self.set_pol(pol, pol_xy_phase_shift);
        if aplanar is not None: self.aplanar = aplanar;
        if foc_field_mode is not None: self.foc_field_mode= foc_field_mode;
        
        r = rr(self.shape[-2:], scale = self.px_freq_step[-2:])*self.wavelength/self.NA
        self.check_sampling();
        #self.r = (r<=1)*r;
        self.r = r;
        self.s = self.NA/self.n; 
    
    def set_pol(self, pol, pol_xy_phase_shift):
        from .config import __DEFAULTS__; 
        import numbers;
        if pol == 'DEFAULT': pol = __DEFAULTS__['TRANSFER_POLARIZATION']; 
        else: self.pol = pol;
        
        if pol == 'eliptical':
            if pol_xy_phase_shift is None:
                raise ValueError('Give pol_xy_phase_shift as multiples of pi for eliptical polaraization');
            self.pol_type = 'circ';
            self.pol_phase = pol_xy_phase_shift;
        elif pol == 'circular':
            self.pol_type = 'circ';
            self.pol_phase = 1;
        elif pol == 'arb':
            self.pol_type = 'circ';
            self.pol_phase = 0;
        elif pol == 'lin' or pol == 'lin_x': 
            self.pol = 0;
            self.pol_type = 'lin';
        elif pol == 'lin_y': 
            self.pol = 90;
            self.pol_type = 'lin';
        elif pol == 'radial':
            self.pol_type = 'lin';
        elif pol == 'azimuthal':
            self.pol_type = 'lin';
        elif isinstance(pol, numbers.Real):
            self.pol = pol;
            self.pol_type = 'lin';
        else:
            raise ValueError('Wrong Polarization');
    
    def set_shape(self, im, z_shape = None):
        '''
            This sets the shape of the transfer volume
            z-shape: If the original image is 2Dimensional you can give a z_shape here
        '''
        import numbers;
        from .image import image;
        if type(im) == list or type(im) == tuple:
            self.shape = im;
        elif type(im) == np.ndarray or type(im) == image:
            self.shape = im.shape;
        if z_shape is not None:
            if isinstance(z_shape, numbers.Integral) == False:
                raise TypeError('Wrong z_shape type: None or int');
          
            self.shape = list(self.shape);
            if len(self.__orig_shape__) == 2:
                if len(self.shape) ==2:
                    self.shape = tuple([z_shape] + self.shape);
                else:
                    self.shape[-3] = z_shape;
                    
                    self.shape = tuple(self.shape);
        
    def set_pxs(self, pxs):
        '''
            This allows to set the pixelsize of the current Transfer class
        '''
        import numbers;
        from .config import __DEFAULTS__; 
        from .coordinates import px_freq_step; 
        
        if pxs == 'DEFAULT':
            ret_pxs_lat = __DEFAULTS__['IMG_PIXELSIZES'][-2:];
            ret_pxs_ax = __DEFAULTS__['IMG_PIXELSIZES'][-3];
        else:
            if isinstance(pxs, numbers.Real):
                ret_pxs_lat = [pxs, pxs];
                ret_pxs_ax = pxs;
            elif type(pxs) == list or type(pxs) == tuple:
                ret_pxs_lat = pxs[-2:];
                if len(pxs) >2:
                    ret_pxs_ax = pxs[-3];
                else:
                    ret_pxs_ax = __DEFAULTS__['IMG_PIXELSIZES'][-3];
            else:
                raise ValueError('Invalid pixelsize');
        if len(self.shape) == 2:
            self.px_freq_step = px_freq_step(im = self.shape, pxs = ret_pxs_lat);
        else:
            self.px_freq_step = px_freq_step(im = self.shape, pxs = ret_pxs_lat+[ret_pxs_ax]);
        self.pixelsize = ret_pxs_lat;
        self.axial_pixelsize = ret_pxs_ax;
        
    def add_aberration(self, strength, aberration):
        '''
            add zernike aberation term:
                
                give:
                    strength      strength of the aberation as multiples of 2pi in the phase (polynomial reach from -1 to 1) 
                    type         can be
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
        '''
        import numbers;
        from .util import zernike   
        from .image import image;
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

        if isinstance(strength, numbers.Real):
            strength = [strength];
        if type(aberration) == str or type(aberration) == np.ndarray or type(aberration)== image:
            aberration = [aberration];
        elif type(aberration) == list or type(aberration) == tuple:
            if  isinstance(aberration[0], numbers.Integral):
                aberration= [aberration];
        for s, ab in zip(strength, aberration):
            if type(ab) == str:
                m = zernike_para[ab][0];
                n = zernike_para[ab][1];
                self.aberration_map += s*zernike(self.r,m,n)*np.pi;
            elif type(ab) == np.ndarray or type(ab)== image:
                self.aberration_map += ab;
            else:     
                m = ab[0];
                n = ab[1];
                self.aberration_map += s*zernike(self.r,m,n)*np.pi;
        
    def check_sampling(self):   
        for fs, size,i in zip(self.px_freq_step,self.shape, [-2,-1]):
            if 2*self.NA/self.wavelength*fs > size/2:  print('Warning: Undersampling along axis '+str(i));
        #if len(self.px_freq_step) >2:        
        #    if self.NA**2/self.wavelength*self.px_freq_step[-3]: print('Warning: Undersampling along z');

    def otf2d(self, NA = None, n = None, wavelength = None, pixelsize = None, resample = None, vectorial = None, pol = None, pol_xy_phase_shift = None ,aplanar = None, foc_field_mode = None,off_focal_dist= 0):
        #self.set_vals(self, NA, n, wavelength, pixelsize, resample , pol, pol_xy_phase_shift,aplanar, foc_field_mode);
        self.set_vals(NA = NA, n = n, wavelength = wavelength, pixelsize = pixelsize, resample = resample, vectorial = vectorial, pol = pol, pol_xy_phase_shift = pol_xy_phase_shift ,aplanar = aplanar, foc_field_mode = foc_field_mode);
        return(self.foc_field(ret_val = 'otf', off_focal_dist = off_focal_dist));
    
    def psf2d(self, NA = None, n = None, wavelength = None, pixelsize = None, resample = None, vectorial = None, pol = None, pol_xy_phase_shift = None ,aplanar = None, foc_field_mode = None, off_focal_dist = 0):
        self.set_vals(NA = NA, n = n, wavelength = wavelength, pixelsize = pixelsize, resample = resample, vectorial = vectorial, pol = pol, pol_xy_phase_shift = pol_xy_phase_shift ,aplanar = aplanar, foc_field_mode = foc_field_mode);
        return(self.foc_field(ret_val = 'intensity', off_focal_dist=off_focal_dist));
    
    def ctf2d(self, NA = None, n = None, wavelength = None, pixelsize = None, resample = None, vectorial = None, pol = None, pol_xy_phase_shift = None ,aplanar = None, foc_field_mode = None,off_focal_dist= 0):
        self.set_vals(NA = NA, n = n, wavelength = wavelength, pixelsize = pixelsize, resample = resample, vectorial = vectorial, pol = pol, pol_xy_phase_shift = pol_xy_phase_shift ,aplanar = aplanar, foc_field_mode = foc_field_mode);
        return(self.foc_field(ret_val = 'ctf',off_focal_dist=off_focal_dist));
    
    def apsf2d(self, NA = None, n = None, wavelength = None, pixelsize = None, resample = None, vectorial = None, pol = None, pol_xy_phase_shift = None ,aplanar = None, foc_field_mode = None, off_focal_dist= 0):
        self.set_vals( NA = NA, n = n, wavelength = wavelength, pixelsize = pixelsize, resample = resample, vectorial = vectorial, pol = pol, pol_xy_phase_shift = pol_xy_phase_shift ,aplanar = aplanar, foc_field_mode = foc_field_mode);
        return(self.foc_field(ret_val = 'field',off_focal_dist=off_focal_dist));    
   
    
    def psf3d(self, NA = None, n = None, wavelength = None, pixelsize = None, resample = None, vectorial = None, pol = None, pol_xy_phase_shift = None ,aplanar = None, foc_field_mode = None, z_shape = None):
        self.set_vals( NA = NA, n = n, wavelength = wavelength, pixelsize = pixelsize, resample = resample, vectorial = vectorial, pol = pol, pol_xy_phase_shift = pol_xy_phase_shift ,aplanar = aplanar, foc_field_mode = foc_field_mode);
        return(self.propagate(z_shape, method = 'angle', ret_val = 'intensity'));
    
    def otf3d(self, NA = None, n = None, wavelength = None, pixelsize = None, resample = None, vectorial = None, pol = None, pol_xy_phase_shift = None ,aplanar = None, foc_field_mode = None, z_shape = None):
        self.set_vals( NA = NA, n = n, wavelength = wavelength, pixelsize = pixelsize, resample = resample, vectorial = vectorial, pol = pol, pol_xy_phase_shift = pol_xy_phase_shift ,aplanar = aplanar, foc_field_mode = foc_field_mode);
        return(self.propagate(z_shape, method = 'angle', ret_val = 'otf'));
    
    def ctf3d(self, NA = None, n = None, wavelength = None, pixelsize = None, resample = None, vectorial = None, pol = None, pol_xy_phase_shift = None ,aplanar = None, foc_field_mode = None, z_shape = None):
        self.set_vals( NA = NA, n = n, wavelength = wavelength, pixelsize = pixelsize, resample = resample, vectorial = vectorial, pol = pol, pol_xy_phase_shift = pol_xy_phase_shift ,aplanar = aplanar, foc_field_mode = foc_field_mode);
        return(self.propagate(z_shape, method = 'angle', ret_val = 'ctf'));
    
    def apsf3d(self, NA = None, n = None, wavelength = None, pixelsize = None, resample = None, vectorial = None, pol = None, pol_xy_phase_shift = None ,aplanar = None, foc_field_mode = None, z_shape = None):
        self.set_vals( NA = NA, n = n, wavelength = wavelength, pixelsize = pixelsize, resample = resample, vectorial = vectorial, pol = pol, pol_xy_phase_shift = pol_xy_phase_shift ,aplanar = aplanar, foc_field_mode = foc_field_mode);
        return(self.propagate(z_shape, method = 'angle', ret_val = 'field'));
    
    
    
    def propagate(self, z_shape = None, method = 'kz', ret_val = 'ctf'):
        '''
            propagate field
            
            field: which field (in focus) to propagate
            
            methods: -> currently implemented for testing purposes
                kz:  Rainers method
                angle: my method
        '''
        from .transformations import ft, ift;
        from .coordinates import freq_ramp, zz, px_freq_step;
        from .image import image;
        field = self.foc_field(ret_val = 'ctf');
        field = np.nan_to_num(field, copy = False);
        
        if len(self.__orig_shape__) == 2:
            if z_shape is None:
                raise ValueError('No z_shape given');
            self.set_shape(self.shape, z_shape);
        
        if method == 'kz':
            # Transposing is necessary for broadcasting
            fx = freq_ramp((field.shape[-2],field.shape[-1]),self.pixelsize[-1], axis =-1); # .transpose();   # RH 2.2.19
            fy = freq_ramp((field.shape[-2],field.shape[-1]),self.pixelsize[-2], axis =-2); # .transpose();
            z = zz(self.shape); # .transpose();
            
            np.seterr(invalid = 'ignore');
            kz = np.nan_to_num(2*np.pi*np.sqrt(self.n**2/self.wavelength**2-(fx**2+fy**2)))   
            np.seterr(invalid = 'warn');
            propagator = kz * z*self.axial_pixelsize; 
            ret = (field.np.exp(-1j*propagator));  # RH 2.2.19
            
        elif method == 'angle':
            z = zz(self.shape); # .transpose();
            np.seterr(invalid = 'ignore');

            propagator = self.n*np.pi*2/self.wavelength*np.sqrt(1-self.r**2*self.s**2)* self.axial_pixelsize*z;
            np.seterr(invalid = 'warn');
            ret = (field*np.exp(-1j*propagator));      # RH 2.2.19   .transpose()   field.transpose()*np.exp(-1j*propagator.transpose()
        
        np.nan_to_num(ret,copy = False);
        if ret_val == 'field':
            ret = ift2d(ret, shift = True, shift_before = True, norm = None);
        elif ret_val == 'intensity':
            ret = ift2d(ret, shift = True, shift_before = True, norm = None);
            ret = (np.abs(ret)**2).astype(np.float64);
            if self.vectorial:
                ret = np.sum(ret, axis = -4).squeeze();
            ret = ret/ret.max();
            ret = ret/ret.sum();
                
        elif ret_val == 'ctf':
            ret = ft(ret, shift = True, shift_before = True, norm = None, axes=(-3));
        elif ret_val == 'otf':
            ret = ift2d(ret, shift = True, shift_before = True, norm = None);
            ret = np.abs(ret)**2;
            if self.vectorial:
                ret = np.sum(ret, axis = -4);
                ret = ret.squeeze();
            ret = ret/ret.max();
            ret = ret/ret.sum();
            ret = ft(ret, shift = True, shift_before = True, norm =None, ret = 'abs');
        else:
            raise ValueError('Wrong ret_val: "ctf", "otf", "field" or "intensity"');    
    
        ret = image(ret, pixelsize = self.pixelsize, unit = self.unit, name = self.name+' ('+ret_val+')', info = 'NA='+str(self.NA)+', n='+str(self.n)+', wavelength='+str(self.wavelength)+', mode='+self.foc_field_mode)
        
        if len(ret.pixelsize) == 2:
            ret.pixelsize += [self.axial_pixelsize]; 
        else:
            ret.pixelsize[-3] = self.axial_pixelsize;
        if ret_val == 'ctf' or ret_val == 'otf':
            ret.pixelsize = px_freq_step(self.shape, ret.pixelsize);
        if self.vectorial and ret_val == 'ctf' or ret_val == 'field':
            ret.dim_description['d3'] = ['Ex','Ey','Ez'];
        return(ret)            
    
    def foc_field(self, ret_val = 'ctf', off_focal_dist = 0):
        '''
            compute the field in the focal plane
                mode:
                    'theoretical':        compute from sinc
                    'circular':           compute field from circular transfer
        
            ret_val:
                'field'      returns apsf
                'intensity'  returns psf
                'ctf'        returns ctf
                'otf'        returns otf
            
            norm : Normalization of return
        '''
        from scipy.special import j1;
        from .coordinates import phiphi;
        from .image import image, cat;
        from .transformations import ft, ift;
        
        shape = self.shape[-2:];

        if self.foc_field_mode == 'theoretical':
            np.seterr(divide ='ignore', invalid ='ignore');
            arg = rr(shape,scale=np.array(self.pixelsize)*2*np.pi*self.NA/self.wavelength) #  RH 3.2.19 was 2*mp.pi*np.sqrt((xx(shape)*self.pixelsize[0])**2+(yy(shape)*self.pixelsize[1])**2)*self.NA/self.wavelength;
            ret = 2*j1(arg)/arg;
            ret[shape[-2]//2, shape[-1]//2] = 1;
            np.seterr(divide='warn', invalid ='warn');  
            ret = ret/np.sum(np.abs(ret));      
        elif self.foc_field_mode == 'circular':
            ret = ift((self.r<= 1)*1.0,  shift = True, shift_before = True, norm = None);
            ret = ret/np.max(np.abs(ret))
        self.TEST =ret;        
        ret = ft(ret, shift = True, shift_before =True, norm = None);
        self.TEST1 = ret
        # add aplanatic factor
        oms2r2=1-self.s**2*self.r**2  # one minus s^2*r^2
        validmask=oms2r2>0
        ret[~validmask]=0.0;
        if self.aplanar == 'illumination':
            ret[validmask] *=  (oms2r2[validmask])**0.25;   # RH 3.2.19
        elif self.aplanar == 'detection':
            ret[validmask] *=  1/(oms2r2[validmask])**0.25;   #RH 3.2.19
        # add focal distance
        if off_focal_dist !=0:
            np.seterr(invalid = 'ignore');
            propagator = self.n*np.pi*2* self.axial_pixelsize*off_focal_dist/self.wavelength*np.sqrt(oms2r2);
            np.seterr(invalid = 'warn');
            ret*= np.exp(-1j*propagator);
           
        # add aberration
        ret *= np.exp(1j*self.aberration_map);
        # add vectorlial effects
        if self.vectorial:
            def __make_components__(pol, phase):
                phase = np.exp(1j*phase*np.pi/2);
                rs=self.r*self.s;
                rs[rs>1.0]=1.0
                oms2r2=1-rs**2  # one minus s^2*r^2
#                oms2r2[oms2r2<0]=0.0;
                soms2r2=np.sqrt(oms2r2)  # sqrt(one minus s^2*r^2)
                if pol == 'x':
                    theta = phiphi(shape, offset = 0);
                    Ex = (np.cos(theta)**2*soms2r2+np.sin(theta)**2)*phase; Ey = ( np.sin(theta)*np.cos(theta)*(soms2r2-1) )*phase; Ez = np.cos(theta)*rs*phase;
                elif pol == 'y':
                    theta = phiphi(shape, offset = 0);
                    Ex = (np.sin(theta)*np.cos(theta)*(soms2r2-1))*phase;Ey = (np.sin(theta)**2*soms2r2+np.cos(theta)**2)*phase; Ez = np.sin(theta)*rs*phase;
                elif pol == 'radial':
                    theta = phiphi(shape, offset = 0);
                    Ex = (np.cos(theta)*soms2r2)*phase; Ey = (np.sin(theta)*soms2r2)*phase; Ez = rs*phase;                
                elif pol == 'azimuthal':
                    theta = phiphi(shape, offset = 0);
                    Ex = -np.sin(theta)*phase; Ey =  np.cos(theta)*phase; Ez = np.zeros_like(theta);
                else:
                    theta = phiphi(shape, offset = pol*np.pi/180);
                    Ex = (np.cos(theta)**2*soms2r2+np.sin(theta)**2)*phase;  # RH: Please fix this, yield rundtime warning
                    Ey = (np.sin(theta)*np.cos(theta)*(soms2r2-1) )*phase;  # RH: Please fix this, yield rundtime warning
                    Ez = np.cos(theta)*rs*phase;
                return(Ex,Ey,Ez);
            nfac=1/np.sqrt(2);
            if self.pol_type == 'lin':
                Ex,Ey,Ez = __make_components__(self.pol, 0);
            elif self.pol_type == 'circ':
                Exx, Eyx, Ezx = __make_components__('x',0);
                Exy, Eyy, Ezy = __make_components__('y',self.pol_phase);
                Ex = nfac*(Exx+Exy);Ey = nfac*(Eyx+Eyy);Ez = nfac*(Ezx+Ezy);
            elif self.pol_type == 'unpolarized':
                print('Waring: Unpolarized light is modelled by overlaying azimuthally and radially')
                Exx, Eyx, Ezx = __make_components__('azimuthal',0);
                Exy, Eyy, Ezy = __make_components__('radial',0);
                Ex = nfac*(Exx+Exy);Ey = nfac*(Eyx+Eyy);Ez = nfac*(Ezx+Ezy);
            ret = cat((ret*Ez, ret*Ey, ret*Ex),-4);
        
        np.nan_to_num(ret,copy = False);  # is this still necessary?

        if ret_val == 'field':
            ret = ift2d(ret, shift = True, shift_before = True, norm = None);
        elif ret_val == 'intensity':
            ret = ift2d(ret, shift = True, shift_before = True, norm = None);
            ret = np.real(abssqr(ret)) # RH 3.2.19  .astype(np.float64);
            if self.vectorial:
                ret = np.sum(ret, axis = -4).squeeze();
#            ret = ret/ret.max();
            ret = ret/ret.sum(); # RH 2.2.19
        elif ret_val == 'ctf':
            pass;
        elif ret_val == 'otf':
            ret = ift2d(ret, shift = True, shift_before = True, norm = None);
            ret = abssqr(ret);
            if self.vectorial:
                ret = np.sum(ret, axis = -4);
                ret = ret.squeeze();
            ret = ret/ret.max();
            ret = ret/ret.sum();
            ret = ft(ret, shift = True, shift_before = True, norm = None, ret ='abs');
        else:
            raise ValueError('Wrong ret_val: "ctf", "otf", "field" or "intensity"');    
    
        ret = image(ret, pixelsize = self.pixelsize, unit = self.unit, name = self.name+' ('+ret_val+')', info = 'NA='+str(self.NA)+', n='+str(self.n)+', wavelength='+str(self.wavelength)+', mode='+self.foc_field_mode)
        if ret_val == 'ctf' or ret_val == 'otf':
            ret.pixelsize = self.px_freq_step;
        if self.vectorial and ret_val == 'ctf' or ret_val == 'field':
            ret.dim_description['d3'] = ['Ez','Ey','Ex'];
        return(ret)
    


class otf2d(image):
    def __new__(cls, im, NA = 'DEFAULT', n = 'DEFAULT', wavelength = 'DEFAULT', pixelsize = 'DEFAULT', resample = 'DEFAULT', vectorial = 'DEFAULT', pol = 'DEFAULT', pol_xy_phase_shift = None ,aplanar = 'DEFAULT', unit = 'DEFAULT', foc_field_mode  =None, off_focal_dist =0):
        t = transfer(im, NA, n, wavelength, pixelsize, resample, vectorial, pol, pol_xy_phase_shift ,aplanar, unit);
        opx = t.__dict__['pixelsize'];
        obj = t.otf2d(foc_field_mode = foc_field_mode, off_focal_dist= off_focal_dist).view(cls);
        px = obj.pixelsize;
        obj.__dict__ = t.__dict__;
        obj.__dict__['pixelsize'] = px;
        obj.old_pxs = opx
        obj.trans = t;
        obj.off_focal_dist = off_focal_dist;
        return(obj)
    
    def add_aberration(self, strength, aberration):
        self.trans.add_aberration(strength, aberration);
        self.recalc();
    def recalc(self):
         self[:] = self.trans.otf2d(NA = self.NA, n = self.n, wavelength = self.wavelength, pixelsize = self.old_pxs, resample = self.resample, vectorial = self.vectorial, pol = self.pol, pol_xy_phase_shift = self.pol_xy_phase_shift ,aplanar = self.aplanar, foc_field_mode = self.foc_field_mode, off_focal_dist = self.off_focal_dist).view(type(self))
        
    def __array_finalize__(self, obj):
        if obj is None: return;
        self.pixelsize = getattr(obj, 'pixelsize', [100,100]);
        self.old_pxs =getattr(obj, 'old_pxs', [100,100])
        self.off_focal_dist = getattr(obj,'off_focal_dist',0);
class psf2d(image):
    def __new__(cls, im, NA = 'DEFAULT', n = 'DEFAULT', wavelength = 'DEFAULT', pixelsize = 'DEFAULT', resample = 'DEFAULT', vectorial = 'DEFAULT', pol = 'DEFAULT', pol_xy_phase_shift = None ,aplanar = 'DEFAULT', unit = 'DEFAULT', foc_field_mode  =None, off_focal_dist =0):
        t = transfer(im, NA, n, wavelength, pixelsize, resample, vectorial, pol, pol_xy_phase_shift ,aplanar, unit);
        opx = t.__dict__['pixelsize'];
        obj = t.psf2d(foc_field_mode = foc_field_mode, off_focal_dist= off_focal_dist).view(cls);
        px = obj.pixelsize;
        obj.__dict__ = t.__dict__;
        obj.__dict__['pixelsize'] = px;
        obj.old_pxs = opx
        obj.trans = t;
        obj.off_focal_dist = off_focal_dist;
        return(obj)
    def add_aberration(self, strength, aberration):
        self.trans.add_aberration(strength, aberration);
        self.recalc();
    def recalc(self):
         self[:] = self.trans.psf2d(NA = self.NA, n = self.n, wavelength = self.wavelength, pixelsize = self.old_pxs, resample = self.resample, vectorial = self.vectorial, pol = self.pol, pol_xy_phase_shift = self.pol_xy_phase_shift ,aplanar = self.aplanar, foc_field_mode = self.foc_field_mode, off_focal_dist = self.off_focal_dist).view(type(self))
        
    def __array_finalize__(self, obj):
        if obj is None: return;
        self.pixelsize = getattr(obj, 'pixelsize', [100,100]);
        self.old_pxs =getattr(obj, 'old_pxs', [100,100])
        self.off_focal_dist = getattr(obj,'off_focal_dist',0);
        
class apsf2d(image):
    def __new__(cls, im, NA = 'DEFAULT', n = 'DEFAULT', wavelength = 'DEFAULT', pixelsize = 'DEFAULT', resample = 'DEFAULT', vectorial = 'DEFAULT', pol = 'DEFAULT', pol_xy_phase_shift = None ,aplanar = 'DEFAULT', unit = 'DEFAULT', foc_field_mode  =None, off_focal_dist =0):
        t = transfer(im, NA, n, wavelength, pixelsize, resample, vectorial, pol, pol_xy_phase_shift ,aplanar, unit);
        opx = t.__dict__['pixelsize'];
        obj = t.apsf2d(foc_field_mode = foc_field_mode, off_focal_dist= off_focal_dist).view(cls);
        px = obj.pixelsize;
        descr = obj.dim_description;
        obj.__dict__ = t.__dict__;
        obj.__dict__['pixelsize'] = px;
        obj.old_pxs = opx
        obj.trans = t;
        obj.dim_description = descr;
        obj.off_focal_dist = off_focal_dist;
        return(obj)
    
    def add_aberration(self, strength, aberration):
        self.trans.add_aberration(strength, aberration);
        self.recalc();
    def recalc(self):
         self[:] = self.trans.apsf2d(NA = self.NA, n = self.n, wavelength = self.wavelength, pixelsize = self.old_pxs, resample = self.resample, vectorial = self.vectorial, pol = self.pol, pol_xy_phase_shift = self.pol_xy_phase_shift ,aplanar = self.aplanar, foc_field_mode = self.foc_field_mode, off_focal_dist = self.off_focal_dist).view(type(self))
        
    def __array_finalize__(self, obj):
        if obj is None: return;
        self.pixelsize = getattr(obj, 'pixelsize', [100,100]);
        self.old_pxs =getattr(obj, 'old_pxs', [100,100])
        self.dim_description= getattr(obj, 'dim_description', [])
        self.off_focal_dist = getattr(obj,'off_focal_dist',0);
        
class ctf2d(image):
    def __new__(cls, im, NA = 'DEFAULT', n = 'DEFAULT', wavelength = 'DEFAULT', pixelsize = 'DEFAULT', resample = 'DEFAULT', vectorial = 'DEFAULT', pol = 'DEFAULT', pol_xy_phase_shift = None ,aplanar = 'DEFAULT', unit = 'DEFAULT', foc_field_mode  =None, off_focal_dist =0):
        t = transfer(im, NA, n, wavelength, pixelsize, resample, vectorial, pol, pol_xy_phase_shift ,aplanar, unit);
        opx = t.__dict__['pixelsize'];
        obj = t.ctf2d(foc_field_mode = foc_field_mode, off_focal_dist= off_focal_dist).view(cls);
        px = obj.pixelsize;
        descr = obj.dim_description;
        obj.__dict__ = t.__dict__;
        obj.__dict__['pixelsize'] = px;
        obj.old_pxs = opx
        obj.trans = t;
        obj.dim_description = descr;
        obj.off_focal_dist = off_focal_dist;
        return(obj)
    
    def add_aberration(self, strength, aberration):
        self.trans.add_aberration(strength, aberration);
        self.recalc();
    def recalc(self):
         self[:] = self.trans.ctf2d(NA = self.NA, n = self.n, wavelength = self.wavelength, pixelsize = self.old_pxs, resample = self.resample, vectorial = self.vectorial, pol = self.pol, pol_xy_phase_shift = self.pol_xy_phase_shift ,aplanar = self.aplanar, foc_field_mode = self.foc_field_mode, off_focal_dist = self.off_focal_dist).view(type(self))
        
    def __array_finalize__(self, obj):
        if obj is None: return;
        self.pixelsize = getattr(obj, 'pixelsize', [100,100]);
        self.old_pxs =getattr(obj, 'old_pxs', [100,100])    
        self.dim_description= getattr(obj, 'dim_description', [])
        self.off_focal_dist = getattr(obj,'off_focal_dist',0);
        
class otf3d(image):
    def __new__(cls, im, NA = 'DEFAULT', n = 'DEFAULT', wavelength = 'DEFAULT', pixelsize = 'DEFAULT', resample = 'DEFAULT', vectorial = 'DEFAULT', pol = 'DEFAULT', pol_xy_phase_shift = None ,aplanar = 'DEFAULT', unit = 'DEFAULT', foc_field_mode  =None, z_shape = None):
        t = transfer(im, NA, n, wavelength, pixelsize, resample, vectorial, pol, pol_xy_phase_shift ,aplanar, unit);
        opx = t.__dict__['pixelsize'];
        obj = t.otf3d(foc_field_mode = foc_field_mode, z_shape = z_shape).view(cls);
        px = obj.pixelsize;
        obj.__dict__ = t.__dict__;
        obj.__dict__['pixelsize'] = px;
        obj.old_pxs = opx
        obj.trans = t;
        obj.z_shape = z_shape;
        return(obj)
    def add_aberration(self, strength, aberration):
        self.trans.add_aberration(strength, aberration);
        self.recalc();
    def recalc(self):
         self[:] = self.trans.otf3d(NA = self.NA, n = self.n, wavelength = self.wavelength, pixelsize = self.old_pxs, resample = self.resample, vectorial = self.vectorial, pol = self.pol, pol_xy_phase_shift = self.pol_xy_phase_shift ,aplanar = self.aplanar, foc_field_mode = self.foc_field_mode, z_shape = self.z_shape).view(type(self))
        
    def __array_finalize__(self, obj):
        if obj is None: return;
        self.pixelsize = getattr(obj, 'pixelsize', [100,100]);
        self.old_pxs =getattr(obj, 'old_pxs', [100,100])
        self.z_shape=getattr(obj, 'z_shape', None)
        self.trans = getattr(obj, 'trans', None)
        
      
class psf3d(image):
    def __new__(cls, im, NA = 'DEFAULT', n = 'DEFAULT', wavelength = 'DEFAULT', pixelsize = 'DEFAULT', resample = 'DEFAULT', vectorial = 'DEFAULT', pol = 'DEFAULT', pol_xy_phase_shift = None ,aplanar = 'DEFAULT', unit = 'DEFAULT', foc_field_mode  =None, z_shape = None):
        t = transfer(im, NA, n, wavelength, pixelsize, resample, vectorial, pol, pol_xy_phase_shift ,aplanar, unit);
        opx = t.__dict__['pixelsize'];
        obj = t.psf3d(foc_field_mode = foc_field_mode, z_shape = z_shape).view(cls);
        px = obj.pixelsize;
        obj.__dict__ = t.__dict__;
        obj.__dict__['pixelsize'] = px;
        obj.old_pxs = opx
        obj.trans = t;
        obj.z_shape = z_shape;
        return(obj)
    def add_aberration(self, strength, aberration):
        self.trans.add_aberration(strength, aberration);
        self.recalc();
    def recalc(self):
         self[:] = self.trans.psf3d(NA = self.NA, n = self.n, wavelength = self.wavelength, pixelsize = self.old_pxs, resample = self.resample, vectorial = self.vectorial, pol = self.pol, pol_xy_phase_shift = self.pol_xy_phase_shift ,aplanar = self.aplanar, foc_field_mode = self.foc_field_mode, z_shape = self.z_shape).view(type(self))
        
    def __array_finalize__(self, obj):
        if obj is None: return;
        self.pixelsize = getattr(obj, 'pixelsize', [100,100]);
        self.old_pxs =getattr(obj, 'old_pxs', [100,100])
        self.z_shape=getattr(obj, 'z_shape', None)
        self.trans = getattr(obj, 'trans', None)
        
class ctf3d(image):
    def __new__(cls, im, NA = 'DEFAULT', n = 'DEFAULT', wavelength = 'DEFAULT', pixelsize = 'DEFAULT', resample = 'DEFAULT', vectorial = 'DEFAULT', pol = 'DEFAULT', pol_xy_phase_shift = None ,aplanar = 'DEFAULT', unit = 'DEFAULT', foc_field_mode  =None, z_shape = None):
        t = transfer(im, NA, n, wavelength, pixelsize, resample, vectorial, pol, pol_xy_phase_shift ,aplanar, unit);
        opx = t.__dict__['pixelsize'];
        obj = t.ctf3d(foc_field_mode = foc_field_mode, z_shape = z_shape).view(cls);
        px = obj.pixelsize;
        descr = obj.dim_description;
        obj.__dict__ = t.__dict__;
        obj.__dict__['pixelsize'] = px;
        obj.old_pxs = opx
        obj.trans = t;
        obj.z_shape = z_shape;
        obj.dim_description = descr;
        return(obj)
    def add_aberration(self, strength, aberration):
        self.trans.add_aberration(strength, aberration);
        self.recalc();
    def recalc(self):
         self[:] = self.trans.ctf3d(NA = self.NA, n = self.n, wavelength = self.wavelength, pixelsize = self.old_pxs, resample = self.resample, vectorial = self.vectorial, pol = self.pol, pol_xy_phase_shift = self.pol_xy_phase_shift ,aplanar = self.aplanar, foc_field_mode = self.foc_field_mode, z_shape = self.z_shape).view(type(self))
        
    def __array_finalize__(self, obj):
        if obj is None: return;
        self.pixelsize = getattr(obj, 'pixelsize', [100,100]);
        self.old_pxs =getattr(obj, 'old_pxs', [100,100])
        self.z_shape=getattr(obj, 'z_shape', None)
        self.trans = getattr(obj, 'trans', None)
        self.dim_description= getattr(obj, 'dim_description', [])
class apsf3d(image):
    def __new__(cls, im, NA = 'DEFAULT', n = 'DEFAULT', wavelength = 'DEFAULT', pixelsize = 'DEFAULT', resample = 'DEFAULT', vectorial = 'DEFAULT', pol = 'DEFAULT', pol_xy_phase_shift = None ,aplanar = 'DEFAULT', unit = 'DEFAULT', foc_field_mode  =None, z_shape = None):
        t = transfer(im, NA, n, wavelength, pixelsize, resample, vectorial, pol, pol_xy_phase_shift ,aplanar, unit);
        opx = t.__dict__['pixelsize'];
        obj = t.otf3d(foc_field_mode = foc_field_mode, z_shape = z_shape).view(cls);
        px = obj.pixelsize;
        descr = obj.dim_description;
        obj.__dict__ = t.__dict__;
        obj.__dict__['pixelsize'] = px;
        obj.old_pxs = opx
        obj.trans = t;
        obj.z_shape = z_shape;
        obj.dim_description = descr;
        return(obj)
    def add_aberration(self, strength, aberration):
        self.trans.add_aberration(strength, aberration);
        self.recalc();
    def recalc(self):
         self[:] = self.trans.apsf3d(NA = self.NA, n = self.n, wavelength = self.wavelength, pixelsize = self.old_pxs, resample = self.resample, vectorial = self.vectorial, pol = self.pol, pol_xy_phase_shift = self.pol_xy_phase_shift ,aplanar = self.aplanar, foc_field_mode = self.foc_field_mode, z_shape = self.z_shape).view(type(self))
        
    def __array_finalize__(self, obj):
        if obj is None: return;
        self.pixelsize = getattr(obj, 'pixelsize', [100,100]);
        self.old_pxs =getattr(obj, 'old_pxs', [100,100])
        self.z_shape=getattr(obj, 'z_shape', None)
        self.trans = getattr(obj, 'trans', None)
        self.dim_description= getattr(obj, 'dim_description', [])
# def create_psf(NA, n, wavelength,pixelsize,magnification):
#     '''
#
#         IS NOT NEEDED ANYMORE : TODO: ELIMINATE IN NEXT VERSION     CK 21.02.2019
#         Computes an ordinary 2D PSF:
#             If the first zero point of the Besselfunction is smaller than half of the pixelsize, than it
#             the psf is just one. Otherwise it is computed according to the grid given by the pixels
#         NA: numerical aperture
#         n: refractive index
#         wavelength in nm
#         pixelsize in mum
#         magnificiation
#
#         We scale the PSF up to the fifths zero point of the besselfunction!
#     '''
#     from scipy import special as spec;
#     N = 1/(2*np.tan(np.arcsin(NA/n)));         # F number
#     wavelength = wavelength/1000;              # wavelength in mum
#     pixelsize = pixelsize/magnification;
#     max_normalize = 7;        # Maximum normalized x -> 16.706 is the 5th zero point of the J1 -function -> can of course be canged
#     if (3.8317*wavelength*N/(np.pi) <= (pixelsize/2)):
#         psf = np.asarray([[1]]);
#     else:
#         x1 = np.arange(0,max_normalize,pixelsize*np.pi/(wavelength*N))
#         x2 = x1[1:];
#         x2 = -1*x2[::-1];
#         x = np.append(x2,x1);
#         X,Y = np.meshgrid(x,x);
#         R = np.sqrt(X**2+Y**2);
#         psf = (2*spec.j1(R)/R)**2
#         psf[np.int8((np.size(x)-1)/2),np.int8((np.size(x)-1)/2)] = 1;
#     return(psf);

def jinc(mysize=[256,256],myscale=None):
    '''
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
    '''
    from scipy.special import j1;
    if myscale is None:
        pixelSize=203;  # half the Nyquist freq
        mylambda=488/pixelSize;
        na=0.3;
        AbbeLimit=mylambda/na;   # coherent Abbe limit, central illumination, not incoherent
        ftradius=1/AbbeLimit*mysize;
        myscales=ftradius/mysize; # [100 100]/(488/0.3);
    myradius=np.pi*nip.rr(mysize,scale=myscale)
    res=j1(2*myradius) / (myradius)
    nip.MidValAsg(res,1.0)
    return res

def perfect_psf(im, NA, n, wavelength, pixelsize, mode = 'lateral'):
    from .image import image;
    #if type(im) == image or type(im) == np.ndarray:
    from numbers import Real;
    from scipy.special import j1;
    from .coordinates import xx,yy;
    if isinstance(pixelsize, Real):
        pixelsize = [pixelsize, pixelsize];
    if type(im) == image:
        pixelsize = im.pixelsize;
    arg = rr(im,scale=(2*pi*np.array(pixelsize)/wavelength/NA));   # RH 3.2.19
#    arg = 2*np.pi*np.sqrt((xx(im)*pixelsize[0])**2+(yy(im)*pixelsize[1])**2)/(wavelength*NA);
    atf = 2*j1(arg)/arg;
    return(atf)
    

def otf2D(im = (256,256), px_size=50, wavelength=500, NA=1.0, exp_damping = 1.0):
    '''
        Compute the OTF in 2D:
            
             im: Image or 2D shape (default is (256,256))
             px_size: pixelsize in nm (Default is 50)
             wavelength in nm (Default is 500)
             Numerical aperture (Default is 1.0)
             exp_damping:   Exponential factor for correcting aberations (default is 1.0)
             
        Algorithm: 
            use "nip.otf_support" using NA/2 to create ATF and autocorrelate those!
            
    '''
    from .mask import otf_support;
    from .image import correl;
    atf = otf_support(im, px_size, wavelength, NA/2);
    otf = correl(atf, atf);
    otf = np.abs(otf)**exp_damping/np.max(np.abs(otf))*np.exp(1j*np.angle(otf));
    return(otf);
    
def get_field_spec_in_focus(im = (256,256), px_size = [50,50], wavelength = 500, NA =1.0, n = 1, method = 'disc', input_pol = 'y', direction = 'confocal'):
    '''
        returns the field spectrum in the focals plane
        im:             image or shape
        px_size:        pixelsizes 
        wavelength:     wavelength
        NA:             NA
        n               refractive index
        method:         method of computation:
                            disc:    returns disc mask accroding to abbe_limit
                            TODO:
                                ft(sinc)
                                much more
                                
            
        input_pol       polarization of incomming beam
        
        
        
    '''    
    if method == 'disc':
        from .mask import otf_support;
        return(otf_support((im[-2], im[-1]), (px_size[-2], px_size[-1]), NA = NA/2, l= wavelength))

def PSF3D(im = (256,256,32), px_size=[50,50,100], wavelength=500, NA=1.0,n = 1.0,method = 'propagation', focal_plane = 'disc', ret_val = 'PSF'):
    '''
        compute the 3D OTF for a given volume:
            
            im:           image or shape of the image
            px_size:      pixelsizes in x,y,z  (if integer given, the pixelsizes will be the same for all)
            wavelngth:    wavelengths;
            NA            NA
            n             refractive index
            method:       How to be computed?
                            - 'propagation'      propagates focal plane around z at given range
            
            focal_plane:  How is the field in the focal plane value to be computed
                            - 'disc'             simple disk mask according to Abbe limit
                            TODO: IMPLEMENT MORE 
                            
            ret_val:       what to return:
                            'PSF':        Intensity PSF
                            'FIELD':      field distribution
                            'OTF':        OTF (ft(PSF))
                            'CTF':        CTF (ft(FIELD))
                            'ALL':        A list of all four in order (FIELD; PSF; OTF; CTF)
            
            
            TODO:
                EXTEND WITH DIFFERENT TECHNIQUES
                INCLUDE MORE FOCAL PLANE VALUES
                
            
    '''
    
    if type(im) == np.ndarray:
        im = np.shape(im);
        
    
    if method == 'propagation':
        if focal_plane == 'disc':
            from .mask import otf_support;
            atf_focus = otf_support((im[-2], im[-1]), (px_size[-2], px_size[-1]), NA = NA/2, l= wavelength);
            field = field_propagation(atf_focus, im[-3], px_size, wavelength = wavelength, refractive_index= n);
        else:
            print('NOT YET IMPLEMENTED');
            return; 
    else:
        print('NOT YET IMPLEMENTED');
        return;
    
    from .transformations import ft;
    if ret_val == 'FIELD':
        return(field);
    elif ret_val == 'PSF':
        return(np.real(abssqr(field)))
    elif ret_val == 'OTF':
        return (ft(abssqr(field),shift_before=True));
    elif ret_val == 'CTF':
        return (ft(field,shift_before=True));
    elif ret_val == 'ALL':
        return ([field,np.real(abssqr(field)), ft(abssqr(field),shift_before=True), ft(field,shift_before=True)])
    
def field_propagation(field, z_shape = 32, pixel_sizes = [50,50,100], field_space = 'fourier_space', wavelength= 500, refractive_index = 1):
    '''
        Propagate the given field in the plane 0 along a z-range
        
        - Field:           The field in the slice z0 (e.g. in the focus)
        - z_shape:         The z-shape of the volume
        
        - pixel_sizes      Pixelsizes in x,y,z (if integer given, the pixelsizes will be the same for all)
        - field_space:     is the field given as spectrum ('fourier_space')? Otherwise: it will be fourier transformened beforehand
        - wavelength       wavelength
        - refractive_index refractive_index
    '''
    
    from .transformations import ft, ift;
    import numbers
    from .coordinates import freq_ramp, zz;

    if type(pixel_sizes) == np.ndarray:
        pixel_sizes = list(pixel_sizes)

    if type(pixel_sizes) == list or type(pixel_sizes) == tuple:
        if len(pixel_sizes) != 3:
            print('Wrong length for pixel sizes');
    elif isinstance(pixel_sizes, numbers.Number):
        pixel_sizes = [pixel_sizes, pixel_sizes, pixel_sizes];


    if field_space != 'fourier_space':
        field = ft(field);
    
    #atf_focus = nip.otf_support((shape[0], shape[1]), pixel_size_x, NA = NA/2, l= l);
    fx = freq_ramp((field.shape[-2],field.shape[-1]),pixel_sizes[-1], axis =-1);
    fy = freq_ramp((field.shape[-2],field.shape[-1]),pixel_sizes[-2], axis =-2);
    z = zz((z_shape,field.shape[-2], field.shape[-1])) # np.rollaxis(zz((field.shape[0], field.shape[1],z_shape)),2,0);  # RH 3.2.19
    np.seterr(invalid = 'ignore');
    kz = np.nan_to_num(2*np.pi*np.sqrt(refractive_index**2/wavelength**2-(fx**2+fy**2)))   
    np.seterr(invalid = 'warn');

    #  ALTERNATIVELY TO CHECK:    
#    ang_x = np.arctan(fx*l);
#    ang_y = np.arctan(fy*l);
#    kz = 2*np.pi*n/(l*np.cos(ang_x)*np.cos(ang_y));
    
    propagator = kz * z*pixel_sizes[-3];
#    field_spectrum_at_z = np.rollaxis(field*np.exp(-1j*propagator),0,3);
    field_spectrum_at_z = field*np.exp(-1j*propagator) # np.rollaxis(field*np.exp(-1j*propagator),0,3);
    return ift2d(field_spectrum_at_z,shift=True) # changed to use the ft2d routines. RH 3.2.19
#    return(np.fft.fftshift(ift(field_spectrum_at_z, axes =(-1,-2)), axes = (-1,-2)))


def PSF2ROTF(psf):
    '''
        Transforms a real-valued PSF to a half-complex RFT, at the same time precompensating for the fftshift
    '''
    o = image(rft(psf,shift_before=True))  # accounts for the PSF to be placed correctly
    o = o/np.max(o)
    return o.astype('complex64')

# this should go into the NanoImagingPack
def convROTF(img,otf): # should go into nip
    '''
        convolves with a half-complex OTF, which can be generated using PSF2ROTF
    '''
    return irft(rft(img,shift_before=False,shift=False) * nip.expanddim(otf,img.ndim),shift_before=False,shift=False);
