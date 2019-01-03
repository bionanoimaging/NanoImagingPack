# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:33:03 2018

@author: ckarras
"""
from .util import get_type;

class THORLABS():
    '''
        Controll the Thorlabs PM100 Power meter
        requires packages:
            pyvisa (pip install pyvisa)
            ThorlabsPM100 (pip install ThorlabsPM100)
        
        wavelength: Which wavelength
        meas_range: Measurement range (e.g. 'auto')
        resource: instrument address (e.g. 'USB0::0x1313::0x8078::P0015032::INSTR')

        methods:
                update(wavelength, meas_range)
                read()
                close()

    '''    
    
    def __init__(self, wavelength = 'DEFAULT', resource = 'DEFAULT'):
        import pyvisa as visa;
        from ThorlabsPM100 import ThorlabsPM100;
        from .config import __DEFAULTS__;
    
        if resource == 'DEFAULT':
            resource = __DEFAULTS__['PM100_RESOURCE'];
        
        self.rm = visa.ResourceManager()
        inst = self.rm.open_resource(resource)
        self.dev = ThorlabsPM100(inst = inst)
        self.dev.sense.power.dc.range.auto = 1;    # set auto range
        self.update(wavelength);
        
    def update(self,wavelength):
        from .config import __DEFAULTS__;
        if wavelength == 'DEFAULT':
            wavelength = __DEFAULTS__['PM100_WAVELENGTH'];
            self.dev.sense.correction.wavelength = wavelength;
        elif wavelength is None:
            pass;
        elif wavelength > self.dev.sense.correction.maximum_wavelength:
            print('WARNING: wavelength to big -> set it to maximum');
            wavelength = self.dev.sense.correction.maximum_wavelength;
            self.dev.sense.correction.wavelength = wavelength;
        elif wavelength < self.dev.sense.correction.minimum_wavelength:
            print('WARNING: wavelength to small -> set it to minimum');
            wavelength = self.dev.sense.correction.minimum_wavelength;
            self.dev.sense.correction.wavelength = wavelength;
        else:
            self.dev.sense.correction.wavelength = wavelength;
     
    def read(self):
        return(self.dev.read);
        
    def close(self):
        self.rm.close();

class HAMAMATSU_SLM():
    
    
    '''
        Provides support for sending images to the Hamamatsu LCOS spatial light modulator;
    
        dll_path:           Path of the "LCosReg.dll" file (excluding file name)
        OVERDRIVE:          Do you use overdrive mode?
        use_corr_pattern:   Do you use the correction pattern from Hamamatsu?
        wavelength:         Which wavelength are you using? (requires string, e.g. '488')
        corr_pattern_path:  Path of the correction pattern files (excluding file names)
        
        
    
    '''
    
    def __init__(self, dll_path = None, OVERDRIVE = None, use_corr_pattern = None, wavelength = None, corr_pattern_path = None):
        from .config import __DEFAULTS__;
        import ctypes as ct;
        import numpy as np;
        import os.path        
        if dll_path == None:
            dll_path = __DEFAULTS__['LCOS_DLL_PATH'];
        
        if os.path.isfile(dll_path+'LcosReg.dll') == False:
            print('dll not found -> LcosReg.dll should be in '+dll_path);
        LIB = ct.cdll.LoadLibrary(dll_path+'LcosReg.dll');  # Links to dll    
        
        self.modulation_correction = [(400,93), (410,101), (420,107), (430,112), (440,117), (450,121), (460,126), 
                                      (470,131), (480,135), (488,139), (490,140), (500,144), (510,149), (520,153), 
                                      (530,157), (532,158), (540,161), (550,164), (560,168), (570,172), (580,175), 
                                      (590,179), (600,183), (610,187), (620,191), (630,196), (633,197), (640,200), 
                                      (650,204), (660,208), (670,213), (680,217), (690,221), (700,225)
                                      ]
                                      
        self.WRITE = LIB.LR_WriteDDR3;
        self.WRITE.argtypes =[ct.c_int32, ct.POINTER(ct.c_uint8),ct.c_int32];    # Define Prototype of dll function -> optionally use WRITE.restype = ct.c_int32 for defining the resutlts type
        self.WRITE.restype = ct.c_int32;
        if OVERDRIVE == None:
            self.overdrive = __DEFAULTS__['LCOS_OVERDRIVE'];
        else:
            self.overdrive = OVERDRIVE;
        if use_corr_pattern == None:
            self.use_corr_pat = __DEFAULTS__['LCOS_USE_CORR_PATTERN'];
        else:
            self.use_corr_pat = use_corr_pattern;
        if corr_pattern_path == None:
            corr_pattern_path = __DEFAULTS__['LCOS_CORR_PATTERN_PATH'];
        if wavelength == None:
            self.wavelength = __DEFAULTS__['LCOS_DEFAULT_WAVELENGTH'];
        else:
            self.wavelength = wavelength;
    
        wl_dist = [];
        for i in self.modulation_correction:
            wl_dist.append(np.abs(i[0]-int(self.wavelength)));
        self.corr_fac = self.__get_corr_factor__();
        if min(wl_dist) != 0:
            self.wavelength = str(self.modulation_correction[wl_dist.index(min(wl_dist))][0])
            print('Waring: Wavelength not possible -> taking '+self.wavelength);
                                          
        if self.use_corr_pat:
            from .image import readim, cat;
            import numpy as np;
            self.corr_img = cat([readim(corr_pattern_path+'CAL_LSH0701847_'+self.wavelength+'nm.bmp'), np.zeros((8,600))],0).astype(np.uint8);
            
            print('Correction pattern read :  '+corr_pattern_path+'CAL_LSH0701847_'+self.wavelength+'nm.bmp');
            print('Correction factor is '+str(self.corr_fac));
        else: 
            self.corr_img =0;
            self.corr_fac = 1;

    def __get_corr_factor__(self):
        wavelength = int(self.wavelength);
        for i in range(len(self.modulation_correction)):
            if self.modulation_correction[i][0] == wavelength:
                return(self.modulation_correction[i][1]);
            elif i < len(self.modulation_correction)-1:
                if (self.modulation_correction[i][0] < wavelength) and (self.modulation_correction[i+1][0] > wavelength):
                    m = (self.modulation_correction[i+1][1]-self.modulation_correction[i][1])/(self.modulation_correction[i+1][0]-self.modulation_correction[i][0]);
                    n = self.modulation_correction[i][1]-m*self.modulation_correction[i][0]
                    return(m*wavelength+n);
        print('Modulation correction not found!!')
    
    def clip_correction(self):
        
        self.corr_img= self.corr_img*280/255;
        self.corr_img[(self.corr_img<128) & (self.corr_img >115)] = 115;
        self.corr_img[(self.corr_img>=128) & (self.corr_img < 140)] = 140;
        return(self.corr_img)
    
    
    def send_dat(self, im, im_number):
        '''
            send image to slm:
                im           image array (shape: 800 X 600, dType: uint8);
                im_number:   address: (0...255)
        '''
        import numpy as np;
        if im.dtype != np.uint8:
                print('WARNING: Wrong data type! uint8 required! currently is '+ str(im.dtype)+' ... RECASTING AS UINT8');
                im = im.astype(np.uint8);
        if self.use_corr_pat:
            self.corr_img =self.corr_img*1;
            im = np.mod((im.astype(np.int32)+self.corr_img.astype(np.int32)),255);
            #im = np.mod((im.astype(np.int32)+self.clip_correction().astype(np.int32)),280);
            im = (im*self.corr_fac)/255;
            im = im.astype(np.uint8);
        im = im.transpose();
        if im.shape != (600,800):
            print('WRONG IMAGE SHAPE! 800X600 is required');
            ret = 0;
            ret2 = 0;
        elif (im_number <0) or (im_number > 255):
            print('Wrong image address! -> Must be between 0 and 255');
            ret = 0;
            ret2 = 0;
        else:
            import ctypes as ct;
            addr = im_number*240000;
            self.send_img = im;
            im = im.reshape((800*600,1));
            PTR = im.ctypes.data_as(ct.POINTER(ct.c_uint8))   # transform numpy array to pointer to uint8 byte    
            ret = self.WRITE(addr, PTR, im.size)    # hand over function -> im.size// 2 because image is 8 bit, but addresses are 16 bit
            if self.overdrive:    
                #ret2 = self.WRITE(addr+67108864, PTR, im.size)    # twice, since you have to write also on the second DDR3 Ram  for Overdrive
                ret2 = self.WRITE(addr+0x4000000, PTR, im.size)
        return(ret*ret2);
    
    def send_dat_to_block(self, im, im_number, block):
        '''
            send image to slm:
                im           image array (shape: 800 X 600, dType: uint8);
                im_number:   address: (0...255)
                block:       RAM - BLOCK (can be 0 or 1);
        '''
        import numpy as np;
        if im.dtype != np.uint8:
                print('WARNING: Wrong data type! uint8 required! currently is '+ str(im.dtype)+' ... RECASTING AS UINT8');
                im = im.astype(np.uint8);
        if self.use_corr_pat:
            im = np.mod((im.astype(np.int32)+self.corr_img.astype(np.int32)),256);
            im = im*self.corr_fac/255;
            im = im.astype(np.uint8);
        if (block != 0) and (block != 1):
            print('Wrong RAM-Block: Give 0 or 1');
            return(0);
        im = im.transpose();
        if im.shape != (600,800):
            print('WRONG IMAGE SHAPE! 800X600 is required');
            ret = 0;
            
        elif (im_number <0) or (im_number > 255):
            print('Wrong image address! -> Must be between 0 and 255');
            ret = 0;
            
        else:
            import ctypes as ct;
            addr = im_number*240000;
            im = im.reshape((800*600,1));
            PTR = im.ctypes.data_as(ct.POINTER(ct.c_uint8))   # transform numpy array to pointer to uint8 byte    
            ret = self.WRITE(addr+67108864*block, PTR, im.size)    # twice, since you have to write also on the second DDR3 Ram  for Overdrive
        return(ret);




    def set_zero(self, addresses = None):
        '''
            erases images at addresses
                addresses:which addresses you want to erase? 
                    if None (default) everything will be erased!
        '''
        import numpy as np;
        im = np.zeros((800,600)).astype(np.uint8);
        if addresses == None:
            addresses = range(256);
        r = 1;
        for a in addresses:
            print('Zeroing addres number '+str(a)+' ...');
            r *= self.send_dat(im, a);
        return(r);