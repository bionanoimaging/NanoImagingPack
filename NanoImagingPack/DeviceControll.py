# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:33:03 2018

@author: ckarras
"""

import numpy as np
import ctypes as ct
import os.path
from .config import __DEFAULTS__
from .image import readim, cat
from .image import image as IMAGE

class THORLABS():
    """
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

    """
    
    def __init__(self, wavelength = 'DEFAULT', resource = 'DEFAULT'):
        import pyvisa as visa
        from ThorlabsPM100 import ThorlabsPM100

        if resource == 'DEFAULT':
            resource = __DEFAULTS__['PM100_RESOURCE']

        self.rm = visa.ResourceManager()
        inst = self.rm.open_resource(resource)
        self.dev = ThorlabsPM100(inst = inst)
        self.dev.sense.power.dc.range.auto = 1  # set auto range
        self.update(wavelength)

    def update(self,wavelength):
        if wavelength == 'DEFAULT':
            wavelength = __DEFAULTS__['PM100_WAVELENGTH']
            self.dev.sense.correction.wavelength = wavelength
        elif wavelength is None:
            pass;
        elif wavelength > self.dev.sense.correction.maximum_wavelength:
            print('WARNING: wavelength to big -> set it to maximum')
            wavelength = self.dev.sense.correction.maximum_wavelength
            self.dev.sense.correction.wavelength = wavelength
        elif wavelength < self.dev.sense.correction.minimum_wavelength:
            print('WARNING: wavelength to small -> set it to minimum')
            wavelength = self.dev.sense.correction.minimum_wavelength
            self.dev.sense.correction.wavelength = wavelength
        else:
            self.dev.sense.correction.wavelength = wavelength

    def read(self):
        return(self.dev.read)

    def close(self):
        self.rm.close()


class HAMAMATSU_SLM():


    """
        Provides support for sending images to the Hamamatsu LCOS spatial light modulator;

        dll_path:           Path of the "LCosReg.dll" file (excluding file name)
        OVERDRIVE:          Do you use overdrive mode?
        use_corr_pattern:   Do you use the correction pattern from Hamamatsu?
        wavelength:         Which wavelength are you using? (requires string, e.g. '488')
        corr_pattern_path:  Path of the correction pattern files (excluding file names)



    """
    
    def __init__(self, dll_path = None, OVERDRIVE = None, use_corr_pattern = None, wavelength = None, corr_pattern_path = None):
        if dll_path == None:
            dll_path = __DEFAULTS__['LCOS_DLL_PATH']

        if os.path.isfile(dll_path+'LcosReg.dll') == False:
            print('dll not found -> LcosReg.dll should be in '+dll_path)
        LIB = ct.cdll.LoadLibrary(dll_path+'LcosReg.dll')  # Links to dll
        
        self.modulation_correction = [(400,93), (410,101), (420,107), (430,112), (440,117), (450,121), (460,126), 
                                      (470,131), (480,135), (488,139), (490,140), (500,144), (510,149), (520,153), 
                                      (530,157), (532,158), (540,161), (550,164), (560,168), (570,172), (580,175), 
                                      (590,179), (600,183), (610,187), (620,191), (630,196), (633,197), (640,200), 
                                      (650,204), (660,208), (670,213), (680,217), (690,221), (700,225)
                                      ]
                                      
        self.WRITE = LIB.LR_WriteDDR3
        self.WRITE.argtypes =[ct.c_int32, ct.POINTER(ct.c_uint8),ct.c_int32]  # Define Prototype of dll function -> optionally use WRITE.restype = ct.c_int32 for defining the resutlts type
        self.WRITE.restype = ct.c_int32
        if OVERDRIVE == None:
            self.overdrive = __DEFAULTS__['LCOS_OVERDRIVE']
        else:
            self.overdrive = OVERDRIVE
        if use_corr_pattern == None:
            self.use_corr_pat = __DEFAULTS__['LCOS_USE_CORR_PATTERN']
        else:
            self.use_corr_pat = use_corr_pattern
        if corr_pattern_path == None:
            corr_pattern_path = __DEFAULTS__['LCOS_CORR_PATTERN_PATH']
        if wavelength == None:
            self.wavelength = __DEFAULTS__['LCOS_DEFAULT_WAVELENGTH']
        else:
            self.wavelength = wavelength

        wl_dist = []
        for i in self.modulation_correction:
            wl_dist.append(np.abs(i[0]-int(self.wavelength)))
        self.corr_fac = self.__get_corr_factor__()
        if min(wl_dist) != 0:
            self.wavelength = str(self.modulation_correction[wl_dist.index(min(wl_dist))][0])
            print('Waring: Wavelength not possible -> taking '+self.wavelength)

        if self.use_corr_pat:
            self.corr_img = cat([readim(corr_pattern_path+'CAL_LSH0701847_'+self.wavelength+'nm.bmp').transpose(), np.zeros((8,600))],0).astype(np.uint8)
            print('Correction pattern read :  '+corr_pattern_path+'CAL_LSH0701847_'+self.wavelength+'nm.bmp')
            print('Correction factor is '+str(self.corr_fac))
        else: 
            self.corr_img =0
            self.corr_fac = 1

    def __get_corr_factor__(self):
        wavelength = int(self.wavelength)
        for i in range(len(self.modulation_correction)):
            if self.modulation_correction[i][0] == wavelength:
                return(self.modulation_correction[i][1])
            elif i < len(self.modulation_correction)-1:
                if (self.modulation_correction[i][0] < wavelength) and (self.modulation_correction[i+1][0] > wavelength):
                    m = (self.modulation_correction[i+1][1]-self.modulation_correction[i][1])/(self.modulation_correction[i+1][0]-self.modulation_correction[i][0])
                    n = self.modulation_correction[i][1]-m*self.modulation_correction[i][0]
                    return(m*wavelength+n)
        print('Modulation correction not found!!')
    
    def clip_correction(self):
        
        self.corr_img= self.corr_img*280/255
        self.corr_img[(self.corr_img<128) & (self.corr_img >115)] = 115
        self.corr_img[(self.corr_img>=128) & (self.corr_img < 140)] = 140
        return(self.corr_img)
    
    
    def send_dat(self, im, im_number):
        """
            send image to slm:
                im           image array (shape: 800 X 600, dType: uint8);
                im_number:   address: (0...255)
        """
        
        import numpy as np
        if im.dtype != np.uint8:
                print('WARNING: Wrong data type! uint8 required! currently is '+ str(im.dtype)+' ... RECASTING AS UINT8')
                im = im.astype(np.uint8)
        if self.use_corr_pat:
            self.corr_img =self.corr_img*1
            im = np.mod((im.astype(np.int32)+self.corr_img.astype(np.int32)),255)
            #im = np.mod((im.astype(np.int32)+self.clip_correction().astype(np.int32)),280);
            im = (im*self.corr_fac)/255
            im = im.astype(np.uint8)
        im = im.transpose()
        if im.shape != (600,800):
            print('WRONG IMAGE SHAPE! 800X600 is required')
            ret = 0
            ret2 = 0
        elif (im_number <0) or (im_number > 255):
            print('Wrong image address! -> Must be between 0 and 255')
            ret = 0
            ret2 = 0
        else:
            import ctypes as ct
            addr = im_number*240000
            self.send_img = im
            im = im.reshape((800*600,1))
            PTR = im.ctypes.data_as(ct.POINTER(ct.c_uint8))   # transform numpy array to pointer to uint8 byte    
            ret = self.WRITE(addr, PTR, im.size)    # hand over function -> im.size// 2 because image is 8 bit, but addresses are 16 bit
            if self.overdrive:    
                #ret2 = self.WRITE(addr+67108864, PTR, im.size)    # twice, since you have to write also on the second DDR3 Ram  for Overdrive
                ret2 = self.WRITE(addr+0x4000000, PTR, im.size)
        return(ret*ret2)

    def send_dat_to_block(self, im, im_number, block):
        """
            send image to slm:
                im           image array (shape: 800 X 600, dType: uint8);
                im_number:   address: (0...255)
                block:       RAM - BLOCK (can be 0 or 1);
        """
        import numpy as np
        if im.dtype != np.uint8:
                print('WARNING: Wrong data type! uint8 required! currently is '+ str(im.dtype)+' ... RECASTING AS UINT8')
                im = im.astype(np.uint8)
        if self.use_corr_pat:
            im = np.mod((im.astype(np.int32)+self.corr_img.astype(np.int32)),256)
            im = im*self.corr_fac/255
            im = im.astype(np.uint8)
        if (block != 0) and (block != 1):
            print('Wrong RAM-Block: Give 0 or 1')
            return(0)
        im = im.transpose()
        if im.shape != (600,800):
            print('WRONG IMAGE SHAPE! 800X600 is required')
            ret = 0

        elif (im_number <0) or (im_number > 255):
            print('Wrong image address! -> Must be between 0 and 255')
            ret = 0

        else:
            import ctypes as ct
            addr = im_number*240000
            im = im.reshape((800*600,1))
            PTR = im.ctypes.data_as(ct.POINTER(ct.c_uint8))   # transform numpy array to pointer to uint8 byte    
            ret = self.WRITE(addr+67108864*block, PTR, im.size)    # twice, since you have to write also on the second DDR3 Ram  for Overdrive
        return(ret)

    def set_zero(self, addresses = None):
        """
            erases images at addresses
                addresses:which addresses you want to erase?
                    if None (default) everything will be erased!
        """
        import numpy as np
        im = np.zeros((800,600)).astype(np.uint8)
        if addresses == None:
            addresses = range(256)
        r = 1
        for a in addresses:
            print('Zeroing addres number '+str(a)+' ...')
            r *= self.send_dat(im, a)
        return(r)


class OrcaFlash():
    '''
        This class handles readout of the Orca Flash 4.0V2 Camera via the FireBird Camera Link cable and a Cuda GPU

        If no Camera is provided, test images will be used, however, CUDA and CUPY are necessary for running the package!

        To Check if camera works run the following test script in console (cmd):

            import... # Import your stuff here

            c = OrcaFlash()
            c.uninit();


        It should depict the Camera information
    '''

    def __init__(self, sample_image=None, gpu_buffersize=None, exposure=None, roi=None, cooler=None, trigger=None,
                 trigger_out=None):
        '''
            Paras (If None, '__DEFAULTS__' values wil be used):
                load_from_file:   if this is None, the camera will be used. If no camera present, a default image file will be loaded. If a file path or a nip.image is given, that image will be read instead of the camera image.

                gpu_buffersize:    how many images to store on gpu buffer?
                exposure:          Exposuretime in sec
                roi               [x_min, x_max, y_min, y_max] pixels
                cooler:            set up cooler at startup
                trigger:          Dictionary how camera is triggered
                    mode:
                        0 Internal
                        1 Edge trigger: Externally triggered by edge (set via polarity if up or down)
                        2 Level Trigger: Externally triggered by level (set via polarity if low or high)
                        3 Trigger starts internal trigger mode (known as "start")
                        4 Sync readout: refere to manual -> readout starts after n trigger counts (set by counts) -> exposure time determined by trigger counts

                    delay:
                        delay in seconds of the readout
                    polarity
                        1 negative
                        2 positive

                trigger_out:     Set the output trigger of the Camera: LIST OF 3 ELEMENTS of dictionary
                    kind:
                        1 LOW
                        2 EXPOSURE
                        3 PROGRAMMABLE
                        4 TRIGGER READY
                        5 HIGH
                    polarity
                        1 negative
                        2 positive
                    delay
                    source
                        1 Reaout_END
                        2 VSYNC (Should be default)



                    readoutime
                        Returns readout time in seconds
                    min_trigger_blanking
                        Returns period from the end of exposure to trigger ready in seconds.

        '''
        import cupy as cp;  # actually not required, but throws error if cupy is not installed


        self.LIB = ct.cdll.LoadLibrary(__DEFAULTS__['ORCUDA_DLL_PATH']);  # Links to dll
        LIB = self.LIB;
        self.CAM_INIT = LIB.CU_init_cam;
        self.CAM_INIT.argtypes = [];  # Define Prototype of dll function -> optionally use WRITE.restype = ct.c_int32 for defining the resutlts type
        self.CAM_INIT.restype = ct.c_int32;

        self.CAM_UNINIT = LIB.CU_uninit_cam;
        self.CAM_UNINIT.argtypes = [ct.c_int32];

        self.CAM_ROI = LIB.CU_set_ROI;
        self.CAM_ROI.restype = ct.c_int32;
        self.CAM_ROI.argtypes = [ct.c_longlong, ct.c_int, ct.c_int, ct.c_int, ct.c_int];

        self.CAM_CLEAR_ROI = LIB.CU_clear_ROI;
        self.CAM_CLEAR_ROI.restype = ct.c_int32;
        self.CAM_CLEAR_ROI.argtypes = [ct.c_longlong];

        self.CAM_EXP_TIME = LIB.CU_set_expTime_sec;
        self.CAM_EXP_TIME.restype = ct.c_int32;
        self.CAM_EXP_TIME.argtypes = [ct.c_longlong, ct.c_double];

        self.CAM_COOLER = LIB.CU_set_cooler;
        self.CAM_COOLER.restype = ct.c_int32;
        self.CAM_COOLER.argtypes = [ct.c_longlong, ct.c_int];

        self.CAM_TRIGGER = LIB.CU_setup_trigger;
        self.CAM_TRIGGER.restype = ct.c_int32;
        self.CAM_TRIGGER.argtypes = [ct.c_longlong, ct.c_int, ct.c_double, ct.c_int, ct.c_int, ct.POINTER(ct.c_double),
                                     ct.POINTER(ct.c_double)];

        self.CAM_TRIGGER_OUT = LIB.CU_set_ouptput_trigger;
        self.CAM_TRIGGER_OUT.restype = ct.c_int32;
        self.CAM_TRIGGER_OUT.argtypes = [ct.c_longlong, ct.c_int, ct.c_int, ct.c_int, ct.c_double, ct.c_int];

        self.CAM_STATUS = LIB.CU_get_status;
        self.CAM_STATUS.restype = ct.c_int32;
        self.CAM_STATUS.argtypes = [];

        self.CAM_BUFFER_SIZE = LIB.CU_set_max_frames_buffer;
        self.CAM_BUFFER_SIZE.restype = None;
        self.CAM_BUFFER_SIZE.argstypes = [ct.c_int32];

        self.CAM_START = LIB.CU_start_capture;
        self.CAM_START.argtypes = [ct.c_longlong, ct.c_int32];
        self.CAM_START.restype = ct.c_int32;

        self.CAM_STOP = LIB.CU_stop_capture;
        self.CAM_STOP.argtypes = [ct.c_uint32]

        self.CAM_SET_DEV_PTR = LIB.CU_set_dev_ptr;
        self.CAM_SET_DEV_PTR.restype = None;
        self.CAM_SET_DEV_PTR.argtypes = [ct.POINTER(ct.c_void_p), ct.c_int32];

        self.CAM_DISP_DEV_PTR = LIB.CU_show_img_ptr;
        self.CAM_DISP_DEV_PTR.restype = None;
        self.CAM_DISP_DEV_PTR.argtypes = [];

        self.CAM_GET_FRAME_INDEX = LIB.CU_get_frame_index;
        self.CAM_GET_FRAME_INDEX.restype = ct.c_int32;
        self.CAM_GET_FRAME_INDEX.argtypes = [];

        self.CAM_LOCK_MEM = LIB.CU_lock_mem;
        self.CAM_LOCK_MEM.restype = None;
        self.CAM_LOCK_MEM.argtypes = [];

        self.CAM_UNLOCK_MEM = LIB.CU_unlock_mem;
        self.CAM_UNLOCK_MEM.restype = None;
        self.CAM_UNLOCK_MEM.argtypes = [];

        self.CAM_FREE_DEV = LIB.CU_free_dev;
        self.CAM_FREE_DEV.restype = None;
        self.CAM_FREE_DEV.argtypes = [];

        self.CAM_IMG_DIM = LIB.CU_get_img_dims;
        self.CAM_IMG_DIM.restype = None;
        self.CAM_IMG_DIM.argtypes = [ct.c_longlong, ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32)];

        self.IMAGE_HEIGHT = 0;
        self.IMAGE_WIDTH = 0;

        self.MEM_LOCKED_STATUS = 0;

        self.__DEFAULT_RUN__ = False;  # Abort flag for Default images
        self.__DEF_FRAME_INDEX__ = 0;
        self.__FILL_ONE_BUFFER__ = 0;  # For filling the buffer once(run_once_flag)
        self.USE_CAMERA = True;

        if sample_image is None:
            print('Initializing Camera ...');
            self.cam_handle = self.CAM_INIT();
            if (self.CAM_STATUS() == 0):
                print('SUCCESS');
                self.DEF_IMG = None
                self.set_roi(roi, reset_mem=False);  # IMG_BYTE_SIZE defined in this function
                self.set_cooler(cooler);
                self.set_output_trigger(trigger_out);
                self.set_trigger(trigger);

            else:
                self.USE_CAMERA = False;
                print('FAILED  -  Camera Status is ' + str(self.CAM_STATUS()));
                self.DEF_IMG = readim(__DEFAULTS__['ORCUDA_DEFAULT_IMG']);
                self.IMAGE_WIDTH = self.DEF_IMG.shape[-1];
                self.IMAGE_HEIGHT = self.DEF_IMG.shape[-2];
                if self.DEF_IMG.ndim < 3:
                    self.DEF_IMG = self.DEF_IMG.expanddim(3);
                self.IMG_BYTE_SIZE = self.DEF_IMG.itemsize * self.DEF_IMG.shape[-1] * self.DEF_IMG.shape[-2];
        elif isinstance(sample_image, np.ndarray):
            self.cam_handle = -1;
            self.USE_CAMERA = False;
            self.DEF_IMG = sample_image;
            self.IMAGE_WIDTH = self.DEF_IMG.shape[-1];
            self.IMAGE_HEIGHT = self.DEF_IMG.shape[-2];
            if self.DEF_IMG.ndim < 3:
                self.DEF_IMG = self.DEF_IMG.expanddim(3);
            self.IMG_BYTE_SIZE = self.DEF_IMG.itemsize * self.DEF_IMG.shape[-1] * self.DEF_IMG.shape[-2];
        elif isinstance(sample_image, str):
            self.cam_handle = -1;
            self.USE_CAMERA = False;
            self.DEF_IMG = readim(__DEFAULTS__['ORCUDA_DEFAULT_IMG']);
            self.IMAGE_WIDTH = self.DEF_IMG.shape[-1];
            self.IMAGE_HEIGHT = self.DEF_IMG.shape[-2];
            if self.DEF_IMG.ndim < 3:
                self.DEF_IMG = self.DEF_IMG.expanddim(3);
            self.IMG_BYTE_SIZE = self.DEF_IMG.itemsize * self.DEF_IMG.shape[-1] * self.DEF_IMG.shape[-2];
        self.set_exposure(exposure);  # outside if because of exposure time for defaults images
        self.set_buffersize(gpu_buffersize, reset_mem=False);
        self.__prepare_memory__();

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.uninit();

        # Thread that feeds images to GPU

    def __feed_default_image__(self, memptrs, image, run_once, exposure_time):
        curr_ptr = 0;
        curr_img = 0;
        import time;
        if exposure_time < 0.01: exposure_time = 0.01;
        if exposure_time > 5: exposure_time = 5;
        im_byte_size = image.itemsize * image.shape[-1] * image.shape[-2];
        while (self.__DEFAULT_RUN__):
            #   print('current run:' +str(curr_ptr))
            MemPtr = memptrs[curr_ptr];
            CurrIm = image[curr_img];
            CurrImPtr = CurrIm.ctypes.data_as(ct.c_void_p);
            MemPtr.copy_from_host(CurrImPtr, im_byte_size);
            self.__DEF_FRAME_INDEX__ = curr_ptr;
            if curr_img == (image.shape[-3] - 1):
                curr_img = 0;
            else:
                curr_img += 1;
            if curr_ptr == (len(memptrs) - 1):
                if run_once != 0:
                    self.__DEFAULT_RUN__ = False;
                else:
                    curr_ptr = 0;
            else:
                curr_ptr += 1;
            time.sleep(exposure_time);

    def start_capture(self, run_once=0):
        '''
            Start capturing data in buffer

            If run_once = 1 only data will only captured until buffer is full otherwise buffer will be overwritten after each run (stop capturing than with stop_captue)
        '''
        import threading;
        self.__FILL_ONE_BUFFER__ = run_once;
        if (self.CAM_STATUS() >= 0) and self.USE_CAMERA:
            print('Capturing started');
            self.CAM_START(self.cam_handle, ct.c_int32(run_once));
        else:
            self.__DEFAULT_RUN__ = True;
            self.__RUN_DEFAULT_THREAD__ = threading.Thread(target=self.__feed_default_image__,
                                                           args=(self.MEMPTRS, self.DEF_IMG, run_once, self.EXP_TIME));
            self.__RUN_DEFAULT_THREAD__.start();

    def get_frame_nr(self):
        '''
            returns current frame number
        '''
        if (self.CAM_STATUS() >= 0) and self.USE_CAMERA:
            return (self.CAM_GET_FRAME_INDEX());
        else:
            return (self.__DEF_FRAME_INDEX__);

    def get_device_array(self, buffer_place_number, copy_away_from_buffer=False):
        '''
            This gives you access to an image in the device

            params:
                    buffer_place_number: Which array do you want to access? (between 0 and buffersize)
                    copy_away_from_buffer: If true array is copied away from buffer

            returns: cp.ndarray (image)
        '''
        import cupy as cp;
        if (self.CAM_STATUS() >= 0) and self.USE_CAMERA:    self.mem_lock(True);

        self.image = cp.ndarray((self.IMAGE_WIDTH, self.IMAGE_HEIGHT), dtype=cp.uint16,
                                memptr=self.MEMPTRS[buffer_place_number], order='C');
        if copy_away_from_buffer:   self.image = cp.copy(self.image);
        if (self.CAM_STATUS() >= 0):    self.mem_lock(False);
        return (self.image);

    def get_all_images(self):
        '''
            copies all images in the buffer to the host and cast them as nd image
        '''
        import cupy as cp;
        for i in range(self.BUFFERSIZE):
            print('Getting image number ... ' + str(i))
            if i == 0:
                # TODO: FIX NAME CONFLICT
                MyIm = IMAGE(cp.asnumpy(self.get_device_array(i, False)));
            else:
                MyIm = cat((MyIm, cp.asnumpy(self.get_device_array(i, False))), -3);
        return (MyIm);

    def force_stop(self):
        '''
            like stop, but doesn't wait for the buffer to be filled in case of run_once flag is set
        '''
        if (self.CAM_STATUS() >= 0) and self.USE_CAMERA:
            self.CAM_STOP(self.cam_handle);
        else:
            self.__DEFAULT_RUN__ = False;
            self.__RUN_DEFAULT_THREAD__.join();

    def stop_capture(self):
        '''
         stops the camera
        '''

        if (self.CAM_STATUS() >= 0) and self.USE_CAMERA:
            if (self.__FILL_ONE_BUFFER__ != 0):
                print(self.get_frame_nr());
                if self.get_frame_nr() == (self.BUFFERSIZE - 1):
                    self.CAM_STOP(self.cam_handle);
                else:
                    print('Waiting for image - framenumber is ' + str(self.get_frame_nr()));
            else:
                self.CAM_STOP(self.cam_handle);
        else:
            if (self.__FILL_ONE_BUFFER__ != 0):
                if self.get_frame_nr() == (self.BUFFERSIZE - 1):
                    self.__DEFAULT_RUN__ = False;
                else:
                    print('Waiting for image - framenumber is ' + str(self.get_frame_nr()));
            else:
                self.__DEFAULT_RUN__ = False;
            self.__RUN_DEFAULT_THREAD__.join();

    def mem_lock(self, lock):
        '''
            Lock or unlock memory of the dll's main thread using a mutex (! thats the thread that is not acquiring any data)
        '''
        if (self.CAM_STATUS() >= 0) and self.USE_CAMERA:
            if lock == True:
                self.CAM_LOCK_MEM();
                self.MEM_LOCKED_STATUS = True;
            else:
                self.CAM_UNLOCK_MEM();
                self.MEM_LOCKED_STATUS = False;

    def __prepare_memory__(self):
        '''
            prepare memory on gpu
        '''
        import cupy as cp;
        # Default image -> creates memory pointer list  in which images should be written
        Adress_list = [];
        self.MEMPTRS = []
        for i in range(self.BUFFERSIZE):  # allocate memory on gpu
            MemPtr = cp.cuda.alloc(self.IMG_BYTE_SIZE);
            self.MEMPTRS.append(MemPtr);  # collect pointers
            Adress_list.append(ct.c_void_p(MemPtr.ptr));  # collect adresses as c-pointers
        if (self.CAM_STATUS() == 0) and self.USE_CAMERA:
            Adress_array = (ct.c_void_p * len(Adress_list))(*Adress_list);
            self.free_device();
            print('setting dev_ptrs');
            self.CAM_SET_DEV_PTR(Adress_array, self.BUFFERSIZE);
            self.CAM_DISP_DEV_PTR();  # For debugging -> run in cmd

    def free_device(self):
        '''
            Free the allocated memory on the device (is allocated using set_buffersize)
        '''
        if (self.CAM_STATUS() == 0) and self.USE_CAMERA:
            self.CAM_FREE_DEV();
            print('Device pointers freed');
        # TODO: HANDLE DEFAULT IMAGE

    def set_trigger(self, trigger):
        '''
            set up the trigger
            args:
                trigger: dictionary: 'mode', 'delay', 'polarity', 'first_exposure', 'counts'
        '''
        if (self.CAM_STATUS() == 0) and self.USE_CAMERA:
            if trigger is None:
                trigger = __DEFAULTS__['ORCUDA_TRIGGER'];
            rt = ct.c_double();
            mtb = ct.c_double();
            self.CAM_TRIGGER(self.cam_handle, ct.c_int32(trigger['mode']), ct.c_double(trigger['delay']),
                             ct.c_int32(trigger['polarity']), ct.c_int32(trigger['counts']), ct.byref(rt),
                             ct.byref(mtb));
            self.READ_OUT_TIME = rt.value;
            self.MIN_TRIGGER_BLANK = mtb.value;
            self.TRIGGER = trigger;
            print('Trigger set');
            print('Readoutime [s]:               ' + str(self.READ_OUT_TIME));
            print('Minimum Trigger Blanking [s]: ' + str(self.MIN_TRIGGER_BLANK));

    def set_output_trigger(self, output_trigger):
        '''
            sets up the trigger outputs:
            list of three dictionaries (for the outputs) 'kind', 'polarity', 'delay', 'source'
        '''
        if (self.CAM_STATUS() == 0) and self.USE_CAMERA:
            if output_trigger is None:
                output_trigger = __DEFAULTS__['ORCDUA_TRIGGER_OUT'];
            for (i, to) in enumerate(output_trigger):
                self.CAM_TRIGGER_OUT(self.cam_handle, ct.c_int32(i), ct.c_int32(to['kind']), ct.c_int32(to['polarity']),
                                     ct.c_double(to['delay']), ct.c_int32(to['source']));
            print('Outputtrigger set');
            self.TRIGGER_OUT = output_trigger;

    def set_exposure(self, exposure_time_in_sec):
        '''
        set up exposure time in seconds
        '''
        if exposure_time_in_sec is None:
            exposure_time_in_sec = __DEFAULTS__['ORCUDA_EXP_TIME'];
        if (self.CAM_STATUS() == 0) and self.USE_CAMERA:
            self.CAM_EXP_TIME(self.cam_handle, ct.c_double(exposure_time_in_sec));
            print('Exposure time set to ' + str(exposure_time_in_sec) + ' s');
        # used for stopping feeding thread
        self.EXP_TIME = exposure_time_in_sec;

    def set_cooler(self, cooler_on):
        '''
            Switch cooler on or off
        '''
        if (self.CAM_STATUS() == 0) and self.USE_CAMERA:
            if cooler_on is None:
                cooler_on = __DEFAULTS__['ORCUDA_COOLER'];
            if cooler_on == True:
                self.CAM_COOLER(self.cam_handle, ct.c_int32(4));
                print('Set cooler on - MAKE SURE WATER IS ON!!!');
            else:
                self.CAM_COOLER(self.cam_handle, ct.c_int32(1));
                print('Set cooler off');
            self.COOLER = cooler_on;

    def set_buffersize(self, buffersize, reset_mem=True):
        '''
            setting up the buffersize on the gpu
            Also Memory is allocated! Don't forget to free it eventually!
            reset_mem: Prepare memory for data acquisition;
        '''
        if buffersize is None:
            buffersize = __DEFAULTS__['ORCUDA_BUFF_SIZE'];
        if (self.CAM_STATUS() == 0) and self.USE_CAMERA:
            self.CAM_BUFFER_SIZE(ct.c_int32(buffersize));
            print('Set buffersize to: ' + str(buffersize));
        self.BUFFERSIZE = buffersize;
        if reset_mem:
            self.__prepare_memory__();

    def set_roi(self, roi, reset_mem=True):
        '''
            set up roi for readout [x_min, x_max, y_min, y_max]

            !!! Attention: Roi must be multiples of 4 -> This is handled by the dll, but the actual roi might differ from the values given -> use get_img_dims for figuring out the actual image size
            reset_mem: Prepare memory for data acquisition;
        '''
        if (self.CAM_STATUS() == 0) and self.USE_CAMERA:
            if roi is None:
                roi = [__DEFAULTS__['ORCUDA_IMG_HORIZONTAL_START'], __DEFAULTS__['ORCUDA_IMG_HORIZONTAL_STOP'],
                       __DEFAULTS__['ORCUDA_IMG_VERTICAL_START'], __DEFAULTS__['ORCUDA_IMG_VERTICAL_STOP']]
            self.CAM_ROI(self.cam_handle, ct.c_int32(roi[0]), ct.c_int32(roi[2]), ct.c_int32(roi[1] - roi[0]),
                         ct.c_int32(roi[3] - roi[2]));
            print('Setting roi');
            self.get_img_dim();
            if reset_mem:
                self.__prepare_memory__();

    def clear_roi(self):
        '''
            clear roi
        '''
        if (self.CAM_STATUS() == 0) and self.USE_CAMERA:
            self.CAM_CLEAR_ROI(self.cam_handle);
            print('ROI cleared');
            self.get_img_dim();

    def get_img_dim(self):
        '''
            get image dimensions
        '''
        if (self.CAM_STATUS() == 0) and self.USE_CAMERA:
            h = ct.c_int();
            w = ct.c_int();
            self.CAM_IMG_DIM(self.cam_handle, ct.byref(h), ct.byref(w));
            self.IMAGE_HEIGHT = h.value;
            self.IMAGE_WIDTH = w.value;
        else:
            self.IMAGE_HEIGHT = self.DEF_IMG[-2];
            self.IMAGE_WIDTH = self.DEF_IMG[-1];
        print('Imgage Dimensions (H X W):' + str(self.IMAGE_HEIGHT) + ' X ' + str(self.IMAGE_WIDTH));
        self.IMG_BYTE_SIZE = 2 * self.IMAGE_HEIGHT * self.IMAGE_WIDTH;

    def uninit(self):
        '''
            uninit the camera
        '''
        if (self.CAM_STATUS != 0) and self.USE_CAMERA:
            self.stop_capture();
        print('Freeing memory');
        self.free_device();
        self.set_cooler(False);
        print('Detatching Camera...');
        self.CAM_UNINIT(self.cam_handle);
        libHandle = self.LIB._handle;
        del self.LIB;
        kernel32 = ct.WinDLL('kernel32', use_last_error=True);
        kernel32.FreeLibrary.argtypes = [ct.c_int64];
        kernel32.FreeLibrary(libHandle)
        print('Done')
