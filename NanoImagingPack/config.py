# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:27:02 2018

@author: ckarras
"""

'''
Basic settings:
'''
from .FileUtils import str_to_path;
import os;



'''
    Do you want to use rfft for noncomplex inputs?
'''
__RFFT__ = True           # DEPRICATE


__DEFAULTS__ ={
        # General default settings
        'CUDA' : False,                    # use cuda for computations -> Has to be implemented
        'DEBUG':6,                         # Debugmessage level (0-10): use it with config.DBG_MSG(value) for controll, when which message should be displayed
        'DIRECTORY': os.getcwd(),          # Default data directory 
        'JAVA': False,                     # Java support
        'LOOKFOR_RATIO' : 0.5,              # How is the lookfor function behaving? -> between 0 and 1 -> higher numbers indicate that a higher agreement between input and output is required
        'ARRAY_RETURN_TYPE': 'image',    # return type of arrays in functions -> 'image' -> returns image, properties might be adapted from input image, 'ndarray' -> numpy array, 'asInput' -> like input
        
        # HAMAMATSU_SLM_SETTINGS:
        'LCOS_OVERDRIVE' : True,
        'LCOS_USE_CORR_PATTERN' : True,
        'LCOS_DLL_PATH' : str_to_path(r'C:\HamamatsuSLM'),  # Path of the Lcos_Reg.dll -> The hpkusbd2.dll has to be in the correct windows folder
        'LCOS_CORR_PATTERN_PATH':str_to_path(r'Z:\FastSIM_Setup\Hamamatsu_SLM_Steuerung\deformation_correction_pattern'),
        'LCOS_DEFAULT_WAVELENGTH' : '488',
        
        #ThorlabsPM100_Settings:
        'PM100_WAVELENGTH' : 500,
        'PM100_RESOURCE' : 'USB0::0x1313::0x8078::P0015032::INSTR',    # Figure this out wiht the help of visa resourc manager
        
        #Viewer_Default_Settings
        'VIEWER_MAX_CLICK_TIME': 0.5,             # Click time for recognizing marker
        'VIEWER_FORMAT_VALUE_STRING':'%1.4f',      # Format value for satus bar
        'VIEWER_DOWNCONVERT_DATA':False,          # downconverting float or complex data in viewer (speeds up stuff)
        'VIEWER_RATIO_AUTO_ASPECT':  20,           # At which ratio between the both axes for the vier the autoaspect should be used?
        'VIEWER_ASPECT':'AUTO',#'IMAGE',                    # How is the aspect ratio for the viewer? 'AUTO'- Automatic (pixel are squares, and if aspcec is too large, than the image is squared), 'CONST'  as statet in const, 'IMAGE' : according to image pixel size
        'VIEWER_CONST_ASPECT': 1,
        'VIEWER_GAMMA_MAX': 10.0,          # Min max values of the gamma slider in the viewer
        'VIEWER_GAMMA_MIN': 0.1,
        'VIEWER_GAMMA_INIT_COMPLEX' : 0.2,   # Gamma value for complex values
        'VIEWER_IMG_ORIGIN': 'upper',       #where is the origin of the image? 'upper' or 'lower' for upper or lower left corner
         
        #Image-Class Default Settings
        #'IMG_PIXELSIZES': [50,50,100],      #Default pixelsizes (list or tuple -> has to be at least 3dimensional)
        'IMG_PIXELSIZES': [100,50, 50],      #Default pixelsizes (list or tuple -> has to be at least 3dimensional)
        'IMG_PIXEL_UNITS': 'nm',             # Default units of image pixelsizes
        'IMG_TIFF_FORMATS': ['tif', 'tiff','TIF', 'TIFF'],
        'IMG_ZEISS_FORMATS': ['czi', 'CZI'],
        'IMG_VIEWER': 'VIEW5D',  # RH 3.2.19         # Default viewer -> currently only implemented viewr -> later also view5D, currently allwowd 'NIP_VIEW', 'INFO'
        'IMG_SQUEEZE_ZEISS': True,         # Do you want to squeeze zeiss files? otherwise theyhave 9 dimensions
        'IMG_NUMBERING': False,     # image numbering -> switch off for Debuging!
        
        'EXTRACT_EXTEND':True,           # should an extracted image be padded with zeros or not?
      
        #PSF_OTF_DEFAULTS  -> also Defaults for the Transfer class
        'TRANSFER_NA' : 0.8,
        'TRANSFER_n' :   1.0,
        'TRANSFER_wavelength': 500,
        'TRANSFER_FOCAL_LENGTH': 10,
        'TRANSFER_POLARIZATION': 0,           # Polariziation of the input can be number (angle in degree) or 'lin','lin_x', 'lin_y', 'circ', None
        'TRANSFER_VECTORIAL': True,           # Vectorial description?
        'TRANSFER_RESAMPLE': True,            # Resample images for efficient computation?
        'TRANSFER_APLANATIC': 'illumination', # aplanatic factor ('illumination', 'detection', None)
        'TRANSFER_FOC_ FIELD_MODE': 'theoretical', # how to compute the focal field? theoretical -> from sinc, circular -> circular mask for ctf
        'TRANSFER_NORM': 'max',                   # normalization of the transfer functions ('max' -> norm to maximum value, 'sum' -> norm to sum)
        
        
        #FT Default settings
        'FT_NORM': None,           # Normalization of FT: if None -> zero frequency ft strengths contains all image pixels, if 'ortho' -> the ft and the ift have the same scaling factor (but than the zero freq. contains only sqrt(pixelsumme))
        'FT_SHIFT':True,          # Shift (AFTER TRANSFORMATION)
        'FT_SHIFT_FIRST':False,    # Shift (BEFORE TRANSFORMATION)
        'FT_RETURN': 'complex',    # Return of the FTs (string values: complex , abs , phase, real, imag, default)
        'FT_REAL_RETURN':None,   # fill up real return to full spectrum? ('full' or None)
        #RFT Default settings   ----- RH 22.12.2018
        'RFT_NORM': None,           # Normalization of FT: if None -> zero frequency ft strengths contains all image pixels, if 'ortho' -> the ft and the ift have the same scaling factor (but than the zero freq. contains only sqrt(pixelsumme))
        'RFT_SHIFT':False,          # Shift (AFTER TRANSFORMATION)
        'RFT_SHIFT_FIRST':False,    # Shift (BEFORE TRANSFORMATION)
        'RFT_RETURN': 'complex',    # Return of the FTs (string values: complex , abs , phase, real, imag, default)
        'RFT_REAL_RETURN':None,   # fill up real return to full spectrum? ('full' or None)
        # Same like above but for IFT
        'IFT_NORM': None,           # Normalization of FT: if None -> zero frequency ft strengths contains all image pixels, if 'ortho' -> the ft and the ift have the same scaling factor (but than the zero freq. contains only sqrt(pixelsumme))
        'IFT_SHIFT':False,          # Shift (AFTER TRANSFORMATION)
        'IFT_SHIFT_FIRST':True,    # Shift (BEFORE TRANSFORMATION)
        'IFT_RETURN': 'complex',    # Return of the FTs (string values: complex , abs , phase, real, imag, default)
        'IFT_REAL_AXIS': 'GLOBAL',   # Which real axis to take for the ift? ('GLOBAL' as stored in global variable. That is useful if the rft was performed before, since there the real axis is defined), None -> the last axis is taken, value -> as stated in value
        # Same like above but for IRFT
        'IRFT_NORM': None,           # Normalization of FT: if None -> zero frequency ft strengths contains all image pixels, if 'ortho' -> the ft and the ift have the same scaling factor (but than the zero freq. contains only sqrt(pixelsumme))
        'IRFT_SHIFT':False,          # Shift (AFTER TRANSFORMATION)
        'IRFT_SHIFT_FIRST':False,    # Shift (BEFORE TRANSFORMATION)
        'IRFT_RETURN': 'complex',    # Return of the FTs (string values: complex , abs , phase, real, imag, default)
        'IRFT_REAL_AXIS': 'GLOBAL',   # Which real axis to take for the ift? ('GLOBAL' as stored in global variable. That is useful if the rft was performed before, since there the real axis is defined), None -> the last axis is taken, value -> as stated in value
        # correlator
        'CC_ABS_RETURN': True      # if true, the absolute value will be returned when correlating real images
        
        
        }

def DBG_MSG(text, level):
    if level < __DEFAULTS__['DEBUG']:
        print(text);

'''
    DO YOU WANNA USE THE FFTW??? -> ONLY IF INSTALLED!
'''
try:
    import pyFFTW;
    __FFTW__ = True;
except ImportError:
    __FFTW__ = False;

    
def set_cfg():
    try:
        import pyFFTW;
        __FFTW__ = True;
    except ImportError:
        __FFTW__ = False;
    if (__DEFAULTS__['IMG_VIEWER'] == 'VIEW5D'):
        try:
            import jnius_config
            import jnius as jn
        except ImportError:
            print("WARNING! Image viewer View5D could not be used as a default, since pyjnius is not properly installed. Reverting to NIP_VIEW as the default.")
            __DEFAULTS__['IMG_VIEWER'] ='NIP_VIEW'
            
    if len(__DEFAULTS__['IMG_PIXELSIZES']) <3:
        print('WARNING: Default pixelsize is not 3 dimensional add default values of 100');
        __DEFAULTS__['IMG_PIXELSIZES'] +=[100 for i in range(3-len(__DEFAULTS__['IMG_PIXELSIZES']))];
#        print('Importing pyFFTW failed!');
#        print('-> TAKING FFT from numpy!');