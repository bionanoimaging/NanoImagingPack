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
from .util import struct;
from enum import Enum;

class __pols__():
    lin = "lin";
    lin_x = "lin_x";
    lin_y = "lin_y";
    azimuthal = "azimuthal";
    radial = "radial";
    circular = "circular";
    elliptic = "elliptic";

class __aplanars__():
    excitation = "excitation";
    emission = "emission";
    excitation2 = "excitation2";
    emission2 = "emission2";
    no = None;

class __zernikes__():
    piston = "piston";
    tiltY='tiltY';
    tiltX='tiltX';
    astigm='astigm';
    defoc='defoc';
    vastig='vastig';
    vtrefoil='vtrefoil';
    vcoma= 'vcoma';
    hcoma = 'hcoma';
    obtrefoil ='obtrefoil';
    obquadfoil= 'obquadfoil';
    asti2nd = 'asti2nd';
    spheric = 'spheric';
    vasti2nd ='vasti2nd';
    vquadfoil='vquadfoil';

# DEFINE AVAILABLE PACKAGE STRUCT DEFAULTS HERE!
PSF_PARAMS = struct();
PSF_PARAMS.explanation= '\n NA                    Numerical aperture \n'
PSF_PARAMS.explanation+=' n                     refractive index \n'
#PSF_PARAMS.explanation+=' dimension             2 or 3 -> dimenstion of the Transfere function \n'
PSF_PARAMS.explanation+=' wavelenght            wavelength in units of the image pixel size \n'
PSF_PARAMS.explanation+=' pol                   polarization: give "lin","lin_x","lin_y", "azimuthal", "radial", "circular", "elliptic" or a tuple or list of polarization maps (has to be of x,y dim of the image, first element is x, second y polarization) \n'
PSF_PARAMS.explanation+=' pols                  polarization: Choose Polarization type from a list\n';
PSF_PARAMS.explanation+=' pol_xy_phase_shift    Only for elliptic polarization: Enter the phase shift between x and y in rad \n'
PSF_PARAMS.explanation+=' pol_lin_angle         Only for linear polarization: whats the angle \n'
PSF_PARAMS.explanation+=' vectorized            True or False -> Vectorial computation \n'
PSF_PARAMS.explanation+=' aplanar               "excitation", "emission","excitation2","emission2" None, aplanatic factor, 2 means squared aplanatic factor  \n'
PSF_PARAMS.explanation+=' apl                   choose aplanatic factor from a list  \n'
PSF_PARAMS.explanation+=' off_focal_distance    distance (in image units!) of the computed focal field to the focus  \n'
#PSF_PARAMS.explanation+=' Z_PXS_FOR_2D_IMG      ONLY IF 2D IMAGE IS USED, BUT 3D TRANSFER REQUIRED: what is the pixelsize in z \n'
#PSF_PARAMS.explanation+=' Z_STEPS_FOR_2D_IMG    ONLY IF 2D IMAGE IS USED, BUT 3D TRANSFER REQUIRED: Number of z-slices \n'
PSF_PARAMS.explanation+=' aberration_strength   For aberration -> refere to nip.set_aberration_map help \n'
PSF_PARAMS.explanation+=' aberration_types      For aberration -> refere to nip.set_aberration_map help \n'
PSF_PARAMS.explanation+=' aberration_zernikes   For aberration types -> choose Zernike polynomial from a list \n'
PSF_PARAMS.explanation+=' aperture_transmission For defining an aperture -> refere to nip.set_aberration_map help \n'
PSF_PARAMS.explanation+=' foc_field_method      How to compute the base focal field "theoretical" from ft(jinc), "circle" from circle in fourier space \n'


PSF_PARAMS.NA = 0.9;
PSF_PARAMS.n = 1;
#PSF_PARAMS.dimension = 2;
PSF_PARAMS.wavelength = 500;
PSF_PARAMS.pol = 'lin_x';
PSF_PARAMS.pols = __pols__;
PSF_PARAMS.pol_xy_phase_shift =0;
PSF_PARAMS.pol_lin_angle =0;
PSF_PARAMS.vectorized = True;
PSF_PARAMS.aplanar = 'excitation';
PSF_PARAMS.apl = __aplanars__;
PSF_PARAMS.off_focal_distance = 0;
#PSF_PARAMS.Z_PXS_FOR_2D_IMG = 100;
#PSF_PARAMS.Z_STEPS_FOR_2D_IMG = 20;
PSF_PARAMS.aberration_strength = None;
PSF_PARAMS.aberration_types = None;
PSF_PARAMS.aberration_zernikes = __zernikes__;
PSF_PARAMS.aperture_transmission = None;
PSF_PARAMS.foc_field_method = 'theoretical';


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
        'IMG_PIXELSIZES': [100,50, 50],      #Default pixelsizes (list or tuple -> has to be at least 3dimensional)
        'IMG_PIXEL_UNITS': 'nm',             # Default units of image pixelsizes
        'IMG_TIFF_FORMATS': ['tif', 'tiff'],
        'IMG_IMG_FORMATS':  ['bmp', 'png'],
        'IMG_ZEISS_FORMATS': ['czi'],
        'IMG_DEFAULT_IMG_FOLDERS' :[os.path.split(__file__)[0]+r'\\resources'],     # Default folders to find images
        'IMG_DEFAULT_IMG_NAME' : 'todesstern',                 # Default image to load
        'IMG_VIEWER': 'VIEW5D',           # Default viewer -> currently only implemented viewr -> later also view5D, currently allwowd 'NIP_VIEW', 'INFO'
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
        
        # TODO DEPRICATED: DELETE AFTER TESTS
        # #FT Default settings
        # 'FT_NORM': "ortho",           # None or "ortho": Normalization of FT: if None -> zero frequency ft strengths contains all image pixels, if 'ortho' -> the ft and the ift have the same scaling factor (but than the zero freq. contains only sqrt(pixelsumme))
        # 'FT_SHIFT':True,          # Shift (AFTER TRANSFORMATION)
        # 'FT_SHIFT_FIRST':True,    # Shift (BEFORE TRANSFORMATION)
        # 'FT_RETURN': 'complex',    # Return of the FTs (string values: complex , abs , phase, real, imag, default)
        # 'FT_REAL_RETURN':None,   # fill up real return to full spectrum? ('full' or None)
        # #RFT Default settings   ----- RH 22.12.2018
        # 'RFT_NORM': None,           # Normalization of FT: if None -> zero frequency ft strengths contains all image pixels, if 'ortho' -> the ft and the ift have the same scaling factor (but than the zero freq. contains only sqrt(pixelsumme))
        # 'RFT_SHIFT':False,          # Shift (AFTER TRANSFORMATION)
        # 'RFT_SHIFT_FIRST':False,    # Shift (BEFORE TRANSFORMATION)
        # 'RFT_RETURN': 'complex',    # Return of the FTs (string values: complex , abs , phase, real, imag, default)
        # 'RFT_REAL_RETURN':None,   # fill up real return to full spectrum? ('full' or None)
        # # Same like above but for IFT
        # 'IFT_NORM': "ortho",           # Normalization of FT: if None -> zero frequency ft strengths contains all image pixels, if 'ortho' -> the ft and the ift have the same scaling factor (but than the zero freq. contains only sqrt(pixelsumme))
        # 'IFT_SHIFT':True,          # Shift (AFTER TRANSFORMATION)
        # 'IFT_SHIFT_FIRST':True,    # Shift (BEFORE TRANSFORMATION)
        # 'IFT_RETURN': 'complex',    # Return of the FTs (string values: complex , abs , phase, real, imag, default)
        # 'IFT_REAL_AXIS': 'GLOBAL',   # Which real axis to take for the ift? ('GLOBAL' as stored in global variable. That is useful if the rft was performed before, since there the real axis is defined), None -> the last axis is taken, value -> as stated in value
        # # Same like above but for IRFT
        # 'IRFT_NORM': None,           # Normalization of FT: if None -> zero frequency ft strengths contains all image pixels, if 'ortho' -> the ft and the ift have the same scaling factor (but than the zero freq. contains only sqrt(pixelsumme))
        # 'IRFT_SHIFT':False,          # Shift (AFTER TRANSFORMATION)
        # 'IRFT_SHIFT_FIRST':False,    # Shift (BEFORE TRANSFORMATION)
        # 'IRFT_RETURN': 'complex',    # Return of the FTs (string values: complex , abs , phase, real, imag, default)
        # 'IRFT_REAL_AXIS': 'GLOBAL',   # Which real axis to take for the ift? ('GLOBAL' as stored in global variable. That is useful if the rft was performed before, since there the real axis is defined), None -> the last axis is taken, value -> as stated in value
        # correlator
        'CC_ABS_RETURN': True ,     # if true, the absolute value will be returned when correlating real images
        
        
        # SIM STUFF
        'SHOW_GRAT_SEARCH_INFO': True
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