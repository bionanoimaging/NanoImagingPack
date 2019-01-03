# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 13:27:29 2018

@author: ckarras
"""

class sim_setup():
    '''
        Analyze a SIM setup:
            regarding length and bfp illumination
            
            
            
            |:---------|------:--------|----------------|-----|__>
           SLM      f_coll   MASK    f_refoc         f_tube   Objective
           
           -> all dimensions in mm
           -> Give also:
                   f_tube_default which is the default objective tube lens length (default is 162.5 which is Zeiss)
                   pixelsize in um
                   M_obj: Objective magnification
                   NA: Objective NA
                   wavelength in nm;
            
           -> Returns: 
               required grating period (um);
               Illumination of back focal plane;
           
           
            
            
    '''
    def __init__(self,f_coll, d_mask, f_refoc, f_tube, M_obj, NA, pxs_slm, wavelength, f_tube_default = 162.5):
        import numpy as np;
        self.f_coll = f_coll;
        self.f_refoc =  f_refoc;
        self.f_tube = f_tube;
        self.M_obj = M_obj;
        self.NA = NA;
        self.pxs = pxs_slm;
        self.mask_diameter= d_mask;
        self.f_tube_default = f_tube_default;
        self.wavelength = wavelength/1000;
        
        
        self.SYSTEM_MAGN = self.M_obj*self.f_tube/self.f_tube_default * self.f_coll/self.f_refoc; # System magnification
        self.DIFFRACTION_ANGLE = np.arctan(self.mask_diameter/(2*self.f_coll))*180/np.pi;
        self.GRATING_PERIOD_mm = self.wavelength/np.sin(self.DIFFRACTION_ANGLE*np.pi/180);
        self.GRATING_PERIOD = self.GRATING_PERIOD_mm/self.pxs;
        self.BFP_ILLU = self.wavelength*self.SYSTEM_MAGN/(self.GRATING_PERIOD_mm*self.NA)*100;
        self.SYSTEM_LENGTH = 2*self.f_coll+2*self.f_refoc+ self.f_tube;
        
        
        # for data logging:
        self.header = 'Length [mm] \t BFP illu \t grating period [px] \t grating period [mm] \t f(coll) [mm] \t f(refoc) [mm] \t f(tube) [mm] \t Mask Diameter [mm] \t NA \t Pixelsize [um] \t wavelength [nm] \t magnification \n'
        self.data_string = '';
        
        self.data_string += str(self.SYSTEM_LENGTH)+' \t';
        self.data_string += str(self.BFP_ILLU)+' \t';
        self.data_string += str(self.GRATING_PERIOD)+' \t';
        self.data_string += str(self.GRATING_PERIOD_mm)+' \t';
        self.data_string += str(self.f_coll)+' \t';
        self.data_string += str(self.f_refoc)+' \t';
        self.data_string += str(self.f_tube)+' \t';
        self.data_string += str(self.mask_diameter)+' \t';
        self.data_string += str(self.NA)+' \t';
        self.data_string += str(self.pxs)+' \t';
        self.data_string += str(self.wavelength*1000)+' \t';                                                                                                                           
        self.data_string += str(self.SYSTEM_MAGN)+' \n';                               