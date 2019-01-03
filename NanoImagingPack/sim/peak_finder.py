# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 17:35:40 2017

@author: ckarras
"""

def sim_peak_finder(raw, phases = 3, angles = 3, stacksize = 1):
    '''
        uses crude separation in order to find SIM peaks by 3d ft of the x-y-phase pattern and than cross correlating zeros and +- first order
        
        Raw can be either a path string of the 3d SIM stack or a ndarray with the SIM stack
    
        if Raw is a stack the crosscorrelations of the directions will be saved
    '''
    
    if type(raw) == str:
        print('Reading file: '+str)