# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 18:52:35 2018

@author: ckarras
"""

import numpy as np;

def make_sim_otf(OTF, k_vec, dir_vec, variance_vec, pxs, orderstrength, method = 'sum'):
    '''
        compute the effective SIM OTF according to Key's thesis out of the given otf and the grating parameters
        
        TO ADD: 
            Order strength
            Zero order suppression
            
        OTF         - Widefield OTF
        k_vec       - k_shift vectors for one direction -> only the positve non - zero orders in 1/[unit_pxs]
                        (e.g. for 3-beam-sim  k_vec = [2E-3, 4E-3]) for BFP 100 %, lambda =500 nm and NA = 1.0 (k_abbe = 4E-3 nm^-1)
                        (e.g. for 2-beam-sim  k_vec = [ 4E-3]) for BFP 100 %, lambda =500 nm and NA = 1.0 (k_abbe = 4E-3 nm^-1)
            
        dir_vec    - The direction vector in degree (e.g. [0, 60, 120])
        variance_vec   - the variance of each individual order
        pxs       - pixelsize (can be number or vector if the pixelsize is differnt for x and y)
        
        order_strength - strengths of the SIM orders!
                    Must be 1D or 2D List of order strength! 
                        1st D -> along all orders
                        2nd D -> along different directions
                        order strength of zeroth (central, widefield) order is normalized to 1 	
                            e.g.  order_strength = [[d1k1, d2k1, d3k1], [d1k2, d2k2, d3k2]] for 3 Directions and 2 elements in k_vec                                                      
                            or order_strength = [k1, k2]   for 3 Direction (all same orderstrength) and 2 elements in k_vec (3-Beam-SIM);
        
        method    - Method of the SIM OTF according to Kay's Thesis:
                    - 'sum'  simple summing
                    - 'wa'   weighted averaging
                    - 'wn'   noise normalised weighted averaging
    '''
    from ..image import shift, cat;
    from ..coordinates import px_freq_step;
    
    
    if len(orderstrength) != len(k_vec):
        print('Wrong order strength vector size');
        return();
    
    import numbers
    for i in range(len(orderstrength)):
        if isinstance(orderstrength[i], numbers.Number):
            orderstrength[i] = [orderstrength[i] for d in dir_vec];
        if type(orderstrength[i]) == list:
            if len(orderstrength[i]) != len(dir_vec):
                print('Wrong order strength vector size');
        
    px_converter = px_freq_step(OTF, pxs);  # conversion factro for k ->pixel
    k_vec = [-k for k in k_vec[::-1]]+k_vec;
    orderstrength = [o for o in orderstrength[::-1]]+ orderstrength;
    shifted_otfs = OTF;
    for alpha in enumerate(dir_vec):
        for k in enumerate(k_vec):
            shift_vec= [np.cos(alpha[1]*np.pi/180)*k[1]/px_converter[0], np.sin(alpha[1]*np.pi/180)*k[1]/px_converter[1]]
            shifted_otfs = cat((shifted_otfs, shift(OTF*orderstrength[k[0]][alpha[0]], shift_vec)),2);
    
    
    v  = variance_vec;   # TODO: ADD VARIANCE_VECTOR HNADLING HERE!
    
    
    
    if method == 'sum':
        sim_otf = np.sum(shifted_otfs, axis = 2);
    elif method == 'wa':
        sim_otf = np.sum(shifted_otfs**2/v, axis =2)/np.sum(shifted_otfs/v, axis =2);
    elif method == 'wn':
        sim_otf = np.sqrt(np.sum(shifted_otfs**2/v, axis =2))
    else:
        print('unknown method');
        sim_otf = shifted_otfs;
    return(sim_otf);