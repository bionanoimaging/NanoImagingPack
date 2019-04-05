# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 09:57:15 2017

@author: ckarras
"""


import numpy as np;
import PIL;
from ..util import get_type, adjust_lists;
#import matplotlib.pyplot as plt
def generate_grating(grating_para,phase_nr, num_phases,dim_slm= [1280, 1024],method = 'binary', blaze = 0,  periods = 1, binary_threshold = 0):
    """
    Generate the grating from the given Grating paramters according to SLM dimension

    grating_para:
            if one dimensional [ahx,ahy,apx,apy]
            if two dimensional [[ahx1,ahy1,apx1,apy1],[ahx2,ahy2,apx2,apy2], ... ];

    periods: over how many grating periods are the phase steps distributed (this is important for the 2D Gratings)
    binary_threshold is for unequal population
    """
    from .find_grating import calc_per, calc_orient;
    from ..coordinates import xx, yy, zz;
    if np.ndim(grating_para) == 1:
        period = calc_per(*grating_para);
        direction = calc_orient(grating_para[2],grating_para[3])
        k =2*np.pi/period*np.asarray([np.sin(direction*np.pi/180), np.cos(direction*np.pi/180)]);
        #grating = np.sin(xx(dim_slm)*k[0]+yy(dim_slm)*k[1]+phase_nr*np.pi*periods/num_phases+1E-4)+1E-4;
        localphase=xx(dim_slm)*k[0]+yy(dim_slm)*k[1]+phase_nr*np.pi*periods/num_phases
        grating = np.sin(localphase)+1E-4;
        if blaze >= 0:
            add_blaze = blaze*np.mod((localphase+1E-4)/(2*np.pi),1.0);
        else:
            add_blaze = np.abs(blaze)*(1-np.mod((localphase+1E-4)/(2*np.pi),1.0));
    else:
        period = calc_per(grating_para[0,:],grating_para[1,:],grating_para[2,:],grating_para[3,:]);
        direction = calc_orient(grating_para[2,:],grating_para[3,:])
        k =2*np.pi/period*np.asarray([np.sin(direction*np.pi/180), np.cos(direction*np.pi/180)]);

        #xx becomes yy, yy become zz due to new indexing
        grating = np.sin(yy(dim_slm+[grating_para.shape[1]])*k[0]+zz(dim_slm+[grating_para.shape[1]])*k[1]+phase_nr*np.pi*periods/num_phases+1E-4);
        if blaze >=0:
            add_blaze = blaze*np.mod((xx(dim_slm+[grating_para.shape[1]])*k[0]+yy(dim_slm+[grating_para.shape[1]])*k[1]+phase_nr*np.pi*periods/num_phases+1E-4)/2/np.pi,1.0);
        else:
            add_blaze = np.abs(blaze)*(1-np.mod((xx(dim_slm+[grating_para.shape[1]])*k[0]+yy(dim_slm+[grating_para.shape[1]])*k[1]+phase_nr*np.pi*periods/num_phases+1E-4)/2/np.pi,1.0));
    if method == 'binary':
        return((grating>binary_threshold)*1+add_blaze);
    else:
        return((grating+1)/2+add_blaze);


def generate_grating2(grating_para, phase_nr, num_phases,dim_slm= [1280, 1024],method = 'binary', blaze = 0,  periods = 1, binary_threshold = 0):
    """
    Generate the grating from the given Grating paramters according to SLM dimension
    Difference to "generate_grating" is that the phases are shifted over 2pi

    grating_para:
            if one dimensional [ahx,ahy,apx,apy]
            if two dimensional [[ahx1,ahy1,apx1,apy1],[ahx2,ahy2,apx2,apy2], ... ];

    periods: over how many grating periods are the phase steps distributed (this is important for the 2D Gratings)
    binary_threshold is for unequal population
    """
    from .find_grating import calc_per, calc_orient;
    from ..coordinates import xx, yy, zz;
    if np.ndim(grating_para) == 1:
        
        period = calc_per(*grating_para);
        direction = calc_orient(grating_para[2],grating_para[3])
        k =2*np.pi/period*np.asarray([np.sin(direction*np.pi/180), np.cos(direction*np.pi/180)]);
        grating = np.sin(xx(dim_slm)*k[0]+yy(dim_slm)*k[1]+phase_nr*2*np.pi*periods/num_phases+1E-4);
        if blaze >= 0:
            add_blaze = blaze*np.mod((xx(dim_slm)*k[0]+yy(dim_slm)*k[1]+phase_nr*periods*2*np.pi/num_phases)/(2*np.pi),1.0);
        else:
            add_blaze = np.abs(blaze)*(1-np.mod((xx(dim_slm)*k[0]+yy(dim_slm)*k[1]+phase_nr*2*np.pi*periods/num_phases)/(2*np.pi),1.0));
    else:
        period = calc_per(grating_para[0,:],grating_para[1,:],grating_para[2,:],grating_para[3,:]);
        direction = calc_orient(grating_para[2,:],grating_para[3,:])
        k =2*np.pi/period*np.asarray([np.sin(direction*np.pi/180), np.cos(direction*np.pi/180)]);
        grating = np.sin(yy(dim_slm+[grating_para.shape[1]])*k[0]+zz(dim_slm+[grating_para.shape[1]])*k[1]+phase_nr*2*np.pi*periods/num_phases);
        if blaze >=0:
            add_blaze = blaze*np.mod((xx(dim_slm+[grating_para.shape[1]])*k[0]+yy(dim_slm+[grating_para.shape[1]])*k[1]+phase_nr*2*np.pi*periods/num_phases)/2/np.pi,1.0);
        else:
            add_blaze = np.abs(blaze)*(1-np.mod((xx(dim_slm+[grating_para.shape[1]])*k[0]+yy(dim_slm+[grating_para.shape[1]])*k[1]+phase_nr*1*np.pi*periods/num_phases)/2/np.pi,1.0));
    if method == 'binary':
        return((grating>binary_threshold)*1+add_blaze);
    else:
        return((grating+1)/2+add_blaze);


def test_grat(folder, im, paras, name, phase_steps):
    from .find_grating import calc_per, calc_orient, lcm;
    from ..image import shift_center;
    print('');
    print('Grating '+name);
    print('**********************************************************************');
    print('Period:')
    print(calc_per(*paras));
    print('Orientation:')
    print(calc_orient(paras[2],paras[3]));
    print('Vert_Steps: (should be 0)')
    #print(np.mod(lcm(np.abs(paras[0]),np.abs(paras[2]))*(paras[3]/paras[2]-paras[1]/paras[0]),phase_steps))
    if paras[0] == 0:
        print(np.mod(paras[1],phase_steps));
    else:
        print(np.mod(lcm(np.abs(paras[0]),np.abs(paras[2]))*paras[3]/paras[2]-paras[1]/paras[0],phase_steps))
    print('Hor_Steps: (should be 0)')
    #print(np.mod(lcm(np.abs(paras[1]),np.abs(paras[3]))*(paras[2]/paras[3]-paras[0]/paras[1]),phase_steps));
    if paras[1] == 0:
        print(np.mod(paras[0],phase_steps));
    else:
        print(np.mod(lcm(np.abs(paras[1]),np.abs(paras[3]))*paras[2]/paras[3]-paras[0]/paras[1],phase_steps));


    print(phase_steps)
               
    deltax = paras[0]#-31#14;
    deltay = -paras[1]#-28#3;

    #r = [10,10];
    #start = [160,110];
    #start = [0,00];
    #r = [500,500]
    #im_shift = nip.shift(im, deltax, 0);
    #im_shift = nip.shift(im_shift, deltay,1);
    im_shift = shift_center(im, deltax, deltay)

    #im = im[start[0]:start[0]+r[0],start[1]:start[1]+r[1]];
    #im_shift = im_shift[start[0]:start[0]+r[0],start[1]:start[1]+r[1]];
    #
    #nip.close();                    
    #view(im,'original');
    #view(im_shift,'shifted')
    #view(im-im_shift,'difference');
    im2 = PIL.Image.fromarray(np.transpose(im - im_shift));
    im2.save(folder+name,"png");
    return(im2);


def save_grating_im(path,num_phase,dim_slm, para_set,circle_rad = -1,name_ext='', test_grating_shift=False, bitdepth = 1,form = 'png', val = [0,255], method = 'binary', blaze = 0, function = 1):
    from ..mask import create_circle_mask;
    zero = np.transpose(np.zeros(dim_slm)+1);
    zero = np.uint8(zero*val[1]);
    im = PIL.Image.fromarray(zero)
    if bitdepth == 1:
        im2=im.convert("1");
    elif bitdepth == 8:
        im2= im.convert("L")
    else:
        print('Invalid bitdepth! Either 1 or 8');
        return;

    im2.save(path+'bright.'+form,form);

    for phase_nr in np.arange(0,num_phase,1):
        if function == 1:
            grat = generate_grating(para_set[2:6],phase_nr, num_phase, dim_slm, method, blaze);
        else:
            grat = generate_grating2(para_set[2:6],phase_nr, num_phase, dim_slm, method, blaze);
        if test_grating_shift:
            test_grat(path, grat, para_set[2:6],'TESTSHIFTw'+str(para_set[0])+'a'+str(int(para_set[1]))+'p'+str(phase_nr)+'.'+form);
        if (circle_rad != -1): grat = grat*create_circle_mask(tuple(dim_slm),maskpos = (0,0) ,radius=circle_rad);  #mask has to be transposed. I don't know why!
        name = name_ext+'_w'+str(para_set[0])+'a'+str(int(para_set[1]))+'p'+str(phase_nr)+'.'+form
        im = PIL.Image.fromarray(np.transpose(np.uint8(grat*(val[1]-val[0])+val[0])));    # has to be transposed to fit!!!
        if bitdepth == 1:
            im2=im.convert("1");
        elif bitdepth == 8:
            im2= im.convert("L")
        else:
            print('Invalid bitdepth! Either 1 or 8');
            return;
        im2.save(path+name,form);


def load_grat_para_file(para_path, version =2):
    """
        load a parameter file for grating generation

        input:
               para file path
               version (should be 2)
        returns:
            Paras
            Num_phases

    """
    with open(para_path) as f:
        lines = f.readlines();
    l  = lines[6];
    NumPhases=int(l[14:]);
    f.close();
    if version <2:
        Paras = np.loadtxt(para_path, dtype = int, skiprows = 16); # axis 0: paras of one combination wavelength, angle etc
    elif version == 2:
        Paras = np.loadtxt(para_path, dtype = int, skiprows = 17); # axis 0: paras of one combination wavelength, angle etc
    return(Paras, NumPhases)

def create_grating(para_path, Save_Grat_Folder, dim_slm =[2048,1536], circle_aperture_radius = -1, name_tag = '', test_grating_shift = False, bitdepth = 1, form = 'png', val = 'max', method = 'binary', version = 2, blaze_vec = None, function = 1):
    """
    This script creates the grating images using the parameter text file which was produced by the "find_grating" script

    Again: some functions (e.g. xx) are implemented directly here (and not in good programming style) instead of using the NanoImagingPack functions
    This is because I wrote this scripts before the NanoImagingPack and now I'm too lazy to change those things (anyway they work!)


    The function takes the following parameters:

    para_path:                  path (including file name) of the parameter text file
    Save_Grat_Folder:           folder of where to save the images
    dim_slm:                    pixelsize of the SLM (a list of 2 compontents)

    circle_aperture_radius      Radius of a circular aperture mask in the center (in pixels)

    name_tag:                   a specifiert for the images. All images will by default be called either "bright.png" (for the bright image) or "<name_tag>_w<wavelength>a<direction>p<phasenumber>.png"
                                Example: lets assume the nametag is "sim" you have 3 phases and 3 directions where the first is at 33 Degree and the wavelengthlist  is [488 nm, 561 nm] the name of the first image will be "sim_w488_a33p0.png"

    Test Grating Shift:         For every phase: save an image where the grating added to a grating which is shifted for one unit cell is saved

    bitdepth:                   1 or 8;
    form:                       image format!

    val:                        value range -> can be tupel or int or list or max
                                    if max -> maximum  will be taken (e.g. grating is 0-1 for 1 bit or 0-255 for 8 bit)
                                    int:- grating is 0 - int (values over 255 will be trunkated)
                                    tupel / list -> min -max (values below 0 or  above 255 will be trunkated)
    method:                     which grating kind? ->
                                            'binary'     - binary gratin
                                            'full'       - full grating

    version                     some integer to track versions a little bit -> before April 2018 is 1, after is 2

    blaze_vec                   state here (as list or tuple) how large an additional blaze for each direction should be -> for multiple wavelenght -> add just to list as stated in the para file
                                        e.g. you have 2 Wavelength, 3 dirs and want add a blace of 1% for the first wl, 2nd direction, and 5 % for the 2nd wavelenght 1st direction state it like:
                                                        blaze_vec = [0, 0.01,0,0.05];
    function:                   if 1: 0-pi phase shift, if 2: 0-2pi phase shift
    """
    def __check_val__(val):
        if val <0: 
            val = 0;
        if val > 255:
            val = 255;
        return(val);

    if bitdepth == 1:
        if method != 'binary' or val != 'max':
            print('WARNING: Bitdepth is 1 -> creating binary image between 0 and 1')
        val = [0,255];
        method = 'binary';
    else:
        if val == 'max':
            val = [0,255];
        elif type(val) == int:
            val = [0,__check_val__(val)];
        elif type(val) == tuple or type(val) == list:
            val = [__check_val__(val[0]),__check_val__(val[1])];
        else:
            val = [0,255];


    Paras,NumPhases = load_grat_para_file(para_path, version);





#    if (Save_Grat_Folder[-1]!= '/'):
#        Save_Grat_Folder = Save_Grat_Folder+'/';
    
    import os;
    if (os.path.isdir(Save_Grat_Folder)) == False:
        os.mkdir(Save_Grat_Folder);


    if get_type(blaze_vec)[0] == 'none':
        blaze_vec = [0 for i in range(len(Paras))];
    elif get_type(blaze_vec)[0] == 'list':
        blaze_vec, Paras = adjust_lists(blaze_vec, Paras)
    else:
        raise TypeError('Wrong type for blazevec -> should be none or list')
    
    for p, blaze_val in zip(Paras, blaze_vec):
        print(p);
        save_grating_im(Save_Grat_Folder, NumPhases, dim_slm, p,circle_aperture_radius,name_tag, test_grating_shift, bitdepth, form, val, method, blaze_val, function);
    np.savetxt(Save_Grat_Folder+'Blaze_info.txt', blaze_vec);
