# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:14:06 2019

@author: mafengjiao
"""

import subprocess,os
import numpy as np
import NanoImagingPack as nip
from .config import DBG_MSG, __DEFAULTS__
from . import image
import uuid
import tempfile

def SimPSF(im, psf_params = None, confocal = 0, pinhole = 1, Pi4Em =0, Pi4Ex = 0,nonorm = 0, circPol = 0, twophoton = 0, scalarTheory=0):
    """
    Calculates point spread functions based on the SimPSF executable, originally written in the khoros framework.
    :param im: Input image (for sizes only)
    :param psf_params: psf parameter structure obtained by nip.PSF_PARAMS(). Default: None, overwrites other parameters
    :param confocal: default: 0. if selected a confocal PSF is computed  \n'
    :param pinhole: Pinhole Size [AU]. (default: 1. Pinhole size in Airy Units ( 1 AU diameter is 2 * 0.61 lambda_mean / NA)) \n'
    :param Pi4Em: 4 Pi Emission (default: 0. If selected, the emission PSFs (widefield and confocal) will be affected accordingly) \n'
    :param Pi4Ex:  4 Pi Excitation (default: 0. This will only influence 4Pi excitation in confocal mode (4Pi C and excitation only mode)) \n'
    :param nonorm: default: 0. if selected the confocal PSF will not be normalized to one but the absobtion at the pinhole will be considered. \n'
    :param circPol: Circular Polarization (default: 0. If selected, the incident polarization is assumed to be circular. The meaning of Ex,Ey,Ez has then changed: The components only correctly denote the degree of modulation in this direction so that the total intensity will be correct.) \n'
    :param twophoton: default: 0  \n'
    :param scalarTheory: default: 0: 'If selected, the high-NA scalar theory will be used. This means that the vectorial components of the light are not cosidered and only I0 is used. Furthermore the term (1+cos Theta) is replaced by 2 to yield energy conservation.
    NOT IMPLEMENTED: computeASF 'Amplitude', default: 0: 'if selected a full vectorial Amplitude spread function will be computed. Excitation is assumed to be circular polarized for now.
    :return:  the calcualted point spread function according to Born & Wolf

    # example
import NanoImagingPack as nip
obj3d = nip.readim("MITO_SIM")
obj3d = nip.extract(obj3d, [10, 100, 100])
obj3d.pixelsize = [100, 50, 50]
h = nip.SimPSF(obj3d, confocal=1)
nip.v5(h)
    """
    if os.path.isfile(__DEFAULTS__['KHOROS_PATH'] + os.sep + r'SimPSF.exe'):
        pass
    else:
        print(r"'SimPSF.exe' doesn't exist, please set __DEFAULTS__['KHOROS_PATH'] appropriately so that 'SimPSF.exe' is in it.")
        raise ValueError("Unknown path for __DEFAULTS__['KHOROS_PATH']")
# image shape
    sX = im.shape[-1]
    sY = im.shape[-2]
    if im.ndim == 2:
        sZ = 1
    else:
        sZ = im.shape[-3]
# pixel size 
    scaleX = im.pixelsize[-1]
    scaleY = im.pixelsize[-2]
    if sZ == 1: 
        scaleZ = 1
    else:
        if im.pixelsize[-3] == None:
            print (r'please give the pixel size in z direction')
            raise ValueError("Unknown pixel size in'z'")
        else:
            scaleZ = im.pixelsize[-3]
        
    '''parameters from psf_params'''
    psf_params = nip.getDefaultPSF_PARAMS(psf_params)
    na = psf_params.NA
    ri = psf_params.n
    lambdaEm = psf_params.wavelength 
    
    unique_filename = tempfile.gettempdir() + os.sep + str(uuid.uuid4())
    
#    comm = [__DEFAULTS__['KHOROS_PATH']+os.sep+r'SimPSF.exe','-o', __DEFAULTS__['KHOROS_PATH']+os.sep+r'myfile',
    #comm = [__DEFAULTS__['KHOROS_PATH']+os.sep+r'SimPSF.exe','-o', r'myfile',  
    comm = [__DEFAULTS__['KHOROS_PATH']+os.sep+r'SimPSF.exe','-o', str(unique_filename),  
                '-lambdaEm',     str(lambdaEm),
                '-na',           str(na) ,
                '-ri',           str(ri),
                '-sX',           str(sX),
                '-sY',           str(sY),
                '-sZ',           str(sZ),
                '-scaleX',       str(scaleX),
                '-scaleY',       str(scaleY),
                '-scaleZ',       str(scaleZ)]
#    if computeASF==1 and confocal==1:
#        ValueError('Confocla ASF currently not implemented !')
        
    if confocal==1:
        lambdaEx = psf_params.lambdaEx # only useful for confocal
        p=['-pinhole',str(pinhole),'-lambdaEx', str(lambdaEx)]
        comm.extend(p)
        
    '''conbine the structures in to the tuple'''
    structure={'-Pi4Em':Pi4Em,'-confocal':confocal,'-circPol':circPol,'-twophoton':twophoton,'-Pi4Ex':Pi4Ex,'-nonorm':nonorm, '-scalarTheory':scalarTheory}
    for key in structure:
        if not structure[key]==0:
            comm.append(key)
            
    
    ''' excute SimPSF '''
    subprocess.run(comm)
    ''' read the created bit-file '''
#    bin_bytes = open(__DEFAULTS__['KHOROS_PATH']+os.sep+r'myfile', "rb").read()
    bin_bytes = open(str(unique_filename), "rb").read()
    ''' delete the bit-file '''
    os.remove(str(unique_filename))
     #os.remove(__DEFAULTS__['KHOROS_PATH']+os.sep+r"myfile")
    h = np.frombuffer(bin_bytes, dtype=np.dtype('<f'))
    h = np.reshape(h,(sZ,sY,sX))
    return image(h)

