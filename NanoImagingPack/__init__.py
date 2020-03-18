#!/usr/bin/env python3<
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 00:03:21 2017

@author: root
"""

# externals
import numpy as np;
# modules
# from .. import __NIP_META__
from .coordinates import *
from .image import *
from .mask import *
from .microscopy import *
from .transformations import *
from .util import *
from .view import *
from .fitting import *
from .noise import *
from .alignment import *
from .FileUtils import *
from .config import *
from .DeviceControll import *
from .functions import *
from .simulations import *
from .separables import *
from .view5d import * # rainer's java viewer
from .config import __DEFAULTS__
from .image import image as image
from .test import *
from .simPSF import *
from .nditerator import *

## packages
from . import resources
from . import sim
from . import EXTERNAL

from .EXTERNAL.contrib import *
set_cfg()

# name = __NIP_META__.__title__   # to be depreciated


"""
    ThorlabsPM100
    visa
"""

def find_calls(name):
    """
        find function calls in packages
        Usefull for debuging if you made major chances

        name: which function or class call
    """
    from inspect import getsource, getmembers,isfunction, isclass;
    import sys;
    flist = getmembers(sys.modules[__name__],isfunction); # function_list
    clist = getmembers(sys.modules[__name__],isclass); # class_list
#    f = flist[0];
#    print(f[0]);
#    print(f[1]);
#    print(f[1].__module__);
#    print(getsource(f[1]));
    print('')
    print('#########################################')
    print('')
    print('Looking for calls of in NanoImagingPack '+name);
    print('')
    print('#########################################')
    print('')
    print('')
    print('Found calls in functions:')
    print('')

    for f in flist:
        if getsource(f[1]).find(name) >= 0:
            print(f[0]+'\t\t'+f[1].__module__);
    print('')
    print('Found calls in classes:')
    print('')
    for c in clist:
        if getsource(c[1]).find(name) >= 0:
            print(c[0]+'\t\t'+str(c[1]));


def lookfor(searchstring,show_doc = False):
    """
    finds a function in the NanoImagingPack toolbox whos description matches what is looked for
    :param searchstring: A string to look for
    :param show_doc:
    :return:

    Example:
        import NanoImagingPack as nip
        nip.lookfor("Fourier")
    """
    from inspect import getmembers, isfunction, isclass;
    import sys;
    from difflib import SequenceMatcher;


    def similar(a,b):
        return SequenceMatcher(None, a,b).ratio();

    flist = getmembers(sys.modules[__name__],isfunction); # function_list
    clist = getmembers(sys.modules[__name__],isclass); # class_list
    fl = []
    for f in flist:
        ratio= similar(searchstring, f[0]);
        ds = f[1].__doc__;
        if type(ds) == str:
            s = ds.split();
            for el in s:
                r = similar(searchstring, el);
                if r > ratio: ratio = r;
        if ratio>=__DEFAULTS__['LOOKFOR_RATIO']: fl.append((f[0],f[1],ratio));
    fl.sort(key=lambda x: x[2],reverse=True);

    cl = []
    for c in clist:
        ratio= similar(searchstring, c[0]);
        ds = c[1].__doc__;
        if type(ds) == str:
            s = ds.split();
            for el in s:
                r = similar(searchstring, el);
                if r > ratio: ratio = r;
        if ratio>=0.5: cl.append((c[0],c[1],ratio));
    cl.sort(key=lambda x: x[2],reverse=True);


    print('');
    print('Did you mean:')
    print('')
    print('Functions:')
    print('==========')
    for f in fl:
        print(f[0]+'\t\t'+f[1].__module__);
        if show_doc == True: print(f[1].__doc__)
    print('');
    print('Classes:');
    print('========');
    for c in cl:
        print(c[0]+'\t\t'+str(c[1]))
        if show_doc == True: print(c[1].__doc__)
    return();



