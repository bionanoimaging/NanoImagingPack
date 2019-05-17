# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 10:27:24 2017

@author: ckarras

In this module some file utilitis are implemented
"""

import os.path
from .config import __DEFAULTS__;

def getFolder():
    """
        Selecte a directory -> gives back correct path - string
    """
    from tkinter import Tk, filedialog
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    return(filedialog.askdirectory()+'/')

def get_sorted_file_list(directory, file_prototype, sort ='date', key = None):
    """
        get a sorted file list:
            directory:
                directory of the files
            file prototype:
                only files which filenmaes include this string fill be shown
             sort:
                 'name'
                 'date'
                 'integer_key' : sort by an inter which follows a key character in the name. For this option you have to give the key characer

            returns file list;
    """
    from os import listdir
    from os.path import isfile, join, getmtime
    flist = [(f,getmtime(join(directory, f)))  for f in listdir(directory) if isfile(join(directory, f))]
    flist = list(filter(lambda x: x[0].find(file_prototype) >=0,flist))
    if sort == 'name':
        flist = sorted(flist, key = lambda x: x[0])
    elif sort == 'date':
        flist = sorted(flist, key = lambda x: x[1])
    elif sort == 'integer_key':
        if key != None:
            from .util import parse_string_for_int
            flist = [(f[0], f[1], parse_string_for_int(f[0],key)) for f in flist]
            flist = sorted(flist, key = lambda x: x[2])
        else:
            print('Error: Give key character')
    return([x[0] for x in flist])


def get_Folderlist(directory, exclude_list = ()):
    """
        Get all subdirectories in the directory.
        Exclude all folders which contain an element in their path given in exclude list

        Example

        dir = nip.str_to_path(r'C:\MyFiles')
        Excludelist = ['timeser488', 'BPAE_Example']
        get_Folderlist(dir, Excludelist):

            You get all Subdirectories in C:\MyFiles except those containing 'timeser488' and 'BPAE_Example'
    """
    import os
    dirs= [x[0] for x in os.walk(directory)]
    dirs = dirs[1:]
    for el in exclude_list:
        dirs = list(filter(lambda x: x.find(el)==-1, dirs))
    return(dirs)


def getFile():
    """
        Select a file or a list of files -> gives back correct path - string (or a list)
    """
    from tkinter import Tk, filedialog
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    return(filedialog.askopenfilenames())

def list_files(path, file_prototype, subfolders = True):
    """
    Returns a list with all files in the given folder AND subfolders which include the "file_prototype" string in their name
    """
    files =[]
    if subfolders:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in [f for f in filenames if f.find(file_prototype)>=0]:
                files.append(os.path.join(dirpath, filename))
    else:
        files = [f for f in os.listdir(path) if os.path.isfile(path+f) and f.find(file_prototype)>=0]        
    return(files)
    
    
def delete_files(file_prototype):
    """
        Scans the given folder (choose via prompt) for files which contain the "file_prototype" string and deletes them
    """
    from os import remove
    folder = getFolder()
    if folder != '/':
        for f in list_files(folder, file_prototype):
            print('Deleting: '+f)
            remove(f)

    return()

def copy_file(src_file,dst_folder):
    """
        copy a source file into a destination folder
    """
    from shutil import copyfile
    from os.path import split
    copyfile(src_file, dst_folder+split(src_file)[1])
    return


def str_to_path(name):
    """
        this changes a string to a path name that can be handled by python.
        You may as well enter a list of strings here.

        note: the path string should start with an r such as:

            s = r'C:\hello_kitty\'

        since this disables the usage of  '\' as controll character

    """
    import os;
    return(os.path.abspath(name));

#   DEPRICATED CODE:
#    def __change2path__(s):
#        import os
#        s = os.path.normpath(s)
#        if (s[-2:] != '\\') and s[-4] != '.':
#            s = s+'\\'
#        return(s)

#    if type(name) == str:
#        return(__change2path__(name))
#    elif type(name) == list:
#        return([__change2path__(s) for s in name])
#    else:
#        print('Not a string or a list')
#        return(':-(')


def parse_m_files(directory, name, ignore_comments = True, subfolders = False, exclude_list = ()):
    """
        parses all matlab (*.m) files in a given directory for a given string and displays at which file in which line it is found!

        if ignore_comments = True then comments will be ignored

        if subfolders = True it checks in subfolders also
        if you want to explicitly exclude folders they can be stated in the exclude_list - list (c.f. help file from  get_Folderlist)

    """
    if not subfolders:
        directory_list = [directory]
    else:
        directory_list = get_Folderlist(directory, exclude_list = exclude_list)
        directory_list = [str_to_path(d) for d in directory_list]
    for d in directory_list:
        files = get_sorted_file_list(directory, '.m')
        print()
        print()
        print('Searching for \''+name+'\' in all matlab files in directory \''+directory+'\'')
        print()
        print('Found in:')
        for file in files:
            with open(directory+file, 'r') as f:
                data = True
                l = 0
                while(data):
                    data = f.readline()
                    l+=1
                    pos = data.find(name)
                    commenter = data.find('%')
                    if pos >=0:
                        if (commenter <0) or (commenter > pos) or (ignore_comments == False):
                            print(file + '\t\t Line: '+str(l))


def read_dcimg(filepath, framelist=None, ret_times=False, high_contrast=False, view=None):
    '''
        Read the images from a dcimg streaming file

        This requires the hamamatsu TMCAMCON.dll -> path has to be set in "config.py"

        filepath:    path to dcimg file
        framelist:   images to read:
                        None: read all images
                        range, int, list, tuple, ndarray: indicate whice images to read
        ret_times: returns second parameter with time stamps if true
        high_contrast: Refere to help of TMCAMCON.DLL
        view:          Refere to help of TMCAMCON.DLL
    '''
    import numpy as np;
    import ctypes as ct;
    from .image import image as IMAGE;
    try:
        TMCAMCON_DLL = ct.cdll.LoadLibrary(__DEFAULTS__['ORCA_TMCAMCON_DLL_Path']);  # Links to dll
    except OSError:
        print(
            "DCIMG opening faild, because I could not hook up to tmcamcon.dll. Install DCAMAPI and/or set the path in nip.__DEFAULTS__['ORCA_TMCAMCON_DLL_Path']");
        return;
    import numbers;

    def __unhook_dll__(TMCAMCON_DLL):
        libHandle = TMCAMCON_DLL._handle;
        del TMCAMCON_DLL;
        kernel32 = ct.WinDLL('kernel32', use_last_error=True);
        kernel32.FreeLibrary.argtypes = [ct.c_int64];
        kernel32.FreeLibrary(libHandle)

    try:
        # Define Dll Functions
        DC_OPEN = TMCAMCON_DLL.TMCC_OPENDCIMGFILE_40;
        DC_OPEN.restype = ct.c_uint32;
        DC_OPEN.argtypes = [ct.POINTER(ct.c_uint32), ct.c_char_p, ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32)];
        DC_CLOSE = TMCAMCON_DLL.TMCC_CLOSEDCIMGFILE_40;
        DC_CLOSE.restype = ct.c_uint32;
        DC_CLOSE.argtypes = [ct.c_uint32, ct.POINTER(ct.c_int32)];
        DC_INFO = TMCAMCON_DLL.TMCC_GETDCIMGFRAMEINFO_40;
        DC_INFO.restype = ct.c_uint32;
        DC_INFO.argtypes = [ct.c_uint32, ct.c_int32, ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
                            ct.POINTER(ct.c_int32)];
        DC_DATA = TMCAMCON_DLL.TMCC_GETDCIMGFRAMEDATA_40;
        DC_DATA.restype = ct.c_uint32;
        DC_DATA.argtypes = [ct.c_uint32, ct.c_int32, ct.POINTER(ct.c_uint16), ct.POINTER(ct.c_double), ct.c_int32,
                            ct.POINTER(ct.c_int32)];

        # open dcimg file
        f_handle = ct.c_uint32(0);
        c_tot_frames = ct.c_int32(0);
        c_err_val = ct.c_int32(0);
        c_img_width = ct.c_int32(0);
        c_img_height = ct.c_int32(0);
        c_img_time_stamp = ct.c_double(0);

        if view is None:
            opt = int(high_contrast) * 16
        elif view == 1:
            opt = int(high_contrast) * 16 + 1048576;
        elif view == 2:
            opt = int(high_contrast) * 16 + 2097152;
        elif view == 2:
            opt = int(high_contrast) * 16 + 3145728;
        elif view == 2:
            opt = int(high_contrast) * 16 + 4194304;
        else:
            print('Wrong veiw setting -> setting it to Default');
            opt = int(high_contrast) * 16
        c_opt = ct.c_int32(opt);
        b_file = filepath.encode('utf-8');
        ret = DC_OPEN(ct.byref(f_handle), b_file, ct.byref(c_tot_frames), ct.byref(c_err_val));
        print('Total frames: ' + str(c_tot_frames.value));
        if c_tot_frames.value > 0:
            # prepare image lists:
            if framelist is None:
                framelist = list(range(c_tot_frames.value + 1));
            elif isinstance(framelist, numbers.Integral):
                framelist = [framelist];
            elif isinstance(framelist, range) or isinstance(framelist, tuple) or isinstance(framelist, np.ndarray):
                framelist = list(framelist);
            else:
                __unhook_dll__(TMCAMCON_DLL);
                raise ValueError('Wrong framelist datatype')
            new_list = []
            for el in framelist:
                if isinstance(el, numbers.Integral):
                    if el > c_tot_frames.value:
                        print('WARNING: ' + str(el) + ' out of range -> ignoring it');
                    else:
                        new_list.append(el);
                else:
                    print('Waringin: ' + str(el) + ' has wrong data type -> ignoring it');
            ret = DC_INFO(f_handle, ct.c_int32(0), ct.byref(c_img_width), ct.byref(c_img_height), ct.byref(c_err_val));
            img = IMAGE((len(new_list), c_img_height.value, c_img_width.value)).astype(np.uint16);
            times = IMAGE([len(new_list)]);

            for num, el in enumerate(new_list):
                print('Reading image number ... ' + str(el));
                buf_im = IMAGE((c_img_height.value, c_img_width.value)).astype(np.uint16);
                ret = DC_DATA(f_handle, ct.c_int32(el), buf_im.ctypes.data_as(ct.POINTER(ct.c_uint16)),
                              ct.byref(c_img_time_stamp), c_opt, ct.byref(c_err_val));
                times[num] = c_img_time_stamp.value;

                img[num] = buf_im;
        else:
            print('File is empty!')
        # close dcimg file
        ret = DC_CLOSE(f_handle, ct.byref(c_err_val))
    except Exception as e:
        print('Return value of dll is' + str(ret));
        print(e)
    __unhook_dll__(TMCAMCON_DLL)
    print('Done');
    if ret_times:
        return (img, np.asarray(times));
    else:
        return (img)
