# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 10:27:24 2017

@author: ckarras

In this module some file utilitis are implemented
"""
from .util import get_type;

def getFolder():
    '''
        Selecte a directory -> gives back correct path - string
    '''
    from tkinter import Tk, filedialog
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    return(filedialog.askdirectory()+'/')

def get_sorted_file_list(directory, file_prototype, sort ='date', key = None):
    '''
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
    '''
    from os import listdir;
    from os.path import isfile, join, getmtime;
    flist = [(f,getmtime(join(directory, f)))  for f in listdir(directory) if isfile(join(directory, f))]
    flist = list(filter(lambda x: x[0].find(file_prototype) >=0,flist));
    if sort == 'name':
        flist = sorted(flist, key = lambda x: x[0]);
    elif sort == 'date':
        flist = sorted(flist, key = lambda x: x[1]);
    elif sort == 'integer_key':
        if key != None:
            from .util import parse_string_for_int;
            flist = [(f[0], f[1], parse_string_for_int(f[0],key)) for f in flist]
            flist = sorted(flist, key = lambda x: x[2])
        else:
            print('Error: Give key character');
    return([x[0] for x in flist]);

def get_Folderlist(directory, exclude_list = []):
    '''
        Get all subdirectories in the directory. 
        Exclude all folders which contain an element in their path given in exclude list
        
        Example
        
        dir = nip.str_to_path(r'C:\MyFiles')
        Excludelist = ['timeser488', 'BPAE_Example']
        get_Folderlist(dir, Excludelist):
            
            You get all Subdirectories in C:\MyFiles except those containing 'timeser488' and 'BPAE_Example'
    '''
    import os;
    dirs= [x[0] for x in os.walk(directory)];
    dirs = dirs[1:];
    for el in exclude_list:
        dirs = list(filter(lambda x: x.find(el)==-1, dirs));
    return(dirs);
    
def getFile():
    '''
        Select a file or a list of files -> gives back correct path - string (or a list)
    '''
    from tkinter import Tk, filedialog
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    return(filedialog.askopenfilenames())

def list_files(path, file_prototype):
    '''
    Returns a list with all files in the given folder AND subfolders which include the "file_prototype" string in their name
    '''    
    import os
    import os.path
    files =[];
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if f.find(file_prototype)>=0]:
            files.append(os.path.join(dirpath, filename));
            
    return(files)
    
    
def delete_files(file_prototype):
    '''
        Scans the given folder (choose via prompt) for files which contain the "file_prototype" string and deletes them
    '''
    from os import remove;
    folder = getFolder();
    if folder != '/':
        for f in list_files(folder, file_prototype):
            print('Deleting: '+f);
            remove(f);
        
    return()

def copy_file(src_file,dst_folder):
    '''
        copy a source file into a destination folder
    '''
    from shutil import copyfile
    from os.path import split;
    copyfile(src_file, dst_folder+split(src_file)[1])
    return;
        
def str_to_path(name):
    '''
        this changes a string to a path name that can be handled by python.
        You may as well enter a list of strings here.
        
        note: the path string should start with an r such as:
            
            s = r'C:\hello_kitty\'
        
        since this disables the usage of  '\' as controll character
            
    '''
    def __change2path__(s):
        import os;
        s = os.path.normpath(s);
        if (s[-2:] != '\\') and s[-4] != '.':
            s = s+'\\';
        return(s);
    if type(name) == str:
        return(__change2path__(name));
    elif type(name) == list:
        return([__change2path__(s) for s in name]);
    else:
        print('Not a string or a list')
        return(':-(');

def parse_m_files(directory, name, ignore_comments = True, subfolders = False, exclude_list = []):
    '''
        parses all matlab (*.m) files in a given directory for a given string and displays at which file in which line it is found!
        
        if ignore_comments = True then comments will be ignored

        if subfolders = True it checks in subfolders also
        if you want to explicitly exclude folders they can be stated in the exclude_list - list (c.f. help file from  get_Folderlist)
        
    '''
    if subfolders == False:
        directory_list = [directory];
    else:
        directory_list = get_Folderlist(directory, exclude_list = exclude_list);
        directory_list = [str_to_path(d) for d in directory_list];
    for d in directory_list:
        files = get_sorted_file_list(directory, '.m');
        print();
        print();
        print('Searching for \''+name+'\' in all matlab files in directory \''+directory+'\'')
        print();
        print('Found in:');
        for file in files:
            with open(directory+file, 'r') as f:
                data = True;
                l = 0;
                while(data):
                    data = f.readline();
                    l+=1;
                    pos = data.find(name);
                    commenter = data.find('%');
                    if pos >=0:
                        if (commenter <0) or (commenter > pos) or (ignore_comments == False):
                            print(file + '\t\t Line: '+str(l));