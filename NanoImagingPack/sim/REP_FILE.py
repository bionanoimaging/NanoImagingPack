# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:32:22 2018

@author: ckarras
"""

class REP_FILE():
    '''
    A class to create a .rep file for the QXGA
    
    path:             Folder of the .rep - file
    Name:             Name of the .rep file
    slm:              Which spatial light modulator are you using? (currently it can be 'FDD' for Forth Dimension Display or 'HAMAMATSU' for Hamamatsu Lcos)
    image_format:     Which format have the images? (String e.g. 'png') -> if 'auto'  (default) it will use the common image format for the respective slm (bmp for Hamamatsu and png for FDD)
    init_list         How should the image list be initialized?
                        'complete'   all images in given folder
                        'bright'     only bright image
                        'empty'      noting -> append images later with 'append_image_list'
    
    How to use?
        Create new repfile (define the destination, if destination not defined: you can change it later using "set_path", otherwise standard path will be used)
        
        1) create sequence string        -> This is not neccessary for Hamamatsu!
                Here DEFAULT values for bright and measurement running order are also defined!
                Those can edited later (using .id_bright.DEFAULT or .id_meas_DEFAULT)
        2) create an image list unsing "append_image_list"                                  
                -> Default ist [bright] so the list contains at least one element!
                -> refere to help in appand image in order to figure out how to use it!
        3) Add running orders (RO's) 

                You can use eiter:
                    
                    add_sim_RO(imlist, title, sequence)
                        That adds a SIM Running order for the given imlist the rep-String
                        
                    add_simple_RO(im_list, sequence,title):
                        That adds a  simple running order including im_list images to the rep_String
                        
                    Use the filterlist method for filtering out images:
                        
        4) Save the rep - file

        EXAMPLE:

            # You need the Nanoimaging Pack:
            import NanoImagingPack as nip;
            
            #create rep_file object
            rf = nip.sim.REP_FILE(path = folder_name, name = file_name, slm = 'FDD', image_format = 'auto');
            
            
            #create sequence string using default settings
            rf.create_seq_string();
            
            #print sequence string -> in the same way it can be edited!
            print(rf.seq_string);
            
            # set image list for start angle 33Degree, wavelength 488, and 561, 3 directions, 3 phases, nametag = TEST, 
            rf.append_image_list(33, 3, 3, [488, 561], ['TEST']);
                                 
            # append another image list wavelength 405 nm, start angle 10 Degree, 3 Phases, 2 Directions, different sets ['FULL' and 'APO50']
            rf.append_image_list(10,3,3,[405],['FULL', 'APO50']);
              
            # add a (simple) Running order that only shows the pattens for wavelength 561 nm and phase 0
            rf.add_simple_RO(im_list = rf.filterlist(conditions = ['w561', 'p0']), title = '561 nm, all angles, phase 0')
            
            # add a SIM running order for 488 nm and 561 nm
            rf.add_sim_RO(im_list = rf.filterlist(conditions = ['w561'])+rf.filterlist(conditions = ['w488']), title = 'SIM for 561 nm and 488 nm')
            
            #save file
            rf.save()
            
            
    '''
    def __init__(self,rep_path = None, name = None, slm = 'FDD', image_format = 'auto',init_list = 'complete') :
        from ..FileUtils import str_to_path;
        if rep_path == None:
            import os;
            rep_path = os.getcwd();              # Use current directory if not given
        else:
            rep_path = str_to_path(rep_path);
        
        if name == None:
            name  = 'repfile.rep';
            
        if image_format == 'auto':
            if slm == 'FDD':
                self.image_format = 'png';
            elif slm == 'HAMAMATSU':
                self.image_format = 'bmp';
        else:
            self.image_format = image_format
            
        self.fname = rep_path+name;
        print('Setting up rep-file: ' + self.fname);
        if init_list == 'empty':
            self.imlist = [];
        elif init_list == 'bright':
            self.imlist = ['bright.'+self.image_format];
        elif init_list == 'complete':
            import os.path as p;
            from ..FileUtils import list_files;
            self.imlist = [p.split(f)[1] for f in list_files(rep_path, self.image_format)]
        self.seq_string = '';
        self.im_string = '';
        self.RO_string = '';
        self.id_bright_DEFAULT = [];
        self.id_meas_DEFAULT = [];
        self.slm = slm;
        

    def add_sim_RO(self, im_list, title, sequence = None):
        '''
            Creates a SIM running order
            im_list: Images of the SIM RO
            sequence: sequences, that should be used, e.g. list:['A+','A-'];
        '''
        if self.slm == 'FDD':
            if sequence == None:
                sequence = self.id_meas_DEFAULT;
            if len(sequence) == 0:
                print('WARNING: SEQUENCE EMPTY!')
            ro='"'+title+'"\n[\n';
            for im in im_list:
                ro+='<t';
                for s in sequence: ro+='('+s+','+str(im)+') ';
                ro+='>\n{f';
                for s in sequence: ro+='('+s+','+str(im)+') ';
                ro+='}\n'
            ro+=']\n\n'
            self.RO_string += ro;
            return ro;      
        elif self.slm == 'HAMAMATSU':
            ro=title+'\n[\n';
            for im in im_list:
                ro += str(im);
                if im != im_list[len(im_list)-1]:
                    ro+= ' ';
            ro+='\n]\n\n'
            self.RO_string += ro;
            return ro;      
            

    def add_simple_RO(self,im_list, title, sequence = None):
        '''
            Creates a simple running order
            im_list : list with image numbers of the images to be included (int)
            sequence: sequences, that should be used, e.g. list:['A+','A-'];
        '''
        if self.slm == 'FDD':
            if sequence == None:
                sequence = self.id_meas_DEFAULT;
            if len(sequence) == 0:
                print('WARNING: SEQUENCE EMPTY!')
            
            ro='"'+title+'"\n[\n<';
            for im in im_list:
                for s in sequence: ro+='('+s+','+str(im)+') ';
            ro+='>\n]\n\n'
            self.RO_string += ro;
            return ro;
        elif (self.slm) == 'HAMAMATSU':
            ro=+title+'\n[\n<';
            for im in im_list:
                ro += str(im);
                if im != im_list[len(im_list)-1]:
                    ro+= ' ';
            ro+='\n]\n\n'
            self.RO_string += ro;
            return ro;   
        
    def filterlist(self, conditions, im_list = None):
        '''
            Gets image list and list of strings that should be contained in imlist 
            
            Returns index numbers of that list 
                        
            If imlist not given, the imagelist of the class will be used
        '''
        if im_list == None:
            im_list = self.imlist;
        def conditioning(x, conditions):
            ok = (x.find(conditions[0])>=0);
            for c in conditions:
                if c != conditions[0]:
                    ok = ok*(x.find(c)>=0)
            return(ok)
        l = list(filter(lambda x: conditioning(x,conditions),im_list))
        return([im_list.index(elem) for elem in l])
        
    def set_path(self, new_path):
        print('Setting new path: '+new_path);
        self.fname = new_path;        
        
    def save(self, path = None):
        '''
            Saves rep file
                Path -> as stated either at initialization, using set_path, or here
                
            concetantes seq_string, im_string (filled in that function) and RO_string 
            
            Also impolements the bright running order as first one
        '''
        if path == None:
            path = self.fname;
         
        if self.slm == 'FDD':
            self.im_string+='IMAGES\n';
            for im in self.imlist:
                self.im_string+='1 "'+im+'"\n';
            self.im_string+='IMAGES_END\n\n';   
            brigth_string= 'DEFAULT "bright" \n[\n <('+self.id_bright_DEFAULT[0]+',0) ('+self.id_bright_DEFAULT[1]+',0)>\n]\n\n' 
            text_file = open(path,'w')
            text_file.write(self.seq_string +  self.im_string +brigth_string+self.RO_string);
            text_file.close();
        elif self.slm == 'HAMAMATSU':
            self.im_string+='IMAGES\n';
            for im in self.imlist:
                self.im_string+=im+'\n';
            self.im_string+='IMAGES_END\n\n';   
            text_file = open(path,'w')
            text_file.write(self.im_string +self.RO_string);
            text_file.close();
            
            
        
        
        
        
    def append_image_list(self, start_angle, num_phase, num_angle, wavelength_list, Name_indicator_list):
        '''
        Appends an imlist to the imlist of the class the imagelist in the following order:
            1. NameInidcaotr (e.g. Full or Aperture) -> THIS MUST BE A LIST!
            2. Wavelength (according to Wavelength_list)
            3. Angles
            4. Phases
            '''
        for name in Name_indicator_list:
            for wavel in wavelength_list:
                for angle in range(num_angle):
                    for phase in range(num_phase):
                        self.imlist.append(name+'_w'+str(wavel)+'a'+str(int(angle*180/num_angle+start_angle))+'p'+str(phase)+'.'+self.image_format);
        return(self.imlist);


    def create_seq_string(self, seq_id_list = ['A+','A-','B+','B-','C+','C-'], seq_file_list =['48037 1ms 1-bit Lit Pair +.seq11','48037 1ms 1-bit Lit Pair -.seq11','48038 2ms 1-bit Lit Pair +.seq11','48038 2ms 1-bit Lit Pair -.seq11','48039 3ms 1-bit Lit Pair +.seq11','48039 3ms 1-bit Lit Pair -.seq11'], id_bright = ['C+','C-'], id_meas = ['B+','B-']):
        '''
        Create the sequence string
        
        seq_id_list:      list of names with the sequences
        seq_file_list:    list of sequence files
    
        id_bright:        Default sequence id for bright pattern
        id_meas:          Default sequence id for measurement
        '''
    
        if len(seq_id_list) != len(seq_file_list):
            print('ERROR: seq_id_list and seq_File_List must have same size!')
            
        else:
            if set(id_bright) != set(id_bright).intersection(set(seq_id_list)) or set(id_meas) != set(id_meas).intersection(set(seq_id_list)):
                print('ERROR: Default values must be in sequence list');
            else:
                self.id_bright_DEFAULT = id_bright;
                self.id_meas_DEFAULT = id_meas;
    
                self.seq_string+='SEQUENCES \n';
                for s, seq_name in zip(seq_id_list, seq_file_list):
                    self.seq_string+=s+'"'+seq_name+'"\n';
                self.seq_string+='SEQUENCES_END\n\n';
                
            
                
        