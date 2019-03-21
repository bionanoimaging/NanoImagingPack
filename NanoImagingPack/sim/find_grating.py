
import numpy as np;
import matplotlib.pyplot as plt;
import datetime
import time
from copy import deepcopy
from ..config import __DEFAULTS__;


class PARA_SET:
    '''
    Some simple class to store parameters of one gratingset
    
    
    One parameter set instance contain:
                - a fixed wavelengths
                - a fixed starting angle
                - a list of possible grating parameters  -> i.e. dimension 4 X x where x is different paramter combinatinos 

                Hence: each para_set belongs to one image displayed on the slm (neglecting phases, they don't change the diffraction behaviour)                
    
    '''
    def __init__(self,wavelength=0, angle=0, para_list=np.asarray([0])):
        if type(para_list) != np.ndarray:
            print('Warning: Parameter Array is not a Numpy array -> Recasting it!!!')
            para_list = np.asarray(para_list); 
        self.wavelength = wavelength;
        self.angle = angle;
        self.para_list = para_list;
        self.opt_para = np.asarray([]);
        self.opt_ratio = 0;
        self.valid = False;
    def disp(self):
        print('');
        print('Elements of parameter set:');
        print('==========================');
        print('');
        print('Wavelength: '+str(self.wavelength));
        print('Angle: '+str(self.angle));
        if np.ndim(self.para_list)==1:
            print('Size of parameter array: 1');
        else:
            print('Size of parameter array: '+str(np.size(self.para_list, axis =1)));    
    def get_first_period(self):
        if np.ndim(self.para_list) ==1: 
            pl = self.para_list;  
        else:
            pl = self.para_list[:,0];
        return(calc_per(*pl));
    
    def filter_sum_ratios(self, num_phase, num_dir, dim_slm, generation, pxs, f, h, square  =True):
        
        if self.para_list.ndim == 1:
            self.sum_ratio = get_ratio_summed_imgs(self.para_list, num_phase = num_phase,num_dir = num_dir,dim_slm = dim_slm, generation = generation,pxs= pxs,f=  f,wl= self.wavelength, h = h, square = True);
            #self.sum_ratio = np.asarray(self.sum_ratio);
            if self.sum_ratio <0.05:
                pass;
            else:
                self.para_list = np.asarray([]);
            
        else:                    
            self.sum_ratio = np.zeros(self.para_list.shape[1]);
            count = 0;
            for paras in self.para_list.transpose():
                #print(paras)
                self.sum_ratio[count] = get_ratio_summed_imgs(paras, num_phase = num_phase,num_dir = num_dir,dim_slm = dim_slm, generation = generation,pxs= pxs,f=  f,wl= self.wavelength, h = h, square = True)
                count += 1;
            condition = self.sum_ratio<0.05;        
            index_list = np.squeeze(np.asarray(condition.nonzero()));
            self.para_list = self.para_list[:,index_list];
            
    def get_ratio_unwanted_order(self,w_gauss,px_size,h,dim_slm,f, num_dir, num_phase, dbg=-1):
        '''
        This function computes the ratio of unwanted orders for all parameters in the given PARA_SET
        It is the ratio between the integral of the holes of where nothing should be transmitted through the fourier mask and the integral of that holes, where the WANTED orders should pass
        Thus unwanted orders that pass through the holes of the wanted ones will create errors! Consequently make sure, that the holes are NOT TOO BIG!
        
        w_gauss: 1/e^2 radius of the Gaussian beam in cm (can be list if different in different directions)
        px_size: pixel_size in um
        h: hole diameter in mm
        dim_slm: dimensions of the slm in [pixel,pixel] 
        num_phase: numbers of phases
        num_dir: numbers of directions
        f = focal length of the collimating lens in mm
        dbg: Debugmode: -1 to switch of (is default) otherwise it shows the particular grating of parameter set "dbg" in para_list
        '''
        
        from ..coordinates import xx, yy;
        from ..functions import gauss2D;
        from ..transformations import ft;
        from .create_grating import generate_grating;
        import numbers;
    
        if isinstance(w_gauss, numbers.Number):
            w_gauss = [w_gauss, w_gauss];           # in case that only one noumber given -> so two different directions possible!
        print('Finding ratio of wanted and unwanted orders for: angle = '+str(self.angle)+" and wavelength = "+str(self.wavelength)+" nm");        
        print('Creating Illumination pattern and masks ...')
        MyIllu = gauss2D(sigma_x = w_gauss[0]*1E4/(2*px_size), sigma_y = w_gauss[1]*1E4/(2*px_size))(xx(dim_slm), yy(dim_slm));
        #compute the fouriermask for the given angle 
        #MyMasks[0] the wanted direction
        #MyMasks[1] the unwanted directon
        MyMasks = generate_mask(num_dir, self.angle, dim_slm, h,self.wavelength,self.get_first_period(),px_size, f);
        print('Generating Gratings and computing FT')
        if self.para_list.ndim == 1:                # this is necessary to exploit numpy broadcasting: in order to compute 3Dim and 2Dim array, Dim1 and 2 must be equal
            grats = generate_grating(self.para_list,0, num_phase,dim_slm)*MyIllu;
            MyFT = ft(grats, axes = (0,1), ret = 'abs');
        else:
            grats = np.rollaxis(generate_grating(self.para_list,0, num_phase,dim_slm),2)*MyIllu;
            MyFT = ft(grats, axes = (1,2), ret = 'abs');
        print('Computing Rations...')        
        Wanted = np.abs(MyFT)*MyMasks[0];
        Unwanted = np.abs(MyFT)*MyMasks[1];
        if self.para_list.ndim == 1:
            Wanted = np.sum(Wanted, axis =(0,1));
            Unwanted = np.sum(Unwanted, axis =(0,1));
        else:
            Wanted = np.sum(Wanted, axis =(1,2));
            Unwanted = np.sum(Unwanted, axis =(1,2));
        Ratio = Unwanted/Wanted;
        if np.ndim(self.para_list)==1:
            self.opt_para = self.para_list;
            self.opt_ratio = Ratio;
        else: 
            self.opt_para = self.para_list[:,np.argmin(Ratio)];
            self.opt_ratio= np.min(Ratio)
        print("Ratio of Unwanted and Wanted orders: "+str(np.min(Ratio)));
        if dbg != -1:
            if dbg > (np.size(self.para_list)+1): 
                print("dbg value to big! Setting to: "+str(np.size(self.para_list)));
                dbg = np.size(self.para_list);
            from ..view import view;
            print('Number directions: '+str(num_dir));
            print('Desired angle: '+str(self.angle));
            print('Dimensions SLM: '+str(dim_slm));
            print('Hole diameter in mm:' +str(h));
            print('parameter list: '+str(self.para_list))
            print('Wavelengths: '+str(self.wavelength));
            print('Period in pixels: '+str(self.get_first_period()));
            print('pixelsize in um: '+str(px_size))
            print('Focal length in mm: '+str(f));
            print('Wanted energy '+str(Wanted))
            print('Unwanted energy '+str(Unwanted))
            print('Ratio: '+str(Ratio))
            print('Grating image shape' + str(grats.shape))
            view(MyMasks[0], title = 'Wanted mask');
            view(MyMasks[1], title = 'UnWanted mask');
            if np.ndim(self.para_list)==1:
                view(grats, title =  'grating')        
                view(MyFT, title =  'FT grating', s= 'log')
                view(Unwanted,title='Unwanted transmission', s = 'log')        
                view(Wanted,title='Wanted transmission', s = 'log')        
            else: 
                view(np.rollaxis(grats,0,3), title =  'grating')        
                view(np.rollaxis(MyFT,0,3), title =  'FT grating', s= 'log')
                view(np.rollaxis(Unwanted,0,3),title='Unwanted transmission', s = 'log')        
                view(np.rollaxis(Wanted,0,3),title='Wanted transmission', s = 'log')        
        return(Ratio)

def get_ratio_summed_imgs(paras, num_phase,num_dir,dim_slm, generation,pxs, f,wl, h, square = True):
    from .create_grating import generate_grating, generate_grating2;
    for d in dim_slm:
        if d<300:
            raise ValueError('For computing the ratio the minimum dimension for the slm is 300X300 pixels');
    
    per = calc_per(*paras);
    ang = calc_orient(*paras[2:]);
    im = None;
    for i in range(num_phase):
        if generation == 1:
            g = generate_grating(paras, i, num_phase, dim_slm);
        elif generation == 2:
            g = generate_grating2(paras, i, num_phase, dim_slm);
        if im is None:
            im = g
        else:
            im = im.cat(g,-3);
    MyMask_W, MyMask_UW = generate_mask(num_dir= num_dir, wanted_dir =ang , dim_slm = dim_slm, h_diameter=h,wl = wl,period = per ,pixelsize =pxs, f = f, zero = False);
    mask = MyMask_W;
    ftims = im.ft2d()
    filtered = mask*ftims
    fim = filtered.ift2d(ret = 'real');
    if square:
        fim **=2;
    fim = fim[:,100:dim_slm[0]-100,100:dim_slm[1]-100];
    ptp = fim.ptp();
    s = fim.sum(0);
    s = s.ptp()
    return(s/ptp);
        
def gcd(a, b):
    '''
        computes the greatest common devisor:
               input variables have to casted as numpy array:
    '''
    if np.ndim(a)==0:
        a = np.expand_dims(a,axis=0);
    if np.ndim(b)==0:
        b= np.expand_dims(b,axis=0);
    a, b = np.broadcast_arrays(a, b)
    a = deepcopy(a)
    b = deepcopy(b)
    pos = np.nonzero(b)[0]
    while len(pos) > 0:
        b2 = b[pos]
        a[pos], b[pos] = b2, a[pos] % b2
        pos = pos[b[pos]!=0]
    return a


def lcm(a,b):
    '''
        Computes the least common multiple of a and b
    '''
    if type(a) != np.ndarray:
        a = np.asarray(a);
    if type(b) != np.ndarray:
        b = np.asarray(b);
    res = (a*b//gcd(a,b))
    return res;

def calc_per(ahx,ahy,apx,apy):
    return(np.sqrt(ahx**2+ahy**2)*np.abs(np.sin((calc_orient(apx,apy)-calc_orient(ahx,ahy))*np.pi/180)))
    #    return (ahx*apy-ahy*apx)/np.sqrt(ahx**2+ahy**2);

def calc_orient(apx,apy):
    if type(apx) != np.ndarray:
        apx = np.asarray(apx);
    if type(apy) != np.ndarray:
        apy = np.asarray(apy);
    apx=apx.astype(float);
    apy=apy.astype(float);
    np.seterr(divide='ignore', invalid='ignore')
    try:
        phi = np.arctan(apy/apx)*180/np.pi;
        phi = np.asarray(phi)
    except ZeroDivisionError:
        phi = 90.;
        phi = np.asarray(phi)
    phi[np.isnan(phi)] = 90;  #This is necessary, since deviding a numpy array by 0 does not produce an error but a warning and creates a nan!
    np.seterr(divide='warn', invalid='warn')
    return(phi)

def check_phase_steps(num_phase, para,PhaseCheckMethod=2):
    '''
      Checks if equidistant phase steps are possilbe 
          -> input: number of phases and list with the parameters (axis0: parameters, axis1: list)
          -> output: list with parameters, where phase steps are possible
          -> PhaseCheckMethod=2
    '''
    ahx = para[0,:];    #h_x
    ahy = para[1,:];    #h_y
    apx = para[2,:];    #theta_x
    apy = para[3,:];    #theta_y
    np.seterr(divide='ignore', invalid='ignore')

    if PhaseCheckMethod == 2:
        apx = apx//gcd(apx,apy);
        apy = apy//gcd(apx,apy);
        vert = (ahx == 0)*(np.mod(ahy,num_phase)==0)+(ahx!=0)*(np.mod((lcm(np.abs(ahx),np.abs(apx))*(apy/apx-ahy/ahx)) ,num_phase)==0)
        hor = (ahy==0)*(np.mod(ahy,num_phase)==0)+(ahy!=0)*(np.mod((lcm(np.abs(ahy),np.abs(apy))*(apx/apy-ahx/ahy)) ,num_phase)==0)
    elif PhaseCheckMethod == 1:    
        vert = (ahx == 0)*(np.mod(ahy,num_phase)==0)+(ahx!=0)*(np.mod(lcm(np.abs(ahx),np.abs(apx))*apy/apx-ahy/ahx ,num_phase)==0)
        hor = (ahy==0)*(np.mod(ahy,num_phase)==0)+(ahy!=0)*(np.mod(lcm(np.abs(ahy),np.abs(apy))*apx/apy-ahx/ahy ,num_phase)==0)
    else:
         raise ValueError('Wrong PhaseStepMethod: Must be 1 or 2 -> Check DocString for help!');
    vert[np.isnan(vert)] = False;
    hor[np.isnan(hor)] = False;
    np.seterr(divide='warn', invalid='warn')
    return  para[:,np.nonzero((hor+vert)*1)[0]];            

def search_for_matching_k(start, end, num_phases, k, dk, fixed_angle_set = None, PhaseCheckMethod=2):
    '''
        This function searches a given pixelspace from start to end in steps of one for phi_x, phi_y, hx and hy
        and returns an array consisting of arrays [ahx, ahy, apx, apy] that contain the pixel values in order to fullfill the grating condition
        and the phase step condition
        
        use fixed_angle_set to define if the angle parameter shall be fixed!
                -> either give -1 (no fixed angle set) or [apx, apy]
    '''
    
    '''
        DEFINE PIXELSPACE TO ANALYZE FOR GRATINGS
        
    '''
    ah_x = np.arange(start,end,1);
    ah_y = np.arange(start,end,1);
    if fixed_angle_set is None:
        ap_x = np.arange(start,end,1);
        ap_y = np.arange(start,end,1);
    else:
        ap_x = np.arange(fixed_angle_set[0],fixed_angle_set[0]+1,1);
        ap_y = np.arange(fixed_angle_set[1],fixed_angle_set[1]+1,1);
    AHX,AHY,APX,APY = np.meshgrid(ah_x,ah_y,ap_x,ap_y)
    ahx = np.ravel(AHX);
    ahy = np.ravel(AHY);
    apx = np.ravel(APX);
    apy = np.ravel(APY);
    '''
        Compute grating period
    '''
    period = calc_per(ahx,ahy,apx,apy)   
    '''
     find parameter combinations, that produce the correct grating period
    '''
 
    condition= (np.abs(period) > (k-dk/2))*(np.abs(period)<(k+dk/2));
    
    index_list = np.squeeze(np.asarray(condition.nonzero()))    #gibt alle indizes zurück, bei denen die Bedinung, in dem Fall die Passende Gitterkosntante, erfüllt ist
    ahx = ahx[index_list]
    ahy = ahy[index_list]
    apx = apx[index_list]
    apy = apy[index_list]
    res = np.concatenate((ahx,ahy,apx,apy), axis =0);
    res=np.reshape(res, (4, np.size(index_list)));
    '''
    Fullfill phase step conditidon
    '''
    res2 = check_phase_steps(num_phases, res, PhaseCheckMethod)
    return(res2)

def search_direction(start_ang, d_ang, num_dir, para, fixed_angle_set= None):
    '''
        Searches all parameter sets that produce gratings with the given orientation. 
        The function returns a list with arrays of parameter sets. Each element in the list stands for one orientation
    '''
    
    apx = para[2,:];    #phi_x
    apy = para[3,:];    #phi_y
    para_list=[];
    
    if fixed_angle_set is None:
        orientation = calc_orient(apx,apy);
        for i in np.arange(0,num_dir,1):
            angle = i*180/num_dir+start_ang;
            if angle > 90: angle = angle -180;
            condition = (orientation > (angle-d_ang/2))*(orientation<((angle+d_ang/2)));
            index_list = np.squeeze(np.asarray(condition.nonzero()))    
            res = para[:,index_list]
            para_list.append(res)     
    else:
        condition = (apx == fixed_angle_set[0])*(apy == fixed_angle_set[1]);
        index_list = np.squeeze(np.asarray(condition.nonzero()))    
        res = para[:,index_list]
        para_list.append(res)     
    return(para_list)

def generate_mask(num_dir, wanted_dir, dim_slm, h_diameter,wl,period,pixelsize, f = 300, zero = False):
    '''
        Generate the fouriermask:
            num_dir: number of directions
            start_dir: desired direction
            dim_slm: dimensions of the slm (required for pixel size of the mst) in pixel
            h_diameter: hole diameter in mm
            wl: wavelength in nm
            period: grating period in pixelsize
            pixel_size in um
           
            f: focal lenght of collimating lens in mm Standard is 300 mm
            zero: zeros order?
            
            The output mask is in pixels in the Fourierspace
    '''
    from ..mask import create_circle_mask;
    from ..coordinates import bfp_coords;
    import numpy as np;
    bfp_xx = bfp_coords(dim_slm,pxs = pixelsize, wavelength = wl/1000, focal_length = f, axis = 0);           # coordinates in back focal plane
    bfp_yy = bfp_coords(dim_slm,pxs = pixelsize, wavelength = wl/1000, focal_length = f, axis = 1);
    # pixelsize in the BackfocalPlane in mm    
    bfp_px_size = ((np.max(bfp_xx)-np.min(bfp_xx))/bfp_xx.shape[0],(np.max(bfp_yy)-np.min(bfp_yy))/bfp_yy.shape[1]);
    d= wl/1000*f/(period*pixelsize);                #distance in the backfocal plane in mm
    
    # compute part of the mask for the wanted transmissions 
    x_pos = d*np.sin(wanted_dir*np.pi/180)/bfp_px_size[1]         # indexes changed due to new indexing
    y_pos = d*np.cos(wanted_dir*np.pi/180)/bfp_px_size[0]
    mask_wanted_dir  = create_circle_mask(dim_slm, maskpos = (x_pos,y_pos), radius = (h_diameter/(2*bfp_px_size[0]),h_diameter/(2*bfp_px_size[1])));
    mask_wanted_dir += create_circle_mask(dim_slm, maskpos = (-x_pos,-y_pos), radius = (h_diameter/(2*bfp_px_size[0]),h_diameter/(2*bfp_px_size[1])));
    # compute the mask for the unwanted transmission
    mask_unwanted_dir = np.zeros(dim_slm);
    for i in range(1,num_dir):
        x_pos = d*np.sin((wanted_dir+180/num_dir*i)*np.pi/180)/bfp_px_size[0]
        y_pos = d*np.cos((wanted_dir+180/num_dir*i)*np.pi/180)/bfp_px_size[1]
        mask_unwanted_dir += create_circle_mask(dim_slm, maskpos = (x_pos,y_pos), radius = (h_diameter/(2*bfp_px_size[0]),h_diameter/(2*bfp_px_size[1])));
        mask_unwanted_dir += create_circle_mask(dim_slm, maskpos = (-x_pos,-y_pos), radius = (h_diameter/(2*bfp_px_size[0]),h_diameter/(2*bfp_px_size[1])));
    if zero:
        mask_unwanted_dir += create_circle_mask(dim_slm, maskpos = (0,0), radius = (h_diameter/(2*bfp_px_size[0]),h_diameter/(2*bfp_px_size[1])));
    return([mask_wanted_dir, mask_unwanted_dir])

def clear_para_list(pl, al, n_dir,lam):
    '''
        This removes all entries from the parameter list, where the angle is not availible for every wavelength:
        pl: parameter list
        al: angle list:
        n_dir: number of directiones
    '''
    if len(al) >0:
        wl_set = set(lam);
        new_l = []
        for ang in al:
            ang_set = set(ang+n*180/n_dir for n in range(n_dir ))
           # print(ang_set)
            ang_set_ok = 1;
            for angle in ang_set:
                if len(list(filter(lambda x: x.angle == angle, pl)))>0:
                    ang_set_ok*=1;
                else:
                    ang_set_ok*=0;
            
            new_l += [p for p in pl if (ang_set_ok > 0) and p.angle in ang_set]
            
            anglist =[];
        for p in new_l:
            if p.angle not in anglist: anglist+=[p.angle];
        new_l2 = [];
        for el in anglist:
            sub_list = list(filter(lambda x: x.angle == el, new_l))
            wl_set_ok = 1;
            for wavelength in wl_set:
                if len(list(filter(lambda x: x.wavelength == wavelength, sub_list)))>0:
                    wl_set_ok *=1;
                else:
                    wl_set_ok *=0;
            if wl_set_ok == 1:
                new_l2+=sub_list;
    else:
        new_l2 = [];
    return(new_l2);


#def clear_para_list(pl, al, n_dir):
#    '''
#        This removes all entries from the parameter list, where the angle is not availible for every wavelength:
#        pl: parameter list
#        al: angle list:
#        n_dir: number of directiones
#    '''
#    return(list(filter(lambda x: np.mod(x.angle,180/n_dir) in al, pl)))

def optimize_grating_sum(para_sets, num_phase, num_dir, dim_slm,h,generation, px_size, f ):
    count = 0;
    for p in para_sets:
        
        p.filter_sum_ratios(num_phase, num_dir, dim_slm, generation, px_size, f, h, square  =True)
        if p.para_list.ndim == 2:
            print('Filtering grating sum ... set '+str(count)+' / '+str(len(para_sets))+'  --> '+str(p.para_list.shape[1])+' sets possible');
        elif p.para_list.ndim == 1 and p.para_list.size!=0:
            print('Filtering grating sum ... set '+str(count)+' / '+str(len(para_sets))+'  --> 1 set possible');
        elif p.para_list.size == 0:
            print('Filtering grating sum ... set '+str(count)+' / '+str(len(para_sets))+'  --> 0 sets possible');
        count +=1;
    para_sets = [p for p in para_sets if p.para_list.size > 0]
    return(para_sets);
    #get_sum_ratios(self, num_phase, num_dir, dim_slm, generation, pxs, f, h, square  =True):
    

def find_optimum_set(pl, w_gauss, px_size, h, angle_array, num_dir,num_phases, dim_slm=[1024,1024], f=300, criterion = 'maximum'):
    average_list = []
    max_list = []
    for p in pl:
        p.get_ratio_unwanted_order(w_gauss,px_size,h,dim_slm,f,num_dir, num_phases)
    for angle in angle_array:
        new_list = list(filter(lambda x: np.mod(x.angle, 180//num_dir)==angle, pl));
        if len(new_list)!= 0:                       
            average_list.append(sum(list(map(lambda x: x.opt_ratio, new_list)))/len(new_list))
            max_list.append(max(list(map(lambda x: x.opt_ratio, new_list))));
    if criterion == 'average':
        opt_angle= angle_array[np.argmin(average_list)]
    if criterion == 'maximum':
        opt_angle= angle_array[np.argmin(max_list)];
    print('')    
    print('Optimum starting angle: '+str(opt_angle));
    print('Maximum ratio unwanted orders: '+str(min(max_list)));
    print('Average ratio unwanted orders: '+str(min(average_list)));
    print('====================================================')
    opt_list = list(filter(lambda x: np.mod(x.angle, 180//num_dir)==opt_angle, pl))
    for el in opt_list: print(str(el.wavelength)+' ; '+str(el.angle)+'  ;  '+str(el.opt_para));
    return(opt_list)

def save_gratings(ol, period, error_period, num_dir, num_phase,px_size, w_gauss, h, dim_slm, error_angle,f, path,bfp_fill,name='para_data.txt', optimized_list = True):
    ts = time.time();
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    meta_text = 'Gratings computed with Christian\'s Python code \n'
    meta_text = meta_text+st+'\n\n'
    meta_text = meta_text+'Desired Period ('+str(ol[0].wavelength)+'nm) : '+'\t'+str(period)+' Pixels\n'
    meta_text = meta_text+'Maximum Error Period: '+'\t'+str(error_period)+' Pixels\n';
    meta_text = meta_text+'Maximum Error Angle: '+'\t'+str(error_angle)+' Degree\n';
    meta_text = meta_text+'Number Phases: '+'\t'+str(num_phase)+'\n';
    meta_text = meta_text+'Number Directions: '+'\t'+str(num_dir)+'\n';
    meta_text = meta_text+'Pixel size: '+'\t'+str(px_size)+' um\n';
    meta_text = meta_text+'Spot diameter on SLM (1/e^2 radius, gaussian): '+'\t'+str(w_gauss)+' cm\n';
    meta_text = meta_text+'Diameter of holes in mask: '+'\t'+str(h)+' mm\n';
    meta_text = meta_text+'Focal length of collimating lens: '+'\t'+str(f)+' mm\n';
    meta_text = meta_text+'Dimensions SLM: '+'\t'+str(dim_slm[0])+' X ' +str(dim_slm[1])+' Pixels\n';
    meta_text += 'BFP Filling for first wavelength: \t'+str(bfp_fill)+'% \n\n';
                                                     
    meta_text = meta_text+'Parameters (Check Ronny\'s Publication to understand [OPT EXPR Vol22 No17 2014]) \n';
    
                                                  
    meta_text = meta_text+'Wavelength [nm] \t angle [DEG] \t h_x [px] \t h_y [px] \t theta_x [px] \t theta_y [px] \t Ratio of unwantet orders \t period \t angle \n';
    for p in ol:
        if optimized_list:
            meta_text = meta_text+str(p.wavelength)+' \t '+str(p.angle)+' \t '+str(p.opt_para[0])+' \t '+str(p.opt_para[1])+ ' \t '+str(p.opt_para[2])+' \t '+str(p.opt_para[3])+' \t '+str(p.opt_ratio)+' \t ' + str(calc_per(p.opt_para[0],p.opt_para[1],p.opt_para[2],p.opt_para[3]))+ ' \t '+ str(calc_orient(p.opt_para[2], p.opt_para[3])) + '\n';
        else:
            if p.para_list.ndim == 1:
                meta_text = meta_text+str(p.wavelength)+' \t '+str(p.angle)+' \t '+str(p.para_list[0])+' \t '+str(p.para_list[1])+ ' \t '+str(p.para_list[2])+' \t '+str(p.para_list[3])+' \t '+str(-1)+' \t ' + str(calc_per(*p.para_list))+ ' \t '+ str(calc_orient(p.para_list[2], p.para_list[3])) + '\n';
            elif p.para_list.ndim ==2:
                for p2 in p.para_list.transpose():
                    meta_text = meta_text+str(p.wavelength)+' \t '+str(p.angle)+' \t '+str(p2[0])+' \t '+str(p2[1])+ ' \t '+str(p2[2])+' \t '+str(p2[3])+' \t '+str(-1)+' \t ' + str(calc_per(*p2))+ ' \t '+ str(calc_orient(p2[2], p2[3])) + '\n';             
    text_file = open(path+name,'w')
    text_file.write(meta_text);
    text_file.close();
    '''
    zero = np.zeros(dim_slm)+1;

    plt.imsave(path+'bright.tif',zero.astype(int),format = "TIFF", cmap = 'Greys_r') 
    for p in ol:
        for phase_nr in np.arange(0,num_phase,1):
            grat = generate_grating(p.opt_para,phase_nr, num_phase, dim_slm)
            grat.astype(int)
            name = 'Grating_wl'+str(p.wavelength)+'_ang'+str(int(p.angle))+'_ph'+str(phase_nr)+'.tif'
            plt.imsave(path+name,grat,format = "TIFF", cmap = 'Greys_r')    
    '''
    return();

     
def create_para_list(start, end, k0, dk0_R, d_ang, lam, num_dir, num_phase,dim_slm,h,generation,px_size,f,opt_grating_sum, fixed_angle = [-1], fixed_angle_set = None,PhaseCheckMethod=2):
    if (fixed_angle == [-1]):
        angle_array = np.arange(1,180/num_dir,1);       #scan whole start angle range
    else:
        angle_array = np.asarray(fixed_angle);
    if fixed_angle_set is not None:
        if __DEFAULTS__['SHOW_GRAT_SEARCH_INFO']:
                print('Grating vector describing the angle was fixed to '+str(fixed_angle_set));
        angle=calc_orient(fixed_angle_set[0],fixed_angle_set[1]);
        angle_array = np.asarray([angle]);
    para_list = [];
    error=0;
    for wl in lam:
        k = k0*wl/lam[0];
        if __DEFAULTS__['SHOW_GRAT_SEARCH_INFO']:
            print('');
            print('Searching parameter sets for wavelength: '+str(wl)+' nm!');
            print('Desired grating period: '+str(k)+' pixels');
            print('Searching grating period ...');    
        res = search_for_matching_k(start, end, num_phase, k, k*dk0_R,fixed_angle_set = fixed_angle_set, PhaseCheckMethod=PhaseCheckMethod);
        if __DEFAULTS__['SHOW_GRAT_SEARCH_INFO']:
            print(str(np.size(res,axis=1)) + ' sets found with matching grating period');
            print('Scanning start angles and searching for directions...')
        angle_arr=np.asarray([])   # I dont't know what that was supposed to be!
        
        #print(angle_array.shape)
        for start_ang in angle_array:                                        # Scan every possible starting angle -> 
            liste = search_direction(start_ang, d_ang, num_dir, res, fixed_angle_set)   
            tester = 1;  # Tells you if there is at least 1 parameter set for each direction begining at starting angle
            for i in np.arange(0,num_dir,1):
                tester = tester*(np.size(liste[i])!=0)
            if tester == 1:
                angle_arr=np.append(angle_arr,start_ang)
                for i in np.arange(0,num_dir,1):
                    para_list.append(PARA_SET(wavelength=wl, angle=start_ang+i*180/num_dir, para_list=liste[i]));
        if np.size(angle_arr) == 0:
            print('NO MATCHING GRAITINGS FOUND: RAISE SEARCH RANGE');
            angle_array = angle_arr;
            error =-1
            break;
        else:
            if __DEFAULTS__['SHOW_GRAT_SEARCH_INFO']:
                print(str(np.size(angle_arr))+' possible starting angles found:')
                print(angle_arr);
            angle_array = angle_arr;
    
    return(clear_para_list(para_list, angle_array, num_dir,lam), angle_array,error)  # Remove all entries from the parameter list, where not all direction for all wavelengths are possible for a given starting angle


def get_period(lam, eta, pixelpitch=8.2, NA = 1.46, magnification=76.8):
    '''
        Compute the grating period for the given illumination level of the BFP (e.g. 90 %)
        lam:   wavelength in nm
        eta: illumination level (Number between 0 and 1 -> e.g. 0.9 is 90 % of BFP of Objective)
        pixelpitch is defined as 8.2 if not given (4DD QXGA)
        NA is defined as 1.46 if not given
        magnification is defined as 76.8 if not given
    '''
    period = lam*magnification/(1000*pixelpitch*eta*NA); # note the 2's where canceled out
    return(period)

    

def create_grating_param_file(path, start = 10, end = 50, num_dir = 3, num_phase = 3, wavelength = [488, 561, 638, 405], error_period = 0.01, error_angle = 0.1,fixed_angle=None, px_size = 8.2, w_gauss = 0.5,h =0.3, dim_slm = [1024,1024], f = 250, NA = 1.46, magnification = 76.8, BFP_filling = 83.4, period =None, fixed_angle_set = None, PhaseCheckMethod=2 ,optimize_for_unwanted_orders = True, opt_grating_sum = False, generation = 1):
    '''
    Created on Wed Nov 16 19:26:01 2016
    
    @author: ckarras
    
    This script generates a parameter set for SLM gratings as presented in Ronnys Publication [OPT EXPR Vol22 No17 2014]
    
    The results will be saved in a Textfile
    
    The algorithm is Brute Force, sanning a 4 Dimensional pixel space (ahx,ahy,apx,apy). The range of the space is given by the "Start" and "End" values
    
    
    
    *********************
    *                   *
    *   !!!WARNING!!!   *
    *                   *
    *********************
    
    !!! A too large space will cause the system to crash due to an memory overflow!!!
    A too small space might result in no matching gratings.
    For a computer with  16 GB memory I would not exceed 40 as difference between start and end value
    
    
    
    Comment: this module doesn't use the other basic modules from the NanoImagingPack Package. This is because I wrote it before I created the package and now I'm to lazy to change the code (also: it works and never change a running system)
    
    Parameters of the function:
        
        path:                           FOLDER of the text file with the parameters, make sure it is existing!!!
        start/end:                      defines the range of the 4D space (ahx,ahy,apx,apy). Be careful of not exaggerating! 
        num_dir:                        Number of directions
        num_phase:                      Number of phases
        wavelength:                     List of the desired wavelength. The k vector is computed based on the first element in the list 
        error_period:                   maximum error of the grating period from the desired value (given by the position of the orders in the BFP) in per cent of the computed or given period 
        error_angle                     maximum deviation of the angles from the desired direction in degrees
        fixed_angle:		             Angle of the first direction. Default is None. This causes the script to scan for the optimum angle by minimizing the unwanted orders
        px_size:                        Pixelsize of the SLM in um
        w_gauss:                        Gaussian widht of the illumination (1/e^2 in cm)
        h:                              hole diameter of the fouriermask in mm
        dim_slm                         dimension of the slm in pixel  (list of 2 elements, !!! only works for square shape rigth now!!!)
        f:                              focal length of the collimation lens (mm)
        NA:                             Numerical aperture of the objective
        magnification:                  Magnification of the objective (of the grating on the SLM)
        BFP_filling:                    Desired position of the orders in the BFP of the objective (in %), corresponds to first wavelength in wavelengths list
        period:                         Desired period in pixels
        
        period or BFP filling has to be given, the other one has to be None type -> Thus the period is defined!
      
        fixed_angle_set                give a list or a tuple with the fixed vector defining the angle ( a_theta in ronnys paper) -> no angle scanning will be performed
        
        PhaseStepMethod                1 or 2:
                                            There is a discrepancy between the Phasestepmethod in Ronny's Paper and in the original Matlab code
                                            With 1 or 2 you can choose either one of them!
                                            The flag is used in function "check_phase_steps"
                                            
                                            1.) Method as descriped in the paper
                                                       +   Accurate with respect to the phase step check (nip.sim.test_grat)
                                                       +   As it has been published it SHOULD (!!!) give accurate phase steps
                                                       -   A very large scanning range might be necessary as it is a very harsh criterium
                                            
                                            2.) Method as in the old Matlabcode from Ronny
                                                       +   A looser criterium as 1 and thus requires smaller scanning range
                                                       -   However I generally did not encounter any problems yet, it is not one houndret percent clear what happened here and if it always gives correct phase steps. In case you use this: Check the gratings afterwards
        
        optimize_for_unwanted_orders    do you want to optimize for unwanted orders? if not, the whole list will be stored!
        opt_grating_sum                 do you want to optimize the grating sums? In this case for each potential grating set, the num_phase gratings will be computed, filtered and summed up. The ratio between the summed peak-to-peak and the ptp-value for one grating is below 5% the parameter set will be excepted. Otherwise it will be expelled.
        generation                      for the opt_grating_sum -> how to shift the phases? 1: shift phases between 0 and py, 2: shift hte phases between 0 and 2pi;
        
      Returns the path of the parameter file so it can easily be read afterwards
      
      NOTE:
          The angle is between the y-axis and the direction! -> i.e. x = d*sin(phi) and y = d*cos(phi)
      
      
    To create images from the parameters you can use the "Create_RepZ" Script
        -> As soon as I have time I will make that one produce proper RepZ files for the SLMs
    
    Regards, Christian
    ''' 
    
    lam = np.asarray(wavelength);
    if fixed_angle == None:
        fixed_angle = [-1];
    else:
        fixed_angle = [fixed_angle];
    
    if fixed_angle_set is not None:
        print('Fixed angle set given -> only 1 Direction possible! -> resetting num_dir to 1');
        num_dir = 1;          
    
    #lam = np.asarray([488,561,638]);
    #lam = np.asarray([488,561]);
    
    #path = 'D:/Fast_SIM/Running_Order_Sets/Parul_Test_grating/'              # Select your path here
    #path ='Z:/FastSIM Setup/Running_Order_Sets/Alginment/29-03-2017/';
    
    
    if (period == None and BFP_filling == None) or (period != None and BFP_filling != None):
        print('Error in defining period: Either period or BFP_filling has to be given! Not both!');
        return('ERROR');
    else:
        if period == None:
            period = get_period(lam[0], BFP_filling/100, px_size, NA, magnification);
        else:
            try:
                BFP_filling = lam[0]*magnification/(1000*px_size*period*NA)*100;
            except:
                BFP_filling = None;
        
        error =0;
       
        error_period = period*error_period/100;   #convert error period from percentage value to pixel value
        print('Error Period in pixels: '+str(error_period));
      
        name='para_'+str(np.round(period,3))+'_'+str(num_phase)+'phases_'+str(num_dir)+'Dir.txt';
        #name='QXGA_TEST_PARA.txt';
        para_list, angle_array,error = create_para_list(start, end, period, error_period, error_angle, lam, num_dir, num_phase,dim_slm,h,generation,px_size,f,opt_grating_sum, fixed_angle, fixed_angle_set = fixed_angle_set, PhaseCheckMethod=PhaseCheckMethod);
        #return(para_list,angle_array)
        if opt_grating_sum:
            para_list = optimize_grating_sum(para_list, num_phase, num_dir, dim_slm,h,generation, px_size, f);
        #return(para_list, angle_array)
        para_list = clear_para_list(para_list, angle_array, num_dir, lam)
        if len(para_list) == 0:
            error = -1;
            print('No elements after para list creation');                                                       
        if (error ==0):
            if optimize_for_unwanted_orders:
                ol = find_optimum_set(para_list, w_gauss, px_size, h,angle_array,num_dir,num_phase, dim_slm, f);
            else:
                ol = para_list;
            #return(ol)
            save_gratings(ol, period, error_period, num_dir, num_phase,px_size, w_gauss, h, dim_slm, error_angle,f,path,BFP_filling,name, optimized_list = optimize_for_unwanted_orders)
        return(path+name);   

