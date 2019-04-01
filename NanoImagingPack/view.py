#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:32:38 2017

@author: root
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.widgets import Slider

try:
    from .config import __DEFAULTS__

    __VIEWER_MAX_CLICK_TIME__= __DEFAULTS__['VIEWER_MAX_CLICK_TIME']  # Click time for recognizing marker
    __VIEWER_FORMAT_VALUE_STRING__ = __DEFAULTS__['VIEWER_FORMAT_VALUE_STRING']  # Format value for satus bar
    __VIEWER_DOWNCONVERT_DATA__= __DEFAULTS__['VIEWER_DOWNCONVERT_DATA']  # downconverting float or complex data in viewer (speeds up stuff)
    __VIEWER_RATIO_AUTO_ASPECT__= __DEFAULTS__['VIEWER_RATIO_AUTO_ASPECT']  # At which ratio between the both axes for the vier the autoaspect should be used?
    __VIEWER_ASPECT__ = __DEFAULTS__['VIEWER_ASPECT']
    __VIEWER_CONST_ASPECT__ = __DEFAULTS__['VIEWER_CONST_ASPECT']
    __VIEWER_GAMMA_MIN__ = __DEFAULTS__['VIEWER_GAMMA_MIN']
    __VIEWER_GAMMA_MAX__ =__DEFAULTS__['VIEWER_GAMMA_MAX']
    __VIEWER_IMG_ORIGIN__ =__DEFAULTS__['VIEWER_IMG_ORIGIN']
    from .util import get_type

except:
     __VIEWER_MAX_CLICK_TIME__ = 0.5  # Click time for recognizing marker
     __VIEWER_FORMAT_VALUE_STRING__ ='%1.4f'  # Format value for satus bar
     __VIEWER_DOWNCONVERT_DATA__= False
     __VIEWER_RATIO_AUTO_ASPECT__= 20
     __VIEWER_ASPECT__ = 'AUTO'
     __VIEWER_CONST_ASPECT__ =1
     __VIEWER_GAMMA_MAX__ = 10.0
     __VIEWER_GAMMA_MIN__ = 0.1
     __VIEWER_IMG_ORIGIN__ ='upper'


class view:
    """
        A nice little image viewer!

        image is a numpy array
    """
  
    def __init__(self, image, title='',s = 'linear', r=None, show_hist = False, scale16bit = False, mark_clicked_positions = True):
        """
             Nice little viewer

             Note for speeding up stuff, the data are converted to float16 if they are floats and complex64 if set in config

             Parameters:
                 scale16bit:
                        linear (default)   scales linear to 16 bit
                        log                    log scale

                 r: z range in which should be shown (standard is 0 to 100 %) -> e.g. [10,90] will clip regions which are larger than 90 % of maximum or smaller than 10 % of maximum


                              title can be a string or a list with as much string items as slices

                 show_hist: opens second plot with histogramm

                 scale16bit: scales the image between 0 and 65535
                 scales the image as 16 bit image

                 mark_clicked_positions: if true, the position where you clicked at will be marked in the image


            If you click in the image you can acces the coordinates via the self.marker variable.
                     -> central click: set marker, left click: delete marker

            Markerpostions can be accessd by self.markerposlist


             Navigation:
                arrow key left rigth:   move in image stack
                arrow key up down       move in 4th dimension
                b/v                     move in 5th dimension
                m/n                     move in 6th dimension

                c:                      move through complex stack (absolute value, phase, real, imageinary)
                r:                      rescale current image
                a:                      scale through 'log scale' and normal scale
                d:                      switch scale to 16 bit on and off
                x:                      switch global stretch on/off
                g:                      switsch grid on/off (naturally implemented by matplotlib)
                p:                      moving tool (naturally implemented by matplotlib)
                o:                      zooming tool (naturally implemented by matplotlib)
                f:                      Full screen (naturally implemented by matplotlib)
                e:                      export image as displayed in self.export
                s:                      Save current image (naturally implemented by matplotlib)
                y:                      Display histogram of current image. Note: Not of the whole image! For that use teh show_hist option
                h:                      Display naviation commands in terminal

             example:

                 import NanoImagingPack as nip;                #import NanoImagingPack and read test image
                 from NanoImagingPack import view
                 img = nip.readim();
                 img2 = img*2;

                 v = view(img, show_hist = True);               # create new view object v and show image
                 v.draw(img2                            # draw img2 in v
                 v.update(im)                            # update image in current viewer -> similar to draw
                 v2 = view(img2);                        # make new figure img2
                 position = v.marker;                    # store the position were you clicked in the image in the position variable


        """
#        matplotlib.use('Qt5Agg', warn=False) # RH, to make the plot work interactively
        #from matplotlib.widgets import Slider;
        if __VIEWER_DOWNCONVERT_DATA__:
            if np.issubdtype(image.dtype, np.complexfloating):
                self.image = image.astype(np.complex64)
            elif np.issubdtype(image.dtype, np.floating):
                self.image = image.astype(np.float32)
            elif np.issubdtype(image.dtype, np.bool):
                self.image = image.astype(np.uint8)
            else:
                self.image = image
        else:
            self.image = image
        #        self.image = np.swapaxes(self.image,0,1)  # RH removed this 2.2.19
        self.image = np.nan_to_num(self.image, False)
        if len(title) == 0:
            try:
                from .image import image as imtype
                if type(image) == imtype:
                    self.title =image.name
                else:
                    self.title =title
            except:
                self.title =title

        else:
            self.title =title
        self.curr_title = title

        if image.dtype==np.bool:
            self.image = image.astype(np.uint8)
        self.image = self.image.transpose()
        self.image = self.image.swapaxes(0,1)
        if np.issubdtype(self.image.dtype, np.complexfloating):
            self.glob_lims = [[np.min(np.abs(self.image)), np.max(np.abs(self.image))],
                              [np.min(np.angle(self.image)), np.max(np.angle(self.image))],
                              [np.min(np.real(self.image)), np.max(np.real(self.image))],
                              [np.min(np.imag(self.image)), np.max(np.imag(self.image))]]
        else:
            self.glob_lims = [[np.min(self.image), np.max(self.image)]]
        self.s = s
        self.r = r
        self.show_hist = show_hist
        self.scale16bit = scale16bit
        self.fig, self.ax =plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.25)
        self.return_pressed = False
        self.global_stretch = False
        self.export = None
        self.gamma_val = 1.0
        self.complex_set = 0  # for displaying complex arrays:  0 -> absolute, 1 -> phase, 2 -> real, 3 -> imag
        self.connect()
        self.markerposlist = []
        self.marker = (None, None)
        self.mark_clicked = mark_clicked_positions
        self.hist_plot = None
        self.current_slice = 0  # 3rd Dimension
        self.current_page = 0  # 4th Dimension
        self.current_sheet = 0  # 5th Dimension
        self.current_book = 0  # 6th Dimension
        self.click_time =0
        self.fig.canvas.set_window_title(self.__make_title__())
        if np.ndim(self.image) == 2:
            self.im_to_draw = self.image
        elif np.ndim(self.image) == 3:
            self.im_to_draw = self.image[:,:,0]
        elif np.ndim(self.image) == 4:
            self.im_to_draw = self.image[:,:,0,0]
        elif np.ndim(self.image) == 5:
            self.im_to_draw = self.image[:,:,0,0,0]
        elif np.ndim(self.image) == 6:
            self.im_to_draw = self.image[:,:,0,0,0,0]
        else:
            print('Wrong dimensions: minium is 2 maximum is 6')
            return

        axcolor = 'lightgoldenrodyellow'
        self.z_range = [np.min(self.__disp_complex__(self.complex_set)), np.max(self.__disp_complex__(self.complex_set))]
        self.max_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        self.min_slider = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        self.gamma_slider = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
        self.z_max = Slider(self.max_slider, 'max', np.min(self.__disp_complex__(self.complex_set)), np.max(self.__disp_complex__(self.complex_set)), valinit = np.max(self.__disp_complex__(self.complex_set)))  #, valstep=(np.max(image)/np.min(image))/1000);
        self.z_min = Slider(self.min_slider, 'min', np.min(self.__disp_complex__(self.complex_set)), np.max(self.__disp_complex__(self.complex_set)), valinit = np.min(self.__disp_complex__(self.complex_set)))  #, valstep=(np.max(image)/np.min(image))/1000);
        self.gamma = Slider(self.gamma_slider, 'gamma', np.log(__VIEWER_GAMMA_MIN__) , np.log(__VIEWER_GAMMA_MAX__), valinit = 0)  #, valstep=(np.max(image)/np.min(image))/1000);
        self.gamma.valtext.set_text(self.gamma_val)

        self.z_max.on_changed(self.update_scale_max)
        self.z_min.on_changed(self.update_scale_min)
        self.gamma.on_changed(self.update_gamma)

        def __set_auto_aspect__():
            if max(self.im_to_draw.shape)/min(self.im_to_draw.shape) < __VIEWER_RATIO_AUTO_ASPECT__:
                self.aspect = None
            else:
                self.aspect = 'auto'

        if __VIEWER_ASPECT__ == 'AUTO':
            __set_auto_aspect__()
        elif __VIEWER_ASPECT__ ==  'CONST':  
            self.aspect =__VIEWER_CONST_ASPECT__
        elif __VIEWER_ASPECT__ ==  'IMAGE':  
            try:
                from .image import image as imtype
                if type(image) == imtype:
                    self.aspect =self.image.pixelsize[1]/self.image.pixelsize[0]

                else:
                    __set_auto_aspect__()
            except:
                __set_auto_aspect__()

        if self.show_hist == True:
                self.draw_hist()
        self.imgplot = self.ax.imshow(self.__disp_complex__(self.complex_set), cmap = 'gray',interpolation = 'none', aspect = self.aspect, origin = __VIEWER_IMG_ORIGIN__)
        self.ax.autoscale(False)
        if r!= None:
            self.rescale(min(r),max(r))            
        plt.show()

    def update_sliders(self, slider_range = None):
        if slider_range == None:
            minimum = np.min(self.__disp_complex__(self.complex_set))
            maximum = np.max(self.__disp_complex__(self.complex_set))
        else:
            minimum = slider_range[0]
            maximum = slider_range[1]
        self.z_max.valmin = minimum
        self.z_max.valmax = maximum
        self.z_max.ax.set_xlim(minimum,maximum)
        self.z_min.valmin = minimum
        self.z_min.valmax = maximum
        self.z_min.ax.set_xlim(minimum,maximum)
        self.z_min.set_val(minimum)
        self.z_max.set_val(maximum)

    def __make_title__(self):
        title_string = ''
        if type(self.title) == str:
            title_string+= self.title
        elif type(self.title) == list:
            title_string += self.title[self.current_slice]
        if np.ndim(self.image) >=3:
            try:
                title_string += ' '+self.image.dim_description['d2'][self.current_slice]
            except:
                title_string += ' Slice '+str(self.current_slice+1)+'/'+str(np.size(self.image,axis=2))
        if np.ndim(self.image) >=4:
            try:
                title_string += ' '+self.image.dim_description['d3'][self.current_page]
            except:
                title_string += ' Page '+str(self.current_page+1)+'/'+str(np.size(self.image,axis=3))
        if np.ndim(self.image) >=5:
            try:
                title_string += ' '+self.image.dim_description['d4'][self.current_sheet]
            except:
                title_string += ' Sheet '+str(self.current_sheet+1)+'/'+str(np.size(self.image,axis=4))
        if np.ndim(self.image) >=6:
            try:
                title_string += ' '+self.image.dim_description['d5'][self.current_book]
            except:
                title_string += ' Book '+str(self.current_book+1)+'/'+str(np.size(self.image,axis=5))

        title_string += ' SCALE: '+self.s
        if np.issubdtype(self.image.dtype, np.complexfloating):
            complex_title = [' abs ', ' angle ', ' real ',' imag ']
            title_string += complex_title[self.complex_set]
        if self.global_stretch:
            title_string += ' GlobStretch: ON'
        else:
            title_string += ' GlobStretch: OFF'
        return(title_string)
    
    def update(self, image, rescale = True):
        """
            update image in current viewer
        """
        if __VIEWER_DOWNCONVERT_DATA__:
            if np.issubdtype(image.dtype, np.complexfloating):
                self.image = image.astype(np.complex64)
            elif np.issubdtype(image.dtype, np.floating):
                self.image = image.astype(np.float32)
            elif np.issubdtype(image.dtype, np.bool):
                self.image = image.astype(np.uint8)
            else:
                self.image = image
        else:
            self.image = image

        if np.ndim(self.image) == 2:
            self.im_to_draw = self.image
        elif np.ndim(self.image) == 3:
            self.im_to_draw = self.image[:,:,0]
        elif np.ndim(self.image) == 4:
            self.im_to_draw = self.image[:,:,0,0]
        elif np.ndim(self.image) == 5:
            self.im_to_draw = self.image[:,:,0,0,0]
        elif np.ndim(self.image) == 6:
            self.im_to_draw = self.image[:,:,0,0,0,0]
        else:
            print('Wrong dimensions: minium is 2 maximum is 6')
            return()
        if rescale:
             self.rescale(self.im_to_draw.min(), self.im_to_draw.max())
        self.draw()

    def add(self, im, axis =2):
        """
        append another image at given axis
        """
        try:
            if axis <0:
                print('Invalid axis -> appending at 0')
                axis =0
            elif axis >5:
                print('Invalid axis (maximum 6 axis ranging form 0 to 5) -> appending at 5')
                axis =5
            from .image import cat
            self.image = cat((self.image, im.swapaxes(0,1)),axis)
            self.fig.canvas.set_window_title(self.__make_title__())
        except:
            print('THIS FUNCTION IS ONLY AVAILABLE IN COMBINATION WITH THE COMPELETE IMAGE PACK')

    def connect(self):
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.onRelease)
        self.cid_move = self.fig.canvas.mpl_connect('motion_notify_event', self.onMove)
        self.cid_press = self.fig.canvas.mpl_connect('key_press_event', self.onPress)

    def update_gamma(self, val):
        self.gamma_val = np.exp(self.gamma.val)
        self.gamma.valtext.set_text(self.gamma_val)
        self.draw()

    def update_scale_max(self, val):
        if val > self.z_range[0]:
            self.z_range[1] = val
            self.rescale(self.z_range[0], self.z_range[1])

    def update_scale_min(self, val):
        if val < self.z_range[1]:
            self.z_range[0] = val
            self.rescale(self.z_range[0], self.z_range[1])

    def onMove(self, event):
        def format_coord(x, y):
            row = int(x + 0.5)
            col = int(y + 0.5)
            return_string = 'x=%1.3f, y=%1.3f' % (x, y)
            try:
                if np.ndim(self.image) == 2:
                    val = self.image[col, row]
                if np.ndim(self.image) == 3:
                    val = self.image[col,row,self.current_slice]
                    return_string+= ' slice='+str(self.current_slice)
                if np.ndim(self.image) == 4:
                    val = self.image[col,row,self.current_slice, self.current_page]
                    return_string+= ' slice='+str(self.current_slice)+' page='+str(self.current_page)
                if np.ndim(self.image) == 5:
                    val = self.image[col,row,self.current_slice,  self.current_page, self.current_sheet]
                    return_string+= ' slice='+str(self.current_slice)+' page='+str(self.current_page)+' sheet='+str(self.current_sheet)
                if np.ndim(self.image) == 6:
                    val = self.image[col,row,self.current_slice, self.current_page, self.current_sheet, self.current_book]
                    return_string+= ' slice='+str(self.current_slice)+' page='+str(self.current_page)+' sheet='+str(self.current_sheet)+' book='+str(self.current_book)
                if np.issubdtype(self.image.dtype, np.complexfloating):
                    if val.imag >=0:
                        return_string += '    val = '+__VIEWER_FORMAT_VALUE_STRING__+' +'+__VIEWER_FORMAT_VALUE_STRING__+'i  ABS='+__VIEWER_FORMAT_VALUE_STRING__+'  PHI='+__VIEWER_FORMAT_VALUE_STRING__
                    else:
                        return_string += '    val = '+__VIEWER_FORMAT_VALUE_STRING__+' '+__VIEWER_FORMAT_VALUE_STRING__+'i  ABS='+__VIEWER_FORMAT_VALUE_STRING__+'  PHI='+__VIEWER_FORMAT_VALUE_STRING__
                    return_string = return_string % (np.real(val),np.imag(val), np.abs(val), np.angle(val))
                else:
                    return_string += '    val = '+__VIEWER_FORMAT_VALUE_STRING__ % val
            except:
                return_string += '  -------   '
            return(return_string)
        self.ax.format_coord = format_coord

    def onRelease(self, event):
        if (time.time()-self.click_time) < __VIEWER_MAX_CLICK_TIME__:
            if event.button == 1:
                self.marker = (event.xdata, event.ydata)
                if self.mark_clicked and event.xdata != None and event.ydata != None and event.xdata >=0 and event.xdata <= self.im_to_draw.shape[0] and event.ydata >=0 and event.ydata <= self.im_to_draw.shape[1]:
                    self.set_mark(self.marker)
            if (event.button == 3) and (len(self.markerposlist) >0):
                minim = 10000000
                for p in self.markerposlist:
                    if np.sqrt((event.xdata-p[0])**2+(event.ydata-p[1])**2)< minim:
                        minim =np.sqrt((event.xdata-p[0])**2+(event.ydata-p[1])**2)
                        minpos = p

                self.markerposlist.remove(minpos)
                lim=self.imgplot.get_clim()
                self.ax.cla()

                self.imgplot = self.ax.imshow(self.im_to_draw, cmap = 'gray',interpolation = 'none')
                self.imgplot.set_clim(lim)
            self.draw()

    def onClick(self, event): 
        self.click_time = time.time()

    def __disp_complex__(self, mode):
        if np.issubdtype(self.im_to_draw.dtype, np.complexfloating):
            if mode == 0:
                return(self.__set_scale__(np.abs(self.im_to_draw)))
            if mode == 1:
                return(self.__set_scale__(np.angle(self.im_to_draw)))
            if mode == 2:
                return(self.__set_scale__(np.real(self.im_to_draw)))
            if mode == 3:
                return(self.__set_scale__(np.imag(self.im_to_draw)))
        else:
            return(self.__set_scale__(self.im_to_draw))

    def onPress(self, event):
        img_changed = False
        if event.key  == 'h':
            try:
                from tkinter import Tk, messagebox
                Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
                messagebox.showinfo('Navigation reference for image viewer',self.__init__.__doc__)
            except:
                print(self.__init__.__doc__)
                
        if event.key == 'd':
            self.scale16bit = not self.scale16bit
        if event.key == 'a':
            if self.s == 'linear':
                self.s = 'log'
                self.update_sliders()
            elif self.s == 'log':
                self.s = 'linear'
                self.update_sliders()
        if event.key == 'enter':
            self.return_pressed = True
        if event.key == 'x':
            self.global_stretch = not(self.global_stretch)
        if event.key == 'y':
            self.draw_hist(self.__disp_complex__(self.complex_set))
            
        if event.key == 'e':
            self.export = np.clip(self.__disp_complex__(self.complex_set), self.z_min.val, self.z_max.val)

        if event.key == 'r':
                self.rescale(np.min(self.__disp_complex__(self.complex_set)), np.max(self.__disp_complex__(self.complex_set)))
                self.update_sliders()
                self.gamma.set_val(0.0)
                self.gamma.valtext.set_text(1.0)
                self.gamma_val = 1.0
        if event.key == 'c':
            if np.issubdtype(self.im_to_draw.dtype, np.complexfloating):
                if self.complex_set == 3:
                    self.complex_set = 0
                else:
                    self.complex_set += 1
                #self.draw();
                self.rescale(np.min(self.__disp_complex__(self.complex_set)), np.max(self.__disp_complex__(self.complex_set)))
                self.update_sliders()
            else:
                self.complex_set = 0
        if np.ndim(self.image) >= 3:
            if event.key == 'right' and self.current_slice != (self.image.shape[2]-1):
                self.current_slice += 1
                img_changed = True

            if event.key == 'left' and self.current_slice != 0:
                self.current_slice -= 1
                img_changed = True

            if event.key == 'pageup':
                if (self.current_slice+10) < self.image.shape[2]:
                    self.current_slice +=10
                    img_changed = True
                else:
                    self.current_slice = self.image.shape[2]-1
                    img_changed = True
            if event.key == 'pagedown':
                if (self.current_slice-10) >=0:
                    self.current_slice -=10
                    img_changed = True
                else:
                    self.current_slice = 0
                    img_changed = True
            if np.ndim(self.image) >= 4:
                if event.key == 'up' and self.current_page != (self.image.shape[3]-1):
                    self.current_page += 1
                    img_changed = True
                if event.key == 'down' and self.current_page != 0:
                    self.current_page -= 1
                    img_changed = True
                if np.ndim(self.image) >= 5:
                    if event.key == 'b' and self.current_sheet != (self.image.shape[4]-1):
                        self.current_sheet += 1
                        img_changed = True
                    if event.key == 'v' and self.current_sheet != 0:
                        self.current_sheet-= 1
                        img_changed = True
                    if np.ndim(self.image) >= 6:
                        if event.key == 'm' and self.current_book != (self.image.shape[5]-1):
                            self.current_book += 1
                            img_changed = True
                        if event.key == 'n' and self.current_book != 0:
                            self.current_book -= 1
                            img_changed = True

        if img_changed:
            img_changed = False
            if np.ndim(self.image) == 3:
                self.im_to_draw = self.image[:,:,self.current_slice]
            if np.ndim(self.image) == 4:
                self.im_to_draw = self.image[:,:,self.current_slice, self.current_page]
            if np.ndim(self.image) == 5:
                self.im_to_draw = self.image[:,:,self.current_slice,  self.current_page, self.current_sheet]
            if np.ndim(self.image) == 6:
                self.im_to_draw = self.image[:,:,self.current_slice, self.current_page, self.current_sheet, self.current_book]

            if self.global_stretch:
                if self.s == 'log':
                    zr = [np.log(1+np.abs(self.glob_lims[self.complex_set][0].astype(np.float64))),np.log(1+np.abs(self.glob_lims[self.complex_set][1].astype(np.float64)))]
                else:
                    zr = [self.glob_lims[self.complex_set][0],self.glob_lims[self.complex_set][1]]
                self.rescale(zr[0], zr[1])
                self.update_sliders(zr)
            else:
                self.rescale(np.min(self.__disp_complex__(self.complex_set)), np.max(self.__disp_complex__(self.complex_set)))
        self.draw()

    def rescale(self, z_min, z_max):
        self.imgplot.set_clim((z_min, z_max))
        self.fig.canvas.draw()

    def __set_scale__(self, im):
        
        if self.s == 'log':
            try:                    
                from .util import scale_log
                im = scale_log(im)
            except:    # for standalone purposes
                im = np.log(1+np.abs(im.astype(np.float64)))
        if self.scale16bit == True:
            try:
                from .util import scale_16bit
                im = scale_16bit(im)
            except:    # for standalone purposes
                 im = (im/np.max(im)*65536)
                 im = im.astype(int)
        return(im)

    def draw(self, image=None):
        """
            Draw the image in the figure
        """
        self.fig.canvas.set_window_title(self.__make_title__())
        if image is not None:
            self.im_to_draw = image

        #self.ax.set_xlim([0,self.image.shape[1]]);
        #self.ax.set_ylim([self.image.shape[0],0]);
        #self.imgplot = self.ax.imshow(self.im_to_draw, cmap = 'gray',interpolation = 'none');
        img = self.__disp_complex__(self.complex_set)

        def __g__(y, gamma):
            return(np.sign(y)*np.abs(y)**gamma*np.abs(np.max(y))**(1-gamma))
        #print(img.shape)
        self.imgplot.set_data(__g__(img, self.gamma_val))

        if len(self.markerposlist) >0:
            pos = np.asarray(self.markerposlist)
            self.ax.plot(pos[:,0],pos[:,1], '.r')
        self.fig.canvas.draw()
        #plt.draw();                         
              
        



    def draw_hist(self, im = None):
        """
        Draw histogramm of the image im. If im is not given, the objects own image is drawn
        """
        self.hist_plot = plt.figure()
        self.hist_plot.canvas.set_window_title('Histogram')
        hax = self.hist_plot.add_subplot(111)
        
        if type(im) != np.ndarray:
            hax.hist(self.image.ravel(),100)
        else:
            hax.hist(im.ravel(),100)

    def set_mark(self,pos):
        """
        draw mark in image
        pos must be tupel of (x,y) where the marker should be drawn
        """
        if pos not in self.markerposlist:
            self.markerposlist.append(pos)
        self.draw()

    def clear_mark(self,number = -1):
        if number == -1:
            self.markerposlist = []
        else:
            if number > len(self.markerposlist):
                print('Index too large. Marker not in list')
            else:
                del self.markerposlist[number]
        self.draw()

    def close(self):
        self.fig.canvas.mpl_disconnect(self.cid_click)
        self.fig.canvas.mpl_disconnect(self.cid_press)
        plt.close(self.fig.number)
        if self.hist_plot != None:
            plt.close(self.hist_plot.number)
    #def __del__(self):
    #    self.close();
#        
        
def close():
    import matplotlib.pyplot as plt
    plt.close('all')
    import NanoImagingPack.view5d as view5d
    view5d.v5close()
  
def graph(y,x=None, title = '', x_label = '', y_label = '', legend = [], markers =[], linestyles = [], linewidths =[], save_path = None, text= None):
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        from .image import image as imtype
        standalone = False
    except:
        standalone = True

    if type(y) != list: 
        y = [y]

    if type(x) != list: 
        if type(x) == np.ndarray or (standalone == False and type(x) == imtype):
            xl = []
            for el in y:
                if np.size(el) != np.size(y[0]):
                    print('Y vectors have varying length! If you only give one x array, all y components musst have same lengths!')
                    return
                else:
                    xl.append(x)
            x=xl
        else:
            if x != None:
                print('Wrong x type')
                return
    for el in y:
        if isinstance(el, np.ndarray) == False:
        #if type(el)!= np.ndarray and (standalone == False and type(el) != imtype):
            print('Y list contains non array type!')
            return
    if (type(x) != list) and (x != None): 
        print('X must be a list of same length than X or empty, if Y is a list')
        return
    else:
        plot_list = []
        label_list = []
        markers_list= []
        line_style_list =[]
        line_width_list = []
        for i in range(len(y)):
            try:
                label_list += [legend[i]]
            except IndexError:
                label_list += ['']
            try:
                markers_list += [markers[i]]
            except IndexError:
                markers_list += ['']
            try:
                line_style_list += [linestyles[i]]
            except IndexError:
                line_style_list += ['-']
            try:
                line_width_list += [linewidths[i]]
            except IndexError:
                 line_width_list += [1]
        if x is None:
            fig = plt.figure()
            if title != '':                
                fig.canvas.set_window_title(title)
            for ely, label, mark, lstyle, lwidth in zip(y, label_list, markers_list, line_style_list, line_width_list):
                elx = np.arange(0,np.size(ely),1)
                plot, = plt.plot(elx,ely, marker = mark, label = label, linestyle = lstyle, linewidth = lwidth)
                plt.xlabel(x_label)
                plot_list.append(plot)
            plt.ylabel(y_label)
            plt.title(title)
            if len(legend) != 0:
                plt.legend(handles = plot_list)
            if text is not None:
                text_pos = ((plt.xlim()[0] + plt.xlim()[1]) / 2, (plt.ylim()[0] + plt.ylim()[1]) / 2)
                plt.text(text_pos[0], text_pos[0], text)
            plt.show()
        else:
            if len(x) != len(y):
                print('X must be a list of same length than X or empty, if Y is a list')
                return
            else:
                for el in x:
                    if type(el)!= np.ndarray and (standalone == False and type(el) != imtype):
                        print('X list contains non array type!')
                        return
                #plt.figure();
                fig = plt.figure()
                if title != '':                
                    fig.canvas.set_window_title(title)
                for elx,ely, label, mark, lstyle, lwidth in zip(x,y,label_list, markers_list, line_style_list, line_width_list):
                    plot, =plt.plot(elx,ely, marker = mark, label = label, linestyle = lstyle, linewidth = lwidth);
                    plot_list.append(plot);
                    
                plt.xlabel(x_label);
                plt.ylabel(y_label);
                plt.title(title);
                if len(legend) != 0:
                    plt.legend(handles = plot_list);
                if text is not None:
                    text_pos = ((plt.xlim()[0]+plt.xlim()[1])/2,(plt.ylim()[0]+plt.ylim()[1])/2)
                    plt.text(text_pos[0],text_pos[0],text)    
                    #plt.table(cellText = text)
                plt.show();
        if save_path is not None:
            fig.savefig(save_path);
    return(fig);
