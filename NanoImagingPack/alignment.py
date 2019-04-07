# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 19:56:06 2017

@author: ckarras
"""

import numpy as np
from .fitting import fit_gauss2D

from .util import get_type, max_coord

from .mask import create_circle_mask
from .image import imsave, readim, match_size, correl, extract, centroid
from .view import view


class alignment():
    """
           Alignment class:

               initiate with the 2 images that shall be aligned. Eventually image 2 will be aligned with respect to image 1
                       -> representative points will be found and correlation around those will be opitmized!


               im1, im2 can be either the path of the image file or the ndarray containing the image

               para_file: path of a file containing stored alignment paramters -> use if old parameters shall be used

               new_alignement ('yes' or 'no', default == 'no'):
                       if 'yes' -> try to find new alignment coordinates -> give 2 calibration images -> find support points via a peak finding algorithm -> create a new set of alignment parameters
                       if 'no'  -> load the old set of parameters


              roi: in which range around a coordinate do you want to investigate the cross correlation

              method: how do you want to find the optimum shift? 'maximum' or 'centroid' or 'fit_gauss'
                          -> if 'centroid' is used -> iterative process for optimizing individual positions -> set 'max_err' for maximum integrated distance for all points

              store_correl: store cross correlations in list;
              align_save_path: if not None (default) than the aligned image 2 will be stored under the given path. This must be a folder!!!
              super_sampling : super sampling factor  -> if you e.g. want to align SIM reconstructed images using a alignent parameter set from a widefield image that has to be 2


              roi_lists: List of ROIs for the different images: [glob_roi_alignm, glob_roi_im, loc_roi_alignm_im1, loc_roi_img_im1, loc_roi_alignm_im2, loc_roi_img_im2]

                  - This is important if the images that shall be aligned were recorded in different ROIs as the image which gives the support points for the alignment
                  - Global and local ROIs may be found in the parameter files -> Global ROI is "Boarders ROI Global:" as stated, local Roi is "Boarders ROI CH..." in the DCIMG to Tiff converter !!!

    """

    def __init__(self, im1, im2,  para_file= None, new_alignment = 'no', roi = (50,50), method = 'centroid', store_correl = False, align_save_path=None, Name_list = None, super_sampling = 1, roi_list = None, remove_bckgrd = False, max_err = 0.5):
        from skimage.feature import corner_harris, corner_peaks
        from skimage.measure import ransac
        from skimage.transform import warp, AffineTransform

        def __chk_load__(M):
            if get_type(M)[0] == 'array':
                return(M)
            elif type(M) == str:
                return(readim(M))
        self.img1 = __chk_load__(im1)
        self.img2 = __chk_load__(im2)

        if remove_bckgrd == True:
            self.img1 = self.img1 - np.min(self.img1)
            self.img2 = self.img2 - np.min(self.img2)

        self.img1, self.img2 = match_size(self.img1,self.img2,0,'const_below', False)
        self.img1, self.img2 = match_size(self.img1,self.img2,1,'const_below', False)
        self.img2_align  =None
        self.para_file =para_file  # were to store alignement information
        self.method = method  # which method to use!?!?
        self.correls = []  # correlations of the coordinate
        self.shift_coords = []  # shift of the postitions!
        self.correl_max  =[]  # maximum POSITION of the correlations
        self.width = 10  # when clipping around the maximum of the correlation -> what is the width of the window???
        self.max_dist = max_err  #threshold for stopping the iteration
        self.max_iter = 300  # maximum iterations
        self.max_rad = 5  # maximum radius of cricular mask for centroid computation
        self.super_sample = super_sampling
        self.msd = 100000
        self.mat = np.eye(3)
        if (new_alignment == 'yes'):              # here: find new parameter set using the support points
            self.new_coords = []
            self.shift_coords = []
            self.correl_max =[]
            self.correls=[]
            img_align = self.img2.astype(float)
            self.img1 = self.img1.astype(float)
            
            print('Finding coordinate supports via corner detection in image 1...')
            # Find appropriates support points for alignment using corner detection algorithm
            self.coords = corner_peaks(corner_harris(self.img1), threshold_rel=0.001, min_distance=roi[0])
            coords = self.coords
            print(str(coords.shape[0]) + ' Cooridnate pairs were found')
            print('Start optimization using method: '+self.method)

            self.shifts = self.coords*0.0

            for i in range(self.max_iter):
                
                self.new_coords = []
                self.shift_coords = []
                self.correl_max =[]
                self.correls=[]

                print('#########################################')
                print('Iteration ' + str(i))
                print('')
                print('Finding Shifts...')  
                
                for center in coords:
                    s = self.get_shift(self.img1, img_align, list(center), list(roi), store_correl)
                    self.new_coords.append([center[0]-s[0],center[1]-s[1]])
                    self.shift_coords.append(s)

                #print(self.shift_coords[0]);
                # compute mean square difference of shiftings as measure for shifting size:
                self.msd = np.mean(np.asarray([np.sqrt(x[0]**2+x[1]**2) for x in self.shift_coords]))
                print('Mean square distance of shiftings:' + str(self.msd))
                print('')
                print('Computing matrix...')
                # Compute transformation matrix
                model, inliers = ransac((np.fliplr(self.coords), np.fliplr(self.new_coords)), AffineTransform, min_samples=3, residual_threshold=2, max_trials=100)
                outliers = inliers == False
                #Propagate transformation matrix
                self.mat = np.dot(model.params, self.mat)
                # check determinante of transformation matrix to prevent singularities!
                d = np.linalg.det(model.params)
                if np.abs(d)< 0.1:
                    print('Transformation matrix becomes singular! Bailing out')
                    return(-1)
                print('SCALE | TRANSLATION | ROTATION | SHEAR')
                print(model.scale, model.translation, model.rotation, model.shear)
                # create aligned image 
                # update image2 for next alignment step                           
                img_align = warp(img_align, model.inverse)
                if self.msd < self.max_dist:
                    print('mean_square distance small enough ... stopping')
                    break
                if i == self.max_iter-1:
                    print('Maximum of iterations reached')
            if para_file != None:
                s = ''
                for i in range(3):
                    for j in range(3):
                        s += str(self.mat[i,j])+'\n'
                if para_file[-4:] != '.txt':
                    para_file = para_file+'.txt'
                f = open(para_file,'w')
                f.write(s)
                f.close()

        else:                        # Here: load parameter set and align it 
            pre_mat = np.eye(3)
            if roi_list != None:
                # In case of different rois between the alignment image and the real image: shift the images using a affine transformation first
                # also take care in case the alignment image was normally recorded and the real images are supersampled
                glob_alginment = roi_list[0]
                glob_im = roi_list[1]
                loc_align_im1 = roi_list[2]
                loc_im_im1 = roi_list[3]
                loc_align_im2 = roi_list[4]
                loc_im_im2 = roi_list[5]
                delta_y = +loc_im_im1[1]-loc_im_im2[1]-loc_align_im1[1]+loc_align_im2[1]
                delta_x = +loc_im_im1[0]-loc_im_im2[0]-loc_align_im1[0]+loc_align_im2[0]
                pre_mat[0,2] = -delta_y*self.super_sample
                pre_mat[1,2] = -delta_x*self.super_sample
            model = AffineTransform(pre_mat)
            img_align = warp(self.img2.astype(float), model.inverse)

            self.load_para()

            model = AffineTransform(self.mat)
            print('SCALE | TRANSLATION | ROTATION | SHEAR')
            print(model.scale, model.translation, model.rotation, model.shear)
            img_align = warp(img_align, model.inverse)

        self.img2_align = img_align.astype(int)

        if align_save_path != None:
            if align_save_path[-1] != '/':
                align_save_path += '/'
            if Name_list == None:
                imsave(self.img1, align_save_path+'img1_align.tif')

                imsave(self.img2_align, align_save_path+'img2_align.tif')
            else:
                imsave(self.img1, align_save_path+Name_list[0])
                imsave(self.img2_align, align_save_path+Name_list[1])

    def load_para(self, parafile = None):
            if parafile == None: 
                parafile = self.para_file
            if parafile !=None:
                with open(parafile) as f:
                    content = f.readlines()
                content = [x.strip() for x in content]
            for i in range(9):
                self.mat[i//3, np.mod(i,3)] = content[i]
            # Take care of supersampling (larger shift!!!!)
            self.mat[0,2] = self.mat[0,2]*self.super_sample
            self.mat[1,2] = self.mat[1,2]*self.super_sample

    ### DEPRICATED CODE ###        
#    def load_para(self, parafile = None):
#        if parafile == None: 
#            parafile = self.para_file;
#        if parafile !=None:
#            with open(parafile) as f:
#                content = f.readlines();
#            content = [x.strip() for x in content]
#        self.coords = [];
#        self.new_coords =[];
#        for i in range(len(content)//4):
#            self.coords.append((float(content[i*4])*self.super_sample, float(content[i*4+1])*self.super_sample));
#            self.new_coords.append((float(content[i*4+2])*self.super_sample, float(content[i*4+3])*self.super_sample));
#    
    def transform_coord(self,coord, new_coord, glob_roi_alignm, glob_roi_im, loc_roi_alignm_im1, loc_roi_img_im1, loc_roi_alignm_im2, loc_roi_img_im2):
        """
        If the ROIs of the alignment support point image and the recorded image are different this transformation is necessary!

        One has to put in the global and the local ROI information which is recordet in the parameter file


        coord: tupel of coordinates of the support points
        alignment_img:             The image which provides the alignment support points
        img:                       The image at which the alignment has to be applied

        Note: the local ROI information is from the dcimg converter part, the global ROI from the experimental part!


        glob_roi_x = [x0, y0, x_length, y_length] -> so it is given in the Parameter files
        loc_roi_x = [x0,y0,x1,y1] -> (x0,y0) and (x1,y1) are the edges of the ROI
        """
        n_coo =[]
        coo = []
        print(coord)
        print(new_coord)
        for i in range(2):
            delta_glob_roi = glob_roi_alignm[i]-glob_roi_im[i]
            coo.append(coord[i]-(loc_roi_img_im2[i]-delta_glob_roi-loc_roi_alignm_im2[i]))  # new start point
            n_coo.append(new_coord[i]-(loc_roi_img_im1[i]-loc_roi_alignm_im1[i]-delta_glob_roi))
        print(coo)
        print(n_coo)
        return(tuple(n_coo), tuple(coo))

        #### DEPRICATED CODE #####
#    def save_paras(self, parafile = None):
#        '''
#        Saves parameter files in list -> block of 4 items: x_old, y_old, x_new,y_new
#        '''
#        if parafile == None: 
#            parafile = self.para_file;
#        
#        if parafile !=None:
#            s  = '';
#            for coo, new_coo in zip(self.coords, self.new_coords):
#                s += str(coo[0])+'\n'+str(coo[1])+'\n'+str(new_coo[0])+'\n'+str(new_coo[1])+'\n';
#            if parafile[-4:] != '.txt':
#                parafile = parafile+'.txt';
#            f = open(parafile,'w');
#            f.write(s);
#            f.close();

    def get_shift(self,im1, im2, center = [0,0], edge_length = [100,100],store_correl=False):
        """
           #get the coordinate shift of two images (2D) within a certain region
        """
        if (np.mod(edge_length[0],2) == 1): edge_length[0] = edge_length[0] + 1;
        if (np.mod(edge_length[1],2) == 1): edge_length[1] = edge_length[1] + 1;
        if self.method == 'maximum':
#            cc = correl(extract_c(im1, center = tuple(center), roi = tuple(edge_length)),extract_c(im2, center = tuple(center), roi = tuple(edge_length)))
            cc = correl(extract(im1, ROIsize=tuple(edge_length), centerpos=tuple(center)),
                        extract(im2, ROIsize=tuple(edge_length), centerpos=tuple(center)),matchsizes=True)
            if store_correl:
                self.correls.append(cc)
            self.correl_max.append((max_coord(cc)[0],max_coord(cc)[1]))
            #return((max_coord(cc[0])[0]-edge_length[0]//2,max_coord(cc[0])[1]-edge_length[1]//2))
            
            #shift of image 2 with respect to image 1 at the selected coordinate and around edge_length
            return((edge_length[0]//2-max_coord(cc)[0],edge_length[1]//2-max_coord(cc)[1]))   # new code 30.11.17
        elif self.method == 'centroid':
            
            mr_old = self.max_rad

            def __get_cc__(edge_length, center, matchSizes=False):
                if (np.mod(edge_length[0],2) == 1): edge_length[0] = edge_length[0] + 1;
                if (np.mod(edge_length[1],2) == 1): edge_length[1] = edge_length[1] + 1;
#                cc = correl(extract_c(im1, center = tuple(center), roi = tuple(edge_length)),extract_c(im2, center = tuple(center), roi = tuple(edge_length)));
                cc = correl(extract(im1, ROIsize = tuple(edge_length), centerpos = tuple(center)), extract(im2, ROIsize = tuple(edge_length), centerpos = tuple(center),matchSizes=matchSizes))
                #mc = max_coord(cc[0]);
                mc = max_coord(cc)
                self.max_rad = min([self.max_rad, abs(edge_length[0]-mc[0]), mc[0], mc[1], abs(edge_length[1]-mc[1]) ])
                return(mc, cc, edge_length)

            mc, cc, edge_length = __get_cc__(edge_length, center)
            if self.max_rad == 0:
                print('Maximum of cross correlation at boarder -> Increasing cc_edge_length by 10')
                self.max_rad = mr_old
                edge_length[0] += 10
                edge_length[1] += 10
                mc, cc, self.max_rad, edge_length = __get_cc__(edge_length, self.max_rad, center, matchSizes=True) # Rh for compatibility
            if store_correl:
                self.correls.append(cc)
            centr_coord = centroid((cc-np.min(cc))*create_circle_mask(mysize =edge_length,maskpos = mc ,radius=self.max_rad, zero = 'image'))
            return((edge_length[0]//2-centr_coord[0],edge_length[1]//2-centr_coord[1]))
    
        elif self.method == 'fit_gauss':
            cc = correl(extract(im1, ROIsize=tuple(edge_length), centerpos=tuple(center)),
                        extract(im2, ROIsize=tuple(edge_length), centerpos=tuple(center)), matchsizes=True)
#            cc = correl(extract_c(im1, center = tuple(center), roi = tuple(edge_length)),extract_c(im2, center = tuple(center), roi = tuple(edge_length)))
            if store_correl:
                self.correls.append(cc)
            clip = extract(cc, ROIsize=(self.width,self.width), centerpos=max_coord(cc[0]))
#            clip = extract_c(cc, max_coord(cc[0]), roi = (self.width,self.width))  # clip more or less symetrical around maximum of correlation
            f= fit_gauss2D(clip, False)  # fit gaussian;
            pos = (f[0][1]+max_coord(cc)[0]-self.width//2, f[0][2]+max_coord(cc)[1]-self.width//2)  # position of new maximum in non-clip correlation
            self.correl_max.append(pos)
            return((pos[0]-edge_length[0]//2,pos[1]-edge_length[1]//2))
    
    def show_correl(self,number = 0):
        """
            depict given correlation and mark center point (maximum or centroid)
        """
        if len(self.correls)>0 and len(self.correl_max)>0:
            # try:
            #     del(v)
            # except:
            #     pass;
            v = view(self.correls[number], title = 'Correlation number '+str(number)+' Coordiates around '+str(round(self.coords[number][0]))+' ; '+str(round(self.coords[number][1])))
            v.set_mark(self.correl_max[number])
        else:
            print('No correlations availible')
