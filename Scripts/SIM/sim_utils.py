from NanoImagingPack import readim, extract;
import numpy as np;

def LoadData(ImageParam):
    """
    Load data set for SIM reconstruction
    :param ImageParam: Including path, pixelsize, roi_sie and roi_center
    :return: data set

    ToDo:   Sorting of images -> [x,y,,z, phase, dir, channels, z] ??? -> double check with upcoming helper funcitons
    """
    data = readim(ImageParam.path);
    data.set_pixelsize(ImageParam.pixelsize);
    data = extract(data, ImageParam.roi_size, ImageParam.roi_center);
    if data.ndim < 4:
        data=data[:,np.newaxis,:,:]
    return(data);

def SeperationTest(data, SimParam):
    """
    Performing 3d ft of 2D Sim images and Phase steps in 3rd dimension

    :param data:  Input data
    :param SimParam:    SimParameter Structure -> requires phase steps
    :return: images of separated orders
    """

    n_dir = data.shape[0]//SimParam.Steps;
    print(str(n_dir)+ ' different directions detected');
    new_shape = tuple([n_dir, SimParam.Steps]+list(data.shape[1:]))
    data = np.reshape(data, new_shape)


    return(data.ft(axes=(-1,-2,1)))

