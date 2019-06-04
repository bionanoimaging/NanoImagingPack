# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 14:38:16 2018
@author: Rainer Heintzmann

implements a class approximating a kernel as separable multiplication of 1D kernels

"""
import numpy as np
from .image import image
from .util import ones, repToList, castdim
from .image import extract
import scipy

class separable():
    """
    this class represents a (convolution) kernel as a sum of separable 1D kernels to be applied consecutively. This allows to speed up the convolution operation.
    It also allows to do spatially variate convolutions, by allowing to alter the weights over the position. See separable.convolve for details.
    To represent a range of images with various changing properties, simply enter examples (e.g. the extreme cases) along another dimension (e.g. -3).
    """
    def __init__(self, animg, maxorders, kernellength=15, verbose=True):
        # if ndims is None:
        ndims = animg.ndim
        if kernellength is not None:
            kernellength = repToList(kernellength, ndims, 1)
            kernellength = [min(kernellength[d], animg.shape[d]) for d in range(len(kernellength))]
            animg = extract(animg, kernellength)
        animg = np.squeeze(animg)

        if ndims == 2 and animg.ndim == 2:  # here we can use the "standard" SVD representation
            U, D, V = np.linalg.svd(animg)
            self.kernels = [[], []]  # list of list of 1D-kernels
            for order in range(maxorders):
                self.kernels[1].append(D[order] * V[order:order+1, :])
                self.kernels[0].append(U[:, order:order+1])
        else:
#            loss = lambda mykernelVec: np.linalg.norm(joinFromVec(mykernelVec, kernellength, maxorders) - animg)
            loss = lambda mykernelVec: GaussianLoss(joinFromVec(mykernelVec, kernellength, maxorders), animg)
            x0 = ones(np.sum(kernellength)*maxorders) / maxorders
            res = scipy.optimize.minimize(loss, x0, method='BFGS', options={'gtol': 1e-6, 'disp': True, 'maxiter': 2000})
            self.kernels = packVec(res.x, kernellength, maxorders)
        if verbose:
            print("Final Loss is:" + str(GaussianLoss(self.join(), animg)))

    def join(self):
        """
        uses the default weigths (of 1.0) to join the self.kernels back together.
        :return: the joined result
        """
        return join(self.kernels)

    def convolve(self, obj, coefficientMap = None):
        """
        convolves the object (obj) by sequential real-space convolution with 1D kernels, as preprocessed by this class.
        :param obj: object to convolve with the kernels
        :param coefficientMap: optional map of local coefficients to be applied prior to convolution to the object. One represents the standard convolution
        :return:
        """
        ndims = len(self.kernels)
        maxorders = len(self.kernels[0])
        res = 0.0
        for order in range(maxorders):
            if coefficientMap is None:
                myconv = obj
            else:
                myconv = obj * coefficientMap[order]
            for d in range(ndims):
                myconv = scipy.ndimage.convolve(myconv, self.kernels[d][order])
            res = res + myconv
        return image(res)

def packVec(avector, shapes, maxorders):
    ndims = len(shapes)
    start = 0
    x0=[]
    for d in range(ndims):
        xd=[]
        for order in range(maxorders):
            xd.append(castdim(avector[start:start+shapes[-d-1]], ndims, -d-1))
            start += shapes[-d-1]
        x0.append(xd)
    return x0

def joinFromVec(kernelVec, shapes, maxorders):
    return join(packVec(kernelVec, shapes, maxorders))

def join(kernels):
    """
    uses the default weigths (of 1.0) to join the kernels back together.
    :param kernels: list of list of kernels
    :return: the joined result
    """
    ndims = len(kernels)
    maxorders = len(kernels[0])
    img = 0.0
    for order in range(maxorders):
        myprod = kernels[-1][order]
        for d in range(1,ndims):
            myprod = myprod * kernels[-d-1][order]
        img = img + myprod
    return image(img)


printIter=0
def GaussianLoss(kernelImg, anImg, verbose=False):
    loss = np.linalg.norm(kernelImg - anImg)
    if verbose:
        if np.mod(printIter,10) == 0:
            print(loss)
        printIter += 1
    return loss
