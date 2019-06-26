# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 14:38:16 2018
@author: Rainer Heintzmann

implements a class approximating a kernel as separable multiplication of 1D kernels

"""
import numpy as np
from .image import image
from .util import ones, repToList, castdim, expanddim
from .image import extract
import scipy
from sktensor import dtensor, cp_als  # from https://github.com/evertrol/scikit-tensor-py3/blob/master/examples/cp_sensory_bread_data.py

class separable():
    """
    this class represents a (convolution) kernel as a sum of separable 1D kernels to be applied consecutively. This allows to speed up the convolution operation.
    It also allows to do spatially variate convolutions, by allowing to alter the weights over the position. See separable.convolve for details.
    To represent a range of images with various changing properties, simply enter examples (e.g. the extreme cases) along another dimension (e.g. -3).
    """
    def __init__(self, animg, maxorders, kernellength=15, verbose=False):
        self.ndims = animg.ndim
        if kernellength is not None:
            kernellength = repToList(kernellength, self.ndims, 1)
            kernellength = [min(kernellength[d], animg.shape[d]) for d in range(len(kernellength))]
            animg = extract(animg, kernellength)
#        if animg is None:
        animg = np.squeeze(animg)

        # self.dimStart = [0]
        # totallength = kernellength[-1]
        # for d in range(1, self.ndims):
        #     self.dimStart.append(totallength)
        #     totallength = totallength + kernellength[-d-1]
        # self.dimStart.append(totallength)

        # self.kernels = np.zeros((maxorders, totallength)) # the kernels have maxorders x alldims   as shape. alldims stack the 1d shapes
        if False: # self.ndims == 2 and animg.ndim == 2:  # here we can use the "standard" SVD representation
            U, D, V = np.linalg.svd(animg)
            # for order in range(maxorders):
            #     self.kernels[order, self.dimStart[0]:self.dimStart[1]] = np.squeeze(U[:, order:order+1])
            #     self.kernels[order, self.dimStart[1]:] = np.squeeze(D[order] * V[order:order+1, :])
            self.kernels = [image(U[:maxorders,:]),image(D[:maxorders,np.newaxis]*V[:maxorders,:])]
        else:
#            loss = lambda mykernelVec: GaussianLoss(joinFromVec(mykernelVec, kernellength, maxorders), animg)
#             if False :
#                 loss = lambda mykernel: GaussianLoss(join(mykernel, self.kernels.shape, self.dimStart), animg)
#                 x0 = ones(self.kernels.shape) / maxorders
#                 res = scipy.optimize.minimize(loss, x0, method='BFGS', options={'gtol': 1e-6, 'disp': True, 'maxiter': 2000})
#                 self.kernels = image(np.reshape(res.x, self.kernels.shape)) # packVec(res.x, kernellength, maxorders)
#                 if verbose:
#                     print("Final Loss is:" + str(GaussianLoss(self.join(), animg)))
#             else:
            T = dtensor(animg)
            if verbose:
                import logging
                logging.basicConfig(level=logging.DEBUG)
            P, fit, itr, exectimes = cp_als(T, maxorders, init='random') # alternating least square tensor factorization based on the toolbox
            # syn = nip.image(P.toarray())
            # nip.catE(q, syn)
            self.kernels = [image(anim) for anim in P.U]

    def join(self):
        """
        uses the default weigths (of 1.0) to join the self.kernels back together.
        :return: the joined result
        """
        # if kernelshape is not None:
        #     kernels = np.reshape(kernels, kernelshape)
        ndims = len(self.kernels)
        maxorders = self.kernels[0].shape[1]
        img = 0.0
        for order in range(maxorders):
            myprod = self.getKernel(order, 0)
            for d in range(1, ndims):
                myprod = myprod * self.getKernel(order, d)
            img = img + myprod
        return image(img)

    def getKernel(self, order, d):
        if d >= 0:
            d = - self.ndims + d
        idx = -1-d
#        return castdim(self.kernels[order, self.dimStart[idx]:self.dimStart[idx + 1]], self.ndims, d)
        return castdim(self.kernels[d][:,order], self.ndims, d)

    def subSlice(self):
        mycopy = self.copy()
        mycopy.ndims = mycopy.ndims-1
        return mycopy

    def convolve(self, obj, coefficientMap = None):
        """
        convolves the object (obj) by sequential real-space convolution with 1D kernels, as preprocessed by this class.
        :param obj: object to convolve with the kernels
        :param coefficientMap: optional map of local coefficients to be applied prior to convolution to the object. One represents the standard convolution
        :return:
        """
        maxorders = self.kernels[0].shape[1]
        obj = expanddim(obj, self.ndims)
        ndims = self.ndims
        if coefficientMap is not None:
            ndims -= 1
        for order in range(maxorders):
            if coefficientMap is None:
                myconv = obj
            else:
                lastKernel = self.getKernel(order, 0)  # get the values of the outermost kernel
                myconv = obj * coefficientMap * lastKernel
            for d in range(ndims):
                aKernel = self.getKernel(order, -d-1)
                myconv = scipy.ndimage.convolve(myconv, aKernel)
            if coefficientMap is not None:
                myconv = np.sum(myconv, 0)
            if order == 0:
                res = myconv
            else:
                res += myconv
        return image(res)

# def packVec(avector, shapes, maxorders):
#     ndims = len(shapes)
#     start = 0
#     x0=[]
#     for d in range(ndims):
#         xd=[]
#         for order in range(maxorders):
#             xd.append(castdim(avector[start:start+shapes[-d-1]], ndims, -d-1))
#             start += shapes[-d-1]
#         x0.append(xd)
#     return x0

# def joinFromVec(kernelVec, shapes, maxorders):
#     return join(packVec(kernelVec, shapes, maxorders))

def join(kernel):
    """
    uses the default weigths (of 1.0) to join the kernels back together.
    :param kernels: list of list of kernels
    :return: the joined result
    """
    return kernel.join()

printIter=0
def GaussianLoss(kernelImg, anImg, verbose=False):
    loss = np.linalg.norm(kernelImg - anImg)
    if verbose:
        if np.mod(printIter,10) == 0:
            print(loss)
        printIter += 1
    return loss
