import NanoImagingPack as nip
import numpy as np
import scipy

obj = nip.readim(pixelsize=[100,100]) + 0.0
psfParam = nip.PSF_PARAMS()
psfParam.aberration_types=[psfParam.aberration_zernikes.astigm, psfParam.aberration_zernikes.defoc]    # define list of aberrations (select from choice, zernike coeffs, or aberration map
psfParam.aberration_strength =[0.5,0.5];
h_a = nip.psf(obj, psfParam)
psfParam.aberration_types=[psfParam.aberration_zernikes.astigm, psfParam.aberration_zernikes.defoc]    # define list of aberrations (select from choice, zernike coeffs, or aberration map
psfParam.aberration_strength =[-0.5,0.5];
h_b = nip.psf(obj, psfParam)

if False:
    h = np.squeeze(h_a)
#    h = h_a
else:
    h = nip.cat((h_a,h_b),-3)

k = nip.separable(h,4,25)  # transforms h into a seperable object for fast convolutions

h2 = k.join()
nip.v5(nip.catE(h2,h,matchsizes=True))

k.kernels

# nip.v5(nip.catE(obj, np.squeeze(k.convolve(obj)), nip.convolve(obj, nip.extract(h2,obj.shape)), nip.convolve(obj, h)))

obj2 = nip.randomDots(obj.shape,1000)

# strength=1.0
# coefficientMap = nip.cat((1.0+nip.xx(obj,freq='ftfreq'),1.0 + nip.yy(obj,freq='ftfreq')*strength,1.0+nip.xx(obj,freq='ftfreq')*strength,1.0+nip.yy(obj,freq='ftfreq')*strength))
# coefficientMap = (0.5+nip.xx(obj,freq='ftfreq'))
weights = nip.rr2(obj,freq='ftfreq')*4.0
weights[weights>1.0] = 1.0
myPhi = np.abs(np.mod((nip.phiphi(obj)/np.pi+1.0)*2,2.0)-1.0)
coefficientMap = weights*myPhi + (1-weights)*0.5

coefficientMap = nip.cat((1.0-coefficientMap, coefficientMap))
blended = k.convolve(obj2, coefficientMap)

# spatially varying convolution
nip.v5(nip.catE(k.convolve(obj2),blended))

# nip.v5(k.subSlice().convolve(obj2))
