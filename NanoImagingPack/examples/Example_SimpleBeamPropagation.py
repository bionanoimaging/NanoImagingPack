# An example propagating a Gaussian peak through vacuum
import NanoImagingPack as nip
import numpy as np

myshape = [100,100,100]
pixelsize = [100.0,100.0,100.0] # in nm
input_plane = nip.gaussian(myshape[-2:], sigma=[3.0,5.0]) * np.exp(1j*nip.xx(myshape[-2:])*2.0*np.pi*0.03)
input_plane.pixelsize=pixelsize[-2:]
PSFpara = nip.PSF_PARAMS();  # wavelength is 520nm
pupil = nip.ft(input_plane)
propagatedFT = nip.propagatePupil(pupil, sizeZ= myshape[0], distZ=pixelsize[0], psf_params=PSFpara)
propagated = nip.ift2d(propagatedFT)

nip.v5(propagated, showPhases=True, gamma=1.0)  # similar but not identical results!
