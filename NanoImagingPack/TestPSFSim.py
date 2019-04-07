import NanoImagingPack as nip
import numpy as np
from NanoImagingPack import v5, view
from importlib import reload
reload(nip)

obj = nip.readim()
psfparams = nip.PSF_PARAMS

hardAperture = nip.pupilAperture(obj,psfparams)
jincAperture = nip.jincAperture(obj,psfparams)

# v5(nip.catE((hardAperture,jincAperture)))

h = nip.psf2d(obj,psfparams)

obj3d = nip.readim("MITO_SIM")
h = nip.psf(obj3d, psfparams)

ObjTime=obj * np.exp(-nip.ramp1D(20, -4)/10)

v=v5(ObjTime)

# v.ElementName(1,'HALLLOO')
