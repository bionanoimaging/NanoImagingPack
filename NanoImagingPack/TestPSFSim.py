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

h = nip.psf(nip.zeros([100, 100, 100]), psfparams)

# ObjTime=obj * np.exp(-nip.ramp1D(20, -4)/10)

# v=v5(ObjTime)

# v.NameElement(1,'HALLLOO')
q = nip.catE(obj3d, - obj3d)

# v5(q)
# v5(obj3d)

q = nip.rr()
q.pixelsize = [50,50]
q.unit = ['nm','µm','m','s','ns']
v = v5(q)
