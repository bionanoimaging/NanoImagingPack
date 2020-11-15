import NanoImagingPack as nip
import numpy as np
obj=nip.readim()
# obj = obj[128:256:,128:256:];
param = nip.PSF_PARAMS()
psf = nip.psf(obj,param)

dist = [16,16]
scip = 4
scan = nip.MakeScan(dist,[scip,scip])
nX = obj.shape[-1]//dist[-1]
nY = obj.shape[-2]//dist[-2]
multiSpot = nip.repmat(scan,[1,nX,nY])

illum = nip.convolve(multiSpot,psf,axes=[-1,-2])

emission = obj * illum
pimg = nip.convolve(emission,psf,axes=[-1,-2])
detection = nip.noise.poisson(pimg,NPhot=100)

# nip.v5(detection)
##
WF=np.mean(detection,-3)
Benedetti_1 = np.max(detection,-3) - np.min(detection,-3)
Benedetti_2 = np.max(detection,-3) + np.min(detection,-3) - 2*WF

nip.v5(nip.catE(obj,WF,Benedetti_1,Benedetti_2))
