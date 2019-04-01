# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:07:46 2019

@author: pi96doc
"""

from importlib import reload
import NanoImagingPack as nip
reload(nip)
from NanoImagingPack import v5

obj=nip.readim() # .astype("float32")
obj=obj[400:,200:]+200.0
psf=nip.psf2d(obj)

# nip.irft(nip.rft(psf,shift_before=True),psf.shape)
res=nip.irft(nip.rft(obj)*nip.rft(psf,shift_before=True),psf.shape)

#objd=nip.DampEdge(obj,rwidth=0.5,func=nip.cossqr) # method="zero"
objd=nip.DampEdge(obj, rwidth=0.1) # method="zero"
v5(nip.cat((nip.cat((objd,objd),-1),nip.cat((objd,objd),-1)),-2))

#%%

#objd=nip.DampEdge(obj) # method="zero"
objd2=nip.DampEdge(obj,method="moisan") # method="zero"
nip.cat((objd2,objd2),-1)
v5(nip.cat((nip.cat((objd2,objd2),-1),nip.cat((objd2,objd2),-1)),-2))

v5(nip.catE((objd,objd2)))
v5(nip.catE((nip.ft(objd),nip.ft(objd2))))


# nip.cat((objd[:,450],objd[:,450]),0)
