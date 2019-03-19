# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:07:46 2019

@author: pi96doc
"""

import NanoImagingPack as nip
import numpy as np


obj=nip.readim() # .astype("float32")
obj=obj[400:,200:]+200.0
psf=nip.psf2d(obj)

# nip.irft(nip.rft(psf,shift_before=True),psf.shape)
res=nip.irft(nip.rft(obj)*nip.rft(psf,shift_before=True),psf.shape)

#objd=nip.DampEdge(obj,rwidth=0.5,func=nip.cossqr) # method="zero"
objd=nip.DampEdge(obj, rwidth=0.1) # method="zero"
nip.cat((objd,objd),-1)
#objd=nip.DampEdge(obj) # method="zero"
objd2=nip.DampEdge(obj,method="moisan") # method="zero"
nip.cat((objd2,objd2),-1)

nip.cat((objd,objd2,nip.ft(objd),nip.ft(objd2)),2)

# nip.cat((objd[:,450],objd[:,450]),0)
