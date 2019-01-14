# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:07:46 2019

@author: pi96doc
"""

import NanoImagingPack as nip
import numpy as np

obj=nip.readim() # .astype("float32")
psf=nip.psf2d(obj)
obj=obj[400:,200:]+200.0

objd=nip.DampEdge(obj,rwidth=0.5,func=nip.cossqr) # method="zero"
#objd=nip.DampEdge(obj,rwidth=0.5) # method="zero"
#objd=nip.DampEdge(obj) # method="zero"

nip.cat((objd[:,450],objd[:,450]),0)

