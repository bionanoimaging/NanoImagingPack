import NanoImagingPack as nip
obj = nip.readim()
obj2 = nip.rot2d(obj,10.3)  # rotate by 10.3 deg
obj2 = nip.extract(nip.shift(obj2,[10.2,-3.4]), obj.shape) # shift and match shape
objt, src, dst, v = nip.findTransformFromMarkers(obj,obj2) # requires user interaction
nip.v5(nip.catE((obj, objt, obj2)))
input('Please readjust your markers and press enter.\n')
objt, src, dst, v = nip.findTransformFromMarkers(obj,obj2,viewer=v) # for correcting positions
nip.v5(nip.catE((obj, objt, obj2))) # visualize again
