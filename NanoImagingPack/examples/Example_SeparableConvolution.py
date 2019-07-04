# An example demonstrating the separable deconvolution trick (see Martin Weigert's thesis for more details).

import NanoImagingPack as nip

obj = nip.readim('obj3d.tif', pixelsize=80) + 0.0
obj = nip.repmat(obj, [2,2,2])
q = nip.psf(obj)

if True:
    kernel = nip.separable(q, 3, kernellength=[31, 31, 31])
else:
    kernel = nip.separable(q, 3, kernellength=[5, 5, 5]) # only then a speedup (2x) is visible ....
joined = kernel.join()
# nip.vv(nip.catE(q,nip.extract(joined,q.shape))) # to compare the joined psfs

t1 = nip.timer()
c1 = nip.convolve(obj, q)   # classical FFT-based convolution. This could even be sped up if the kernel is pre-FFTed.
t1.add('FFT convolve')
c2 = kernel.convolve(obj)  # should be faster but really is not for realistic kernelsizes of 31,31,31 (about 2x slower)
t1.add('separable convolve')
mytimes = t1.get(mode = 'mut', comm= True)
nip.vv(nip.catE(c1, c2,obj))  # similar but not identical results!
