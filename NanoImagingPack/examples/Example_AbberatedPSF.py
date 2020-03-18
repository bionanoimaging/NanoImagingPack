# An example calculating an aberrated PSF based on the electric field vector theory of light

import NanoImagingPack as nip

obj = nip.readim('obj3d.tif', pixelsize=80) + 0.0
paraNoAber = nip.PSF_PARAMS();

paraAbber = nip.PSF_PARAMS();
aber_map = nip.xx(obj.shape[-2:]).normalize(1);  # Define some aberration map (x-ramp in this case)
paraAbber.aberration_types = [paraAbber.aberration_zernikes.spheric, (3, 5), aber_map]  # define list of aberrations (select from choice, zernike coeffs, or aberration map
paraAbber.aberration_strength = [0.5, 0.8, 0.2];
psf = nip.psf(obj, paraNoAber);
psfAbber = nip.psf(obj, paraAbber);

nip.vv(nip.catE(psf,psfAbber),gamma=0.5)  # similar but not identical results!
