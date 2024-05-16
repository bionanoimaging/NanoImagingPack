import NanoImagingPack as nip
import numpy as np
from scipy.linalg import lstsq
from matplotlib.pyplot import plot, figure

fg = nip.readtimeseries(r"C:\NoBackup\Data\FrankGarwe\tritium\ganzneu2\tritium_2pinholes_direkt_apertur0p15_1s_gain20_offset10\\")
bg = nip.readtimeseries(r"C:\NoBackup\Data\FrankGarwe\tritium\ganzneu2\tritium_2pinholes_direkt_apertur0p15_1s_gain20_offset10\background_1s_gain20_offset10_ganzneu2\\")
# otherbg = nip.readim(r"C:\NoBackup\Data\FrankGarwe\tritium\tritium_2pinholes_apertur_0p15_direct_1s_gain20_offset10_bg.tiff")
# nip.v5(a+0.0-bg)
nip.cal_readnoise(fg, bg) #

# lstsq()
# meanvarvar =

# b=nip.readtimeseries(r"C:\NoBackup\Data\FrankGarwe\tritium\neu3\\")
# nip.v5(b*1.0-b[-1])

if True:
    fg = nip.readim(r"C:\Users\pi96doc\Documents\MATLAB\Hamamatsu\10ms.BTF")
    bg = nip.readim(r"C:\Users\pi96doc\Documents\MATLAB\Hamamatsu\10msdark.BTF").squeeze()
    qq=nip.cal_readnoise(fg, bg) # validRange=[80,30000]
