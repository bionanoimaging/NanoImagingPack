from NanoImagingPack import cal_readnoise
import numpy as np
import tifffile
import types
from clize import run
import pathlib
import matplotlib.pyplot as plt
import ast



GUI = False
# def cal_readnoise(fg, bg, numBins=100, validRange=None, linearity_range=None, histRange=None, CameraName=None, correctBrightness=True,
#  correctOffsetDrift=True, exclude_hot_cold_pixels=True, noisy_pixel_percentile=98, doPlot=True, exportpath=None, exportFormat="png",
#  brightness_blurring=True, plotWithBgOffset=True, plotHist=False, check_bg=True, saturationImage=False):

# complete type annotation
# any way to
from clize import run, parser


def cal_readnoise_cli(bright_path, dark_path, *, skip_first:int=None, xchunks:int=1, ychunks:int=1,
                      numBins:int=100, validRange=None, linearity_range=None, histRange=None, CameraName=None, correctBrightness:bool=True,
                      correctOffsetDrift:bool=True, exclude_hot_cold_pixels:bool=True, noisy_pixel_percentile:float=98, doPlot:bool=True, exportpath:pathlib.Path="./results", exportFormat:str="png",
                      brightness_blurring:bool=True, plotWithBgOffset:bool=True, plotHist:bool=False, check_bg:bool=False, saturationImage:bool=False):
    """
    :param numBins: Number of bins for the histogram 
    :param histRange: If provided, the histogram will only be generated over this range.
    :param validRange: If provided, the gain fit will only be performed over this range of mean values.
    :param linearity_range: If provided, the linearity will only be evaluated over this range.
    :param CameraName: If provided, sets a plot title with CameraName
    :param correctBrightness: Attempt to correct the calibration for a fluctuating illumination brightness.
    :param correctOffsetDrift:
    :param exclude_hot_cold_pixels: Exclude hot and cold pixels from the fit
    :param noisy_pixel_percentile: Only include this percentile of least noisy pixels.
                                   Useful for supressing the effect of RTS noise. 
    :param doPlot: Plot the mean-variance curves
    :param exportpath: If provided, the plots will be saved to this directory.
    :param exportFormat: PNG or SVG files possible.
    :param brightness_blurring: A filter to blur the brightness estimate. Useful for sCMOS sensors
    :param plotWithBgOffset: If false, then the background value will be subtracted from the pixel ADUs in the plot.
    :param plotHist: If true, then a histogram of brightness bins will be plotted in the plot background
    :param check_bg: If true, then the background images will be checked.
    :param saturationImage: If true, then the peak of the photon transfer curve will be used to estimate the saturation level and calculate a dynamic range.
    :return: tuple of fit results (offset [adu], gain [electrons / adu], readnoise [e- RMS])
    """
    if validRange is not None:
        validRange = ast.literal_eval(validRange)
    if linearity_range is not None:
        linearity_range = ast.literal_eval(linearity_range)
    if histRange is not None:        
       histRange = ast.literal_eval(histRange)      

    def stack_frompaths(fullpaths):
        stack = []
        for file in fullpaths:
            img = tifffile.imread(file).squeeze()
            err_msg = "If input consists of multiple files, each must be a 2D image."
            try:
                assert img.ndim == 2
            except:
                if GUI: tkinter.messagebox.showinfo("Shape Error", err_msg)
                print(err_msg)
                return
            stack.append(img[:])
        stack = np.array(stack)
        return stack
    # collect either file list or 3D file, load as numpy array and pass to wrapped function

    dark = types.SimpleNamespace(path=pathlib.Path(dark_path))
    bright = types.SimpleNamespace(path=pathlib.Path(bright_path))

    for nameSp in (dark, bright):  #cli specific
        fullpath = nameSp.path

        # ----------------------------------------------------------------
        # shared
        # ----------------------------------------------------------------
        # could be dir or 3D file
        if fullpath.is_dir():
            files = list(fullpath.glob("*"))
            file_varieties = len(set([a.suffix for a in files]))
            try:
                assert file_varieties == 1
            except:
                err_msg = "Directory must contain a single file type"
                if GUI: tkinter.messagebox.showinfo("More than one file type", err_msg)
                print(err_msg)
                return
            stack = stack_frompaths(files)
        else:
            try:
                img = tifffile.imread(fullpath).squeeze()
                # img = nip.readim(str(fullpath)).squeeze()
            except tifffile.tifffile.TiffFileError as e:
                print(e, "trying numpy.load")
                img = np.load(fullpath).squeeze()
            try:
                assert img.ndim == 3, err_msg
            except:
                err_msg = "If input is a single file, it must be a 3D stack."
                if GUI: tkinter.messagebox.showinfo("Shape Error", err_msg)
                print(err_msg)
                return
            stack = img

        # ----------------------------------------------------------------
        stack = stack[skip_first:]
        nameSp.stack = stack
        # TODO: have it be a list of files
        # 

    fg = bright.stack
    bg = dark.stack
    exportpath = pathlib.Path(exportpath)
    exportpath.mkdir(parents=True, exist_ok=True)

    (bg_total_mean, gain, Readnoise, mean_el_per_exposure, validmap, figures, 
        Text) = cal_readnoise(fg, bg, numBins=numBins, validRange=validRange, linearity_range=linearity_range, histRange=histRange, CameraName=CameraName, correctBrightness=correctBrightness,
        correctOffsetDrift=correctOffsetDrift, exclude_hot_cold_pixels=exclude_hot_cold_pixels, noisy_pixel_percentile=noisy_pixel_percentile, doPlot=doPlot, exportpath=exportpath, exportFormat=exportFormat,
        brightness_blurring=brightness_blurring, plotWithBgOffset=plotWithBgOffset, plotHist=plotHist, check_bg=False, saturationImage=saturationImage)
    
    # Chunking evaluation
    plt.ioff()
    if (xchunks != 1) or (ychunks != 1):
        gain_profile = np.empty((ychunks, xchunks))
        # with array_split, we don't need to worry about the array not being divisible
        print(fg.shape)

        exportpath_parent = exportpath
        for ix, (fgxc, bgxc) in enumerate(zip(np.array_split(fg, xchunks, axis=-1), np.array_split(bg, xchunks, axis=-1))):
            print(fgxc.shape)
            for iy, (fgyc, bgyc) in enumerate(zip(np.array_split(fgxc, ychunks, axis=-2), np.array_split(bgxc, ychunks, axis=-2))):
                
                if exportpath_parent is not None:
                    exportpath = pathlib.Path(exportpath_parent/f"chunk_X{ix}Y{iy}")
                    exportpath.mkdir(parents=True, exist_ok=True)
                else:
                    exportpath = None
                (bg_total_mean, gain, Readnoise, mean_el_per_exposure, validmap, figures, 
                    Text) = cal_readnoise(fgyc, bgyc, numBins=numBins, validRange=validRange, linearity_range=linearity_range, histRange=histRange, CameraName=CameraName, correctBrightness=correctBrightness,
                    correctOffsetDrift=correctOffsetDrift, exclude_hot_cold_pixels=exclude_hot_cold_pixels, noisy_pixel_percentile=noisy_pixel_percentile, doPlot=doPlot, exportpath=exportpath, exportFormat=exportFormat,
                    brightness_blurring=brightness_blurring, plotWithBgOffset=plotWithBgOffset, plotHist=plotHist, check_bg=False, saturationImage=saturationImage)

                # (bg_total_mean, gain, Readnoise, mean_el_per_exposure, validmap, figures, Text) = nip.cal_readnoise(
                #     fgyc, bgyc, validRange=validRange, histRange=histRange, numBins=numBins, exportpath=exportpath, exportFormat=exportFormat,
                #     brightness_blurring=scmos_filter_var.get(), plotHist=showHist_var.get())

                gain_profile[iy, ix] = gain
                # Debug: save the image
                # nip.imsave(nip.image(fgyc.mean((0))), str(exportpath/"sub_image.tif"))
        if exportpath_parent is not None:
            np.savetxt(exportpath_parent/"gain_profile.csv", gain_profile)
    
    return


if __name__ == '__main__':
    run(cal_readnoise_cli)