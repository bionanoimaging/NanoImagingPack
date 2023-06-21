# NanoImagingPack

This is a package for simple image processing. It is oriented on the DIP-Image
package, which is available for Matlab. The goal is to keep things simple and make
it available for a broad community.

 <!-- Hence its development aimed for: -->

<!-- - Make it independent from as much packages as possible -> it only requires a view Python packages. Some are provided by e.g. Anaconda (e.g. Numpy, pyPlot, inspect, scipy). Only tifffile (a 3D Tiff writer) is required as external package
- keep it simple to install -->

## Installation

1. Download Anaconda https://docs.anaconda.com/anaconda/install/
1. Open an anaconda prompt
1. create a new environment (tested in Python 3.8)
    ```
    conda create --name nanoimaging python=3.8 anaconda tifffile
    ```
1. Activate 
    ```
    conda activate nanoimaging
    ```
1. Install this feature branch of NanoImagingPack
    ```
    pip install git+https://gitlab.com/bionanoimaging/nanoimagingpack
    ```

<!-- 
conda create -y --name nanoimaging python=3.8 anaconda tifffile &
conda activate nanoimaging &
pip install git+https://gitlab.com/bionanoimaging/nanoimagingpack
 -->

## Getting started

Start an ipython shell
```
ipython
```
Load and view a sample image

```
import NanoImagingPack as nip
import napari

img = nip.readim("erika")
nip.vv(img)
```

The created image is of type "image"

## Gain calibration from an inhomogenous stack

Perform a gain calibration using simulated data.

```
import NanoImagingPack as nip
import numpy as np

# define the input parameters
NPHOT = 100 # max number of photons in simulation
OFFSET = 100 # black level offset
READNOISE = 4 # read noise to simulate
STACK_SIZE = 30

img = nip.readim("MITO_SIM")[0] # load the first frame from the MITO_SIM sample

fg = np.tile(img,(STACK_SIZE,1,1)) # make a stack of frames
fg = nip.poisson(fg, NPhot=NPHOT) # simulate poissonian shot noise

fg = fg + np.random.normal(loc=OFFSET, scale=READNOISE, size=fg.shape) # add gaussian noise

bg = np.random.normal(loc=OFFSET, scale=READNOISE, size=fg.shape) # generate background stack

# perform a calibration and plot the results
nip.cal_readnoise(fg, bg, brightness_blurring=False)

```


## Notes
    
* the command nip.lookfor('some string') allows you to search for functions and methods
* nip.view() or nip.vv() provides an image viewer
    * The default viewer is currently Napari, but this can be changed by nip.setDefault('IMG_VIEWER',myViewer) with myViewer being one of 'NIP_VIEW', 'VIEW5D','NAPARI','INFO'
    * in NIP_VIEW press 'h' to get help. In VIEW5d press "?"
* graph() provides a simple viewer for 2D graphs -> you can add lists for multiple graphs

# Features

* multidimensional fft/ifft
* image alignment
* finding for SLM based SIM images
* Controlling Hamamatsu LCOS SLM
* Creating OTFs, PSFs etc.
* Image manipulation (convolution, (simple) EdgeDamping, extracting, concattenating, Fourier-space extractions and padding)
* helper functions such as ramps, xx, yy, zz, freq/realspace coord transforms

See "dependencies.txt" for help required dependencies