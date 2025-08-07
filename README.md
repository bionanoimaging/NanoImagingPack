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
1. create a new environment
    ```
    conda create --name nanoimaging anaconda tifffile
    ```
1. Activate 
    ```
    conda activate nanoimaging
    ```
1. Install this feature branch of NanoImagingPack
    ```
    pip install git+https://github.com/bionanoimaging/NanoImagingPack.git
    ```

<!-- 
conda create -y --name nanoimaging anaconda tifffile &
conda activate nanoimaging &
pip install git+https://github.com/bionanoimaging/NanoImagingPack.git
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
viewer = napari.view_image(img)
```

The created image is of type "image".

## How to be able to use View5D (and python-bioformats)
To be able to use View5D as a viewer (eg. using nip.vv(mydata)) you need a Java installation in your system.
Here are the installation instructions based on a (recommended) tool called "uv".
This installation was tested (Aug. 2025) for Windows 11, 64 bit:
- Unpack a recent java JDK installation. Note that a JRE (as shipped with Fiji or ImageJ) is not sufficient. You can obtain an installation from [java]
(https://download.oracle.com/java/24/latest/jdk-24_windows-x64_bin.zip) or from [openJdk](https://openjdk.org/)
- Go to your system path definition and add the variable `JDK_HOME` to you user variables with the path of the java installation (e.g. `C:\NoBackup\java\jdk-24.0.2`) and add the corresponding `bin` folder (e.g. `C:\NoBackup\java\jdk-24.0.2\bin`) as one entry to the `PATH` variable. Be sure to use `JDK_HOME` and NOT `JAVA_HOME`.
- Install vsBuildTools 2022 (17.14.11) from `https://aka.ms/vs/17/release/vs_BuildTools.exe`. You need to select `Desktopdevelopment with C++` (top left) but it sufficient to select only the first two `optional` choices (up to Windows_11_SDK which is required!). This is needed to hava a working `cl.exe` to compile C code, which `javabridge` (see later) needs. I wonder, if there is a less heavy version via VSCode today. Let me know if you know one and tried it successfully via raising an issue.
- make a folder (e.g. uv-environments)  where you keep all your uv environments.
- open powershell (or cmd) and cd into this folder where you store the uv environments.
- Install uv via the powershell: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- create a new environment: `uv venv --python=3.11 nip-py11`. Be sure to use pyhton 3.11 and NOT python 3.12 or later.
- cd into this environment: `cd nip-py11`
- if you have not already done so allow script execution via: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`
- activate that environment: `.\Scripts\activate`
- install numpy into this environment: `uv pip install "numpy<1.24"`. Be sure to limit the numpy version to below 1.24
- verify that the environment is set correctly: `echo $env:JDK_HOME`should show you the corresponding path in that powershell. If not, you may need to restart the powershell.
- activate the cl environment `. "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"`. Note that this works in the powershell only if you have activated a uv environment. Otherwise you may need to use a cmd environment for the next steps.
- verify that cl is working now: `cl`  should result in a reply by the compiler. If this does not work, you may have to use a `cmd` environment for the next steps.
- install setuptools: `uv pip install setuptools`. This is needed to prevent an error in NanoImagingPack in the newer Python versions. Eventually this should be revised.
- install python-bioformats: `uv pip install python-bioformats`. It is essential that you do it this way. Do NOT try to install `javabridge` as this leads to lots of version trouble. The python-bioformats package seems to use matching versions here.
- install NanoImagingPack: `uv pip install  git+https://github.com/bionanoimaging/NanoImagingPack.git`. Note that the `git+` in the beginning is essential. Of course you can also clone the git repo and use the -e flag for pip and state only the local folder, if you plan to modify NanoImagingPack.
- Optionally you may want to install ipython: `uv pip install ipython`

You are now ready to go and do a fist test: e.g. type `uv run python` or `uv run ipython`
Here is a little test script to see if the viewer starts corretly:
```
import NanoImagingPack as nip
nip.setDefault('IMG_VIEWER','VIEW5D') # set default viewer to View5D()
q = nip.xx() # generate a ramp along x
v = nip.vv(q) # display the image in the Java viewer View5D.
v.setColormap(13, 0) # change colormap to RdBu
v.SetGamma(0, 0.3) # set the gamma of the colormap of element 0 to 0.3
```
Type `?` in the viewer to find more information on View5D or look [here](https://nanoimaging.de/View5D) or at the [videos](https://www.youtube.com/watch?v=fqa82MmJlAA&list=PL3LueK3ij6Wm2VjaaibNdulxFvA6VhVRv).







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
