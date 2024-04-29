# Analysis tool for inhomogeneous illumination

David McFadden 2021-06-02

This gui tool uses the NanoImagingPack image processing library
(https://gitlab.com/bionanoimaging/nanoimagingpack/),
written in python. As it requires a number of other libraries, the size is
relatively large (~300MB). However, it should work stand-alone on Windows
machines without requiring installation of any other software.

Extract the zip archive to a new folder. There are two versions of the tool. One with a graphical user interface ("gui_calibration_tool.exe"), and a command line interface ("cli_calibration_tool.exe").

To get started, there are a few example files are included in the
"examples" zip archive. Extract these to a new folder if you wish.

## GUI

Run the program "gui_calibration_tool.exe".
You should already have both a stack of dark images and a stack of bright
images. These should be in the tiff format, and be **either** a sequence of 2D
files, **or** a file containing a single 3D stack.

Drag and drop the dark images into the panel on the left, and do the same for
the bright images on the right.

#### Process
The "Process" button reads the image files and shows plots of the fluctuations
across the images. These are a good sanity check for the acquired data. Finally,
a photon transfer curve is generated and various statistics, including the gain,
are printed on the left of the window. The plots are matplotlib windows and can
be interacted with.

#### Process and save
The "Process and save" button does the same, except that it allows users to
select an output directory to save the plots and the statistics to files.

The error lines surrounding the fit line do not attempt to follow the data
points, but are based on the observed standard deviation for the pixel variances
in the corresponding bin.


#### Options

##### Show histogram
Show a histogram of the pixel bins in the background of the plot

##### Skip first
Otherwise, the fit range automatically goes up to the 99th percentile of
brightest pixels and down to 5% of the range between offset and the 99th
percentile.

##### PNG/SVG
Save the plots as PNG or SVG files.

#### Advanced Options

##### Clip at
Clip data above specified value. This can be useful if the program isn't
correctly recognizing the bit-depth of the image

##### X Chunks / Y Chunks
The image can be divided across X and Y into sub-images ("chunks"), which are then evaluated separately and saved into subdirectories. This is useful if you want to inspect whether the image edges have different gains than the center. Keep in mind that this will reduce the statistics, so you may need much more data for informative results.

## CLI

The command line tool "cli_calibration_tool.exe" can be run from within a shell. Use the --help option to list the available options.

.\cli_calibration_tool.exe --help

A simple example includes a bright stack, a dark stack. If the exposure is at the saturation level, then the --saturation-image can yield further parameters:

.\cli_calibration_tool.exe "saturation_examples\lowres\fg" "saturation_examples\lowres\bg" --saturation-image

The optional arguments for fit and histograms ranges are given by the upper and lower bounds, seperated by a comma:

.\cli_calibration_tool.exe "saturation_examples\lowres\fg" "saturation_examples\lowres\bg" --saturation-image --hist-range=10,2000

## Example files

The "Examples" folder contains multiple example images acquired using the
betalight calibration source.

* EMCCD_conventional_mode: In the conventional, i.e. non-EM-gain mode, the line
  fits nicely to the curve.
* EMCCD_em_gain_96: The exposure is the same, but the EM mode is turned on and
  the gain set to 96. Notice that the brightness plot drifts as more images are
  acquired. Due to this, it is a good idea to skip the first 30 images. On the
  photon transfer curve, the pixels start to saturate and the curve flattens off
  at the upper end. The nonlinearity of the curve is also apparent when we
  compare it to the fit.
* sCMOS: For this example, the stack is split across many files. Select all dark
  images and drag and drop them into the panel. Do the same with the bright
  images. Notice that here, too, the first frame is clearly an anomaly which
  causes a brightness fluctuation of 21%. This also falsifies the gain estimate.
  Skipping the first image reduces the fluctuation to 0.5% and this changes gain
  estimate changes from 0.55 e-/ADU to 0.62 e-/ADU.

## Outputs

Results are output as graphical plots, and as a YAML text file "calibration_results.txt".
The meaning of the individual entries is elaborated in "definitions.txt".

Comments, questions, bugs and requests to:
* david.mcfadden777@gmail.com
* heintzmann@gmail.com
