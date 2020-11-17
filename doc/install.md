# Installation Instructions

## Download and install anacondaFollow this [link](https://www.anaconda.com/distribution/#download-section) and choose the distribution for your operation system. 

Install for all users and add the PATH variable once you will be asked. Eventually restart your computer 


Open the Anacoda prompt (e.g. Start-> Anaconda command in Windows)

## Create the NIP Anaconda Environment

Type the following inside the Anaconda prompt:

```conda create -n IP python=3.6* pip git numpy matplotlib pillow git Spyderconda activate IPconda install -c conda-forge scikit-image pyqt=5.12.3pip install tifffile==2020.2.16 scipy==1.3.3pip install git+https://gitlab.com/bionanoimaging/nanoimagingpack# pip install --extra-index-url https://test.pypi.org/simple/ NanoImagingPack
```

Now you can open Spyder (or any other IDE) for programming with NIP in Python. 

Therfore type ```spyder```in the Anconda command line.

## GUI tweaks:

To get the most of Spyder + NIP you can do the following- open by typing ‚spyder‘
- tools→ seetings→ ipython console → start
- add ```%gui qt```

This will open Napari in the external rendering environment. 