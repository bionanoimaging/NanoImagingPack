Welcome to the NanoImagingPack's documentation!
====================================

NanoImagingPack (*NIP*) is meant to be used for image processing in microscopy. It hosts a large variety of useful tools which makes a microscopist's life much easier. It for examples offers most important routines for PSF creation (1D, 2D, 3D) and adjacent convolution for coherent and incoherent situations. 
Most of the naming convention and the namespaces, as well as the available methods are inspired by the open-source image processing library ```DIPLIB``` or ```DIPIAMGE``` for matlab from Bernd Rieger et al. available here [[DIP_IMAGE](http://www.diplib.org/)]. In large parts, *NIP* is a wrapper for Numpy and only runs on the CPU (so far). 

We will give a brief introduction into the core concept of the NanoImagingPack (NIP) and how you can use it for your everyday tasks in image processing with N-dimensional data. 


## Aim

```
- Make it independent from as much packages as possible -> it only requires a view Python packages. Some are provided by e.g. Anaconda (e.g. Numpy, pyPlot, inspect, scipy). Only tifffile (a 3D Tiff writer) is required as external package
- keep it simple to install 
- Open-Source for educational and academic use
```


## Features

```
- N-dimensional convolution
- PSF creation (N dimension)
- creator-functions (e.g. ramp, circle)
- interactive displaying methods (View5D, Napari, NIP-viewer)
- Filemanagement (input/output)
- Simulating camera behaviour (e.g. noise)
- multidimensional fft/ifft
- image alignment
- finding for SLM based SIM images
- Controlling Hamamatsu LCOS SLM
- Creating OTFs, PSFs etc.
- Image manipulation (convolution, (simple) EdgeDamping, extracting, concattenating, Fourier-space extractions and padding)
- helper functions such as ramps, xx, yy, zz, freq/realspace coord transforms
- ```lookfor('some string')``` allows you to search for available methods


```

## Installation

A guid on how to install NIP on your computer can be found in this [tutorial](install.md).

## Tutorials 

The tutorials start from the very basics of how to read, write and display image data. We will guide the user through the pipeline of creating a virtual microscope with realistic imaging properties. 

An introduction to the NIP ecosystem can be found in a growing list of tutorials:

- [Tutorial 1](tutorial_1.md) - Introduction and generator functions
- [Tutorial 2](tutorial_2.md) - Helper functions and Fourier transforms


## Origin

NIP is actively developed by members of the [Nanoimaging](https://nanoimaging.de/) Lab by [Rainer Heintzmann](https://scholar.google.com/citations?user=zWZsh0wAAAAJ&hl=de) at the [Leibniz Institute of Photonic Technology (IPHT)](https://www.leibniz-ipht.de/), Jena, Germany.


## License

MIT

