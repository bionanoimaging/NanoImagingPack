This is a package for simple image processing.
It is oriented on the DIP-Image package, which is available for Matlab. 
Goal is to keep things simple and make it available for a broad community. Hence its development aimed for:

	- Make it independent from as much packages as possible -> it only requires a view Python packages. Some are provided by e.g. Anaconda (e.g. Numpy, pyPlot, inspect, scipy). Only tifffile (a 3D Tiff writer) is required as external package
	- keep it simple to install 


Getting started:
	- The created image is of type "image"

	- the command lookfor('some string') allows you to search for available methods

	- view() or vv() provides an image viewer
	    - The default viewer is currently Napari, but this can be changed by

	    	setDefault('IMG_VIEWER',myViewer) with myViewer being one of 'NIP_VIEW', 'VIEW5D','NAPARI','INFO'
	        - in NIP_VIEW press 'h' to get help. In VIEW5d press "?"

	- graph() provides a simple viewer for 2D graphs -> you can add lists for multiple graphs

	- supports:

		multidimensional fft/ifft

		image alignment

		finding for SLM based SIM images

		Controlling Hamamatsu LCOS SLM

		Creating OTFs, PSFs etc.

		Image manipulation (convolution, (simple) EdgeDamping, extracting, concattenating, Fourier-space extractions and padding)

		helper functions such as ramps, xx, yy, zz, freq/realspace coord transforms


See "dependencies.txt" for help required dependencies