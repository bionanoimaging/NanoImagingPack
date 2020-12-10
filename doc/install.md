# Installation Instructions

## Download and install anaconda

Follow this [link](https://www.anaconda.com/distribution/#download-section) and choose the distribution for your operation system. 

Install for all users and add the PATH variable once you will be asked. Eventually restart your computer 


Open the Anacoda prompt (e.g. Start-> Anaconda command in Windows)

## Create the NIP Anaconda Environment

Type the following inside the Anaconda prompt:

```py
conda create -n IP python=3.6* pip git numpy matplotlib pillow git Spyder
conda activate IP
conda install -c conda-forge scikit-image pyqt=5.12.3
pip install tifffile==2020.2.16 scipy==1.3.3
pip install git+https://gitlab.com/bionanoimaging/nanoimagingpack
# pip install --extra-index-url https://test.pypi.org/simple/ NanoImagingPack
```


## Using NIP inside your Anaconda enviornment

In order to use the freshly installed NIP toolbox, you always need to activate it using the following comand:

```
conda activate IP
```

## NIP inside Google Colab 

Open a new [Google Colab document](https://colab.research.google.com/) and add the following lines in order to make the NIP toolbox working inside the cloud-based notebook:



## NIP inside Spyder 

Now you can open Spyder (or any other IDE) for programming with NIP in Python. 

Therfore type ```spyder```in the Anconda command line.

### GUI tweaks:

To get the most of Spyder + NIP you can do the following

- open by typing ‚spyder‘
- tools→ seetings→ ipython console → start
- add ```%gui qt```

This will open Napari in the external rendering environment. 


## NIP inside Jupyter Notebook

Start the Jupyter Notebook from the Anaconda Notebook by typing:

```jupyter notebook```

Inside a new Notebook you can import the NIP toolbox with the usual import statement:

```import NanoImagingPack```


## Troubleshooting

If you get an error message about NanoImagingPack not existing, check the following: 

1. Did you install nip in the conda environment you created. 
2. Did you start "jupyter notebook" from within that environment (the left side of your prompt has to be marked "IP"). 
3. Did you install jupyter WITHIN that conda environment ``pip install jupyter```