#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 23:38:28 2018

@author: grinse
"""

import setuptools
from __NIP_META__ import __author__, __author_email__, __description__, __License__, __Operating_system__, __Programming_language__,__title__, __url__, __version__

# If you want to incorporate c-files
# For the structure of an c-file -> Check resource files
#from distutils.core import  Extension
# define the extension module
# cos_module = Extension('cos_module', sources=['External_code/cos_module.c'])

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    # ext_mdoules = [External_code/cos_module],        <- for incorporating c-files
    name=__title__,
    version=__version__,
    author=__author__,
    author_email=__author_email__,
    description=__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # install_requires=['numpy>=1.16','scipy','tifffile', 'imageio', 'matplotlib', 'scikit-tensor-py3>0.2.1', 'napari'],  # scikit-tensor-py3 for sktensor
    install_requires=['numpy>=1.16',
                        'scipy',
                        'tifffile',
                        'imageio',
                        'matplotlib',
                        'napari',
                        'scikit-tensor-py3'],  # scikit-tensor-py3 for sktensor
    # 'python-bioformats', 'javabridge'  are removed from the requirements to make the basic installation simple.
    #dependency_links=['https://github.com/blink1073/tifffile'],
    #url="https://test.pypi.org/legacy/",   # <- Add gitHubLink here!
    include_package_data=True,
    #package_dir = {'':'NanoImagingPack'},
    package_data={'NanoImagingPack':['resources/*', 'examples/*', 'sim/*']},
    packages=setuptools.find_packages(),
    classifiers=[
        __Programming_language__,
        __License__,
        __Operating_system__,
    ],
)
