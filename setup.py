#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 23:38:28 2018

@author: grinse
"""

import setuptools


# If you want to incorporate c-files
# For the structure of an c-file -> Check resource files
#from distutils.core import  Extension
# define the extension module
# cos_module = Extension('cos_module', sources=['External_code/cos_module.c'])


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    # ext_mdoules = [External_code/cos_module],        <- for incorporating c-files
    name="NanoImagingPack",
    version="1.0.0 dev6",
    author="Christian Karras",
    author_email="christian-karras@gmx.de",
    description="A nice little image processing package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    install_requires=['tifffile', ],
    #dependency_links=['https://github.com/blink1073/tifffile'],
    #url="https://test.pypi.org/legacy/",   # <- Add gitHubLink here!
    include_package_data=True,
    #package_dir = {'':'NanoImagingPack'},
    package_data={'NanoImagingPack':['resources/*']},
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
)
