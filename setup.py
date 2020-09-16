from setuptools import setup, find_packages
import os, sys
import subprocess

# Define version
__version__ = 0.02

setup( name             = 'comptic'
     , version          = __version__
     , description      = 'Computational Microscopy Helper Functions'
     , license          = 'BSD'
     , packages         = find_packages()
     , include_package_data = True
     , package_data={'': ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.json']}
     , install_requires = ['planar', 'sympy', 'numexpr', 'contexttimer', 'imageio', 'matplotlib_scalebar', 'tifffile', 'numpy', 'scipy', 'scikit-image', 'planar']
     )
