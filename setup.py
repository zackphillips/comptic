from setuptools import setup, find_packages
from setuptools.command.develop import develop
from subprocess import check_call
import os, sys
import subprocess
from os import path

# Define version
__version__ = 0.3



setup(name             = 'comptic',
      cmdclass = {"develop": update_submodules},
      version          = __version__,
      description      = 'Computational Microscopy Helper Functions',
      license          = 'BSD',
      package_dir       = {"": ".", "llops": "./submodules/llops/llops"},
      packages         = find_packages() + find_packages(where='./submodules/llops'),
      include_package_data = True,
      package_data={'': ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.json']},
      install_requires = ['planar', 'sympy', 'numexpr', 'contexttimer', 'imageio', 'matplotlib_scalebar', 'tifffile', 'numpy', 'scipy', 'scikit-image'])
