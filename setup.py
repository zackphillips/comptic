from setuptools import setup, find_packages
from setuptools.command.develop import develop
from subprocess import check_call
import os, sys
import subprocess
from os import path

# Define version
__version__ = 0.3

import sys

current_directory = os.getcwd()

# Find git submodules
sub_mod_bash_command = "git config --file " + \
   os.getcwd()+"/.gitmodules --get-regexp path | awk '{ print $2 }'"
bash_output = subprocess.check_output(['bash', '-c', sub_mod_bash_command])
sub_mods = [x for x in bash_output.decode("utf-8").split("\n") if x != '']

# Install submodule and it's requirements
for sub_mod in sub_mods:
     submod_setup_path = os.path.join(os.getcwd(), sub_mod, "setup.py")
     if os.path.exists(submod_setup_path) and (current_directory != submod_setup_path):
          print("Installing submodule at %s" % submod_setup_path)
          subprocess.call([sys.executable, submod_setup_path, sys.argv[1]])
     elif len(os.listdir(os.path.dirname(submod_setup_path))) == 0:
          subprocess.call('git submodule init')
          subprocess.call('git submodule update')

setup(name             = 'comptic',
      version          = __version__,
      description      = 'Computational Microscopy Helper Functions',
      license          = 'BSD',
      package_dir       = {"": ".", "llops": "./submodules/llops/llops"},
      packages         = find_packages() + find_packages(where='./submodules/llops'),
      include_package_data = True,
      package_data={'': ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.json']},
      install_requires = ['planar', 'sympy', 'numexpr', 'contexttimer', 'imageio', 'matplotlib_scalebar', 'tifffile', 'numpy', 'scipy', 'scikit-image'])
