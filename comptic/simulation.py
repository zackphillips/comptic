'''
Copyright 2017 Zack Phillips, Waller lambd
The University of California, Berkeley

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import imageio
import json
import numpy as np
import os
from skimage.transform import resize
import llops as yp

# Default simulation shape
simulation_shape_default = (256, 256)

# Default image directory (relative path)
test_images_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'resources/test_images')

# Load image dictionary
with open(test_images_directory + '/index.json') as f:
    _image_dict = json.load(f)

def print_available_objects():
    for image_label in _image_dict:
        image = imageio.imread(test_images_directory + '/' + _image_dict[image_label]['filename'])
        print('%s : %d x %d (%s)' % (image_label, image.shape[0], image.shape[1], image.dtype))

def get_available_objects():
    return _image_dict.keys()

# Load images, process color, and resize
def _loadImage(image_label, shape, dtype=None, backend=None, **kwargs):

    # Determine backend and dtype
    backend = backend if backend is not None else yp.config.default_backend
    dtype = dtype if dtype is not None else yp.config.default_dtype

    # Load image
    image = np.asarray(imageio.imread(test_images_directory + '/' + _image_dict[image_label]['filename']))

    # Process color channel
    if yp.ndim(image) > 2:
        color_processing_mode = kwargs.get('color_channel', 'average')
        if color_processing_mode == 'average':
            image = np.mean(image, 2)
        elif color_processing_mode == None:
            pass
        else:
            assert type(color_processing_mode) in [np.int, int]
            image = image[:, :, int(color_processing_mode)]

    # Resize image if requested
    if shape is not None:

        # Warn if the measurement will be band-limited in the frequency domain
        if any([image.shape[i] < shape[i] for i in range(len(shape))]):
            print('WARNING : Raw image size (%d x %d) is smaller than requested size (%d x %d). Resolution will be lower than bandwidth of image.'
                  % (image.shape[0], image.shape[1], shape[0], shape[1]))

        # Perform resize operation
        image = resize(image, shape, mode=kwargs.get('reshape_mode', 'constant'),
                       preserve_range=True, anti_aliasing=kwargs.get('anti_aliasing',
                       False)).astype(np.float)

    return yp.cast(image, dtype, backend)

def object(absorption, shape=None, phase=None, **kwargs):
    return testObject(absorption, shape, phase, **kwargs)

def testObject(absorption, shape=None, phase=None, invert=False, invert_phase=False, dtype=None, backend=None, **kwargs):

    # Load absorption image
    test_object = _loadImage(absorption, shape, dtype, backend, **kwargs)

    # Normalize
    test_object -= yp.min(test_object)
    test_object /= yp.max(test_object)

    # invert if requested
    if invert:
        test_object = 1 - test_object

    # Apply correct range to absorption
    absorption_max, absorption_min = kwargs.get('max_value', 1.1),  kwargs.get('min_value', 0.9)
    test_object *= (absorption_max - absorption_min)
    test_object += absorption_min

    # Add phase if label is provided
    if phase:
        # Load phase image
        phase = _loadImage(phase, shape, **kwargs)

        # invert if requested
        if invert_phase:
            phase = 1 - phase

        # Normalize
        phase -= yp.min(phase)
        phase /= yp.max(phase)

        # Apply correct range to absorption
        phase_max, phase_min = kwargs.get('max_value_phase', 0),  kwargs.get('min_value_phase', 1)
        phase *= (phase_max - phase_min)
        phase += phase_min

        # Add phase to test_object
        test_object = yp.astype(test_object, 'complex32')
        test_object *= yp.exp(1j * yp.astype(yp.real(phase), yp.getDatatype(test_object)))

    # Cast to correct dtype and backend
    return yp.cast(test_object, dtype, backend)


def brain(shape=None, **kwargs):
    return testObject('brain', shape, **kwargs)


def ucb(shape=(512, 512), **kwargs):
    return testObject('ucblogo', shape, phase='ucbseal', invert_phase=True, **kwargs)


def california(shape=None, **kwargs):
    return testObject('california', shape, phase=None, **kwargs)


def cells(shape=(512, 512), **kwargs):
    return testObject('flourescense', shape, phase='blood', **kwargs)


def brainstrip(shape=None, **kwargs):
    return testObject('brainstrip', shape, **kwargs)


def cameraman(shape=None, **kwargs):
    return testObject('cameraman', shape, **kwargs)


def letters(shape=None, **kwargs):
    return testObject('A', shape, **kwargs)
