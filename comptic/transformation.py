'''
Copyright 2017 Zack Phillips, Waller lambd
The University of California, Berkeley

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import planar
import numpy as np
import llops as yp


def translation(shift):
    """Translation Matrix for 2D"""
    return np.asarray(planar.Affine.translation(shift)).reshape(3, 3)


def shear(shear_amount):
    """Shear Matrix in 2D"""
    return np.asarray(planar.Affine.shear(shear_amount[0], shear_amount[1])).reshape(3, 3)


def rotation(angle, pivot=None):
    """Shear Matrix in 2D"""
    return np.asarray(planar.Affine.rotation(angle)).reshape(3, 3)


def scale(scaling):
    """Shear Matrix in 2D"""
    return np.asarray(planar.Affine.scale(scaling)).reshape(3, 3)


def affineHomographyBlocks(coordinate_list):
    """Generate affine homography blocks which can be used to solve for homography coordinates."""
    # Store dtype and backend
    dtype = yp.getDatatype(coordinate_list)
    backend = yp.getBackend(coordinate_list)

    # Convert coordinates to numpy array
    coordinate_list = yp.asarray(yp.real(coordinate_list), dtype='float32', backend='numpy')

    # Ensure coordinate_list is a list of lists
    if yp.ndim(coordinate_list) == 1:
        coordinate_list = np.asarray([coordinate_list])

    # Determine the number of elements in a coordinate
    axis_count = yp.shape(coordinate_list)[1]

    # Loop over positions and concatenate to array
    coordinate_blocks = []
    for coordinate in coordinate_list:
        block = np.append(coordinate, 1)
        coordinate_blocks.append(np.kron(np.eye(len(coordinate)), block))

    # Convert to initial datatype and backend
    coordinate_blocks = yp.asarray(np.concatenate(coordinate_blocks), dtype, backend)

    return coordinate_blocks


class ImageRotation:
    """
    Rotation class create plans for a matrix with a given size that perform rotation using FFT
    """

    def __init__(self, shape, axis=0, pad=False, pad_value=0):

        dim = np.asarray(shape)
        self.axis = axis
        self.pad_value = pad_value

        if pad:
            self.pad_size = np.ceil(dim / 2).astype('int')
            self.pad_size[axis] = 0
            dim += 2 * self.pad_size
        else:
            self.pad_size = np.asarray([0, 0, 0])

        self.range_rot_axis_z = slice(self.pad_size[0], self.pad_size[0] + shape[0])
        self.range_rot_axis_y = slice(self.pad_size[1], self.pad_size[1] + shape[1])
        self.range_rot_axis_x = slice(self.pad_size[2], self.pad_size[2] + shape[2])

        self.z = np.arange(dim[0]) - dim[0] / 2
        self.y = np.arange(dim[1]) - dim[1] / 2
        self.x = np.arange(dim[2]) - dim[2] / 2

        self.kz = _genGrid(dim[0], 1. / dim[0], flag_shift=True)
        self.ky = _genGrid(dim[1], 1. / dim[1], flag_shift=True)
        self.kx = _genGrid(dim[2], 1. / dim[2], flag_shift=True)

        self.z_fft = Fourier(tuple(dim), (0,))
        self.y_fft = Fourier(tuple(dim), (1,))
        self.x_fft = Fourier(tuple(dim), (2,))

    def rotate(self, obj, theta):
        if theta == 0:
            return obj
        else:
            return self._rotate3DImage(obj, theta, self.pad_value)

    def rotate_adj(self, obj, theta):
        if theta == 0:
            return obj
        else:
            return self._rotate3DImage(obj, -1 * theta, 0)

    def _rotate3DImage(self, obj, theta, pad_value):
        """
        This function rotates a 3D image by shearing, (applied in Fourier space)
        ** Note: the rotation is performed along the z axis

        [ cos(theta)  -sin(theta) ] = [ 1  alpha ] * [ 1     0  ] * [ 1  alpha ]
        [ sin(theta)  cos(theta)  ]   [ 0    1   ]   [ beta  1  ]   [ 0    1   ]
        alpha = tan(theta/2)
        beta = -sin(theta)

        Shearing in one dimension is applying phase shift in 1D fourier transform
        Input:
            obj: 3D array (supposed to be an image), the axes are [z,y,x]
            theta: desired angle of rotation in *degrees*
        Output:
            obj_rotate: rotate 3D array
        """
        theta_resid = theta * (np.pi / 180.)
        naxis = np.newaxis
        alpha = 1. * np.tan(theta_resid / 2.)
        beta = np.sin(-theta_resid)

        range_z = self.range_rot_axis_z
        range_y = self.range_rot_axis_y
        range_x = self.range_rot_axis_x

        obj_rotate = np.pad(obj, ((self.pad_size[0], self.pad_size[0]),
                                  (self.pad_size[1], self.pad_size[1]),
                                  (self.pad_size[2], self.pad_size[2])), mode='constant', constant_values=pad_value)

        if self.axis == 0:
            x_phaseshift = np.exp(-2j * np.pi * self.kx[naxis, naxis, :] * self.y[naxis, :, naxis] * alpha)
            y_phaseshift = np.exp(-2j * np.pi * self.ky[naxis, :, naxis] * self.x[naxis, naxis, :] * beta)

            obj_rotate = self.x_fft.inverseFourierTransform(self.x_fft.fourierTransform(obj_rotate) * x_phaseshift)
            obj_rotate = self.y_fft.inverseFourierTransform(self.y_fft.fourierTransform(obj_rotate) * y_phaseshift)
            obj_rotate = self.x_fft.inverseFourierTransform(self.x_fft.fourierTransform(obj_rotate) * x_phaseshift)
        elif self.axis == 1:
            x_phaseshift = np.exp(-2j * np.pi * self.kx[naxis, naxis, :] * self.z[:, naxis, naxis] * alpha)
            z_phaseshift = np.exp(-2j * np.pi * self.kz[:, naxis, naxis] * self.x[naxis, naxis, :] * beta)

            obj_rotate = self.x_fft.inverseFourierTransform(self.x_fft.fourierTransform(obj_rotate) * x_phaseshift)
            obj_rotate = self.z_fft.inverseFourierTransform(self.z_fft.fourierTransform(obj_rotate) * z_phaseshift)
            obj_rotate = self.x_fft.inverseFourierTransform(self.x_fft.fourierTransform(obj_rotate) * x_phaseshift)
        elif self.axis == 2:
            z_phaseshift = np.exp(2j * np.pi * self.kz[:, naxis, naxis] * self.y[naxis, :, naxis] * alpha)
            y_phaseshift = np.exp(2j * np.pi * self.ky[naxis, :, naxis] * self.z[:, naxis, naxis] * beta)

            obj_rotate = self.z_fft.inverseFourierTransform(self.z_fft.fourierTransform(obj_rotate) * z_phaseshift)
            obj_rotate = self.y_fft.inverseFourierTransform(self.y_fft.fourierTransform(obj_rotate) * y_phaseshift)
            obj_rotate = self.z_fft.inverseFourierTransform(self.z_fft.fourierTransform(obj_rotate) * z_phaseshift)

        return obj_rotate[range_z, range_y, range_x]
