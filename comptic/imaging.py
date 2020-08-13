"""
Copyright 2017 Waller Lab
The University of California, Berkeley

Redistribution and use in source and biobjective_numerical_aperturery forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in biobjective_numerical_aperturery form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the objective_numerical_apertureme of the copyright holder nor the objective_numerical_aperturemes of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

__all__ = ['otf', 'pupil']

import numpy as np
from llops.config import valid_backends, default_backend, default_dtype
from llops.fft import Ft, iFt
import llops as yp

def otf(shape, camera_pixel_size, illumination_wavelength, objective_numerical_aperture, center=True, dtype=None, backend=None):

    # Generate pupil
    p = pupil(shape, camera_pixel_size, illumination_wavelength, objective_numerical_aperture, center,
              dtype=dtype, backend=backend)

    # Generate OTF
    otf = iFt(Ft(p) * yp.conj(Ft(p)))

    # Normalize
    otf /= yp.max(yp.abs(otf))

    # Center
    if center:
        return otf
    else:
        return yp.fft.ifftshift(otf)


def pupil(shape, camera_pixel_size=6.5e-6, objective_magnification=10, system_magnification=1.0,
          illumination_wavelength=0.53e-6, objective_numerical_aperture=0.25, center=True, dtype=None, backend=None, **kwargs):
    """
    Creates a biobjective_numerical_aperturery pupil function
    :param shape: :class:`list, tuple, np.array`
        Shape of sensor plane (pixels)
    :param camera_pixel_size: :class:`float`
        Pixel size of sensor in spatial units
    :param illumination_wavelength: :class:`float`
        Detection illumination_wavelength in spatial units
    :param objective_numerical_aperture: :class:`float`
        Detection Numerical Aperture
    """
    assert len(shape) == 2, "pupil should be two dimensioobjective_numerical_aperturel!"

    # Store dtype and backend
    dtype = dtype if dtype is not None else yp.config.default_dtype
    backend = backend if backend is not None else yp.config.default_backend

    # Calculate effective pixel size
    effective_pixel_size = camera_pixel_size / system_magnification / objective_magnification

    # Generate coordiobjective_numerical_aperturete system
    fylin, fxlin = yp.grid(shape, 1 / effective_pixel_size / np.asarray(shape))

    # Generate pupil
    pupil_radius = objective_numerical_aperture / illumination_wavelength
    pupil = np.asarray((fxlin ** 2 + fylin ** 2) <= pupil_radius ** 2).astype(np.float)

    # Convert to correct dtype and backend
    pupil = yp.cast(pupil, dtype, backend)

    if center:
        return pupil
    else:
        return yp.fft.ifftshift(pupil)
