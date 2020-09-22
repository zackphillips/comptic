import llops as yp
import numpy as np

def propKernelFresnelFourier(shape, pixel_size, wavelength, prop_distance, angle_deg=None, RI=1.0):
    '''
    Creates a fresnel propagation kernel in the Fourier Domain
    :param shape: :class:`list, tuple, np.array`
        Shape of sensor plane (pixels)
    :param pixel_size: :class:`float`
        Pixel size of sensor in spatial units
    :param wavelength: :class:`float`
        Detection wavelength in spatial units
    :param prop_distance: :class:`float`
        Propagation distance in spatial units
    :param angle_deg: :class:`tuple, list, np.array`
        Propagation angle, degrees
    :param RI: :class:`float`
        Refractive index of medium
    '''
    assert len(shape) == 2, "Propigation kernel size should be two dimensional!"

    # Determine propagation angle and spatial frequency
    angle = len(shape) * [0.0] if angle_deg is None else np.deg2rad(angle_deg)
    fy_illu, fx_illu = [RI * np.sin(a) / wavelength for a in angle]

    # Generate coordinate system
    fylin = _genLin(shape[0], 1 / pixel_size / shape[0])
    fxlin = _genLin(shape[1], 1 / pixel_size / shape[1])

    # Calculate wavenunmber
    k = (2.0 * np.pi / wavelength) * RI

    prop_kernel = yp.exp(1j * k * yp.abs(prop_distance)) * yp.exp(-1j * np.pi * wavelength * yp.abs(prop_distance) * ((fxlin[np.newaxis, :] - fx_illu) ** 2 + (fylin[:, np.newaxis] - fy_illu) ** 2))

    return prop_kernel if prop_distance >= 0 else prop_kernel.conj()

def propKernelFresnelReal(shape, pixel_size, wavelength, prop_distance, RI=1.0, position=None):
    '''
    Creates a fresnel propagation kernel in the Real Domain
    :param shape: :class:`list, tuple, np.array`
        Shape of sensor plane (pixels)
    :param pixel_size: :class:`float`
        Pixel size of sensor in spatial units
    :param wavelength: :class:`float`
        Detection wavelength in spatial units
    :param prop_distance: :class:`float`
        Propagation distance in spatial units
    :param RI: :class:`float`
        Refractive index of medium
    :param position: :class:`list, tuple, np.array`
        Position of particle center in spatial units
    '''
    assert len(shape) == 2, "Propigation kernel size should be two dimensional!"

    # Parse position input
    position = (0,0) if (not position or len(position) != 2) else position

    # Generate coordinate system
    ygrid, xgrid = yp.grid(shape, pixel_size, offset=position)

    # Divice by a common factor of 1000 to prevent overflow errors
    prop_distance /= 1000.
    wavelength /= 1000.
    ygrid /= 1000.
    xgrid /= 1000.

    # Calculate wavenunmber
    k = (2.0 * np.pi / wavelength) * RI

    # Generate propagation kernel (real-space)
    rr = xgrid ** 2 + ygrid ** 2

    # Generate propagation kernel
    prop_kernel = yp.exp(1j * k * prop_distance) / (1j * wavelength * prop_distance) * yp.exp(1j * k / (2 * prop_distance) * rr)

    # Return
    return prop_kernel

def propKernelRayleighSpatial(shape, pixel_size, wavelength, prop_distance):

    # Generate coordinate system
    ylin, xlin = yp.grid(shape, pixel_size)

    kb = 2 * np.pi / wavelength
    R2 = prop_distance ** 2 + xlin[np.newaxis,:] ** 2 + ylin[:,np.newaxis] ** 2
    R = yp.sqrt(R2)
    prop_kernel = yp.abs(prop_distance) * kb / (2j * np.pi) * yp.exp(1j * kb * R) * (1 + 1j / (kb * R)) / R2

    return prop_kernel if prop_distance >= 0 else prop_kernel.conj()

def propKernelRayleightFourier(shape, pixel_size, wavelength, prop_distance):

    # Generate coordinate system
    fylin, fxlin = yp.grid(shape, 1 / pixel_size / np.asarray(shape))

    # Generate Squared Coordinate System
    fy2 = (fylin ** 2)[:, np.newaxis]
    fx2 = (fxlin ** 2)[np.newaxis, :]
    fz2 = 1 / wavelength ** 2 - fy2 - fx2

    fz = np.lib.scimath.sqrt(fz2)
    prop_kernel = yp.exp(2j * np.pi * fz * yp.abs(prop_distance))

    return prop_kernel if prop_distance >= 0 else np.conj(prop_kernel)
