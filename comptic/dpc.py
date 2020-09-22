import numpy as np
import copy
import llops as yp
import matplotlib.pyplot as plt
from . import imaging

# Valid source types used in the functions below
valid_source_types = ['top', 'bottom', 'left', 'right',
                      'monopole_top', 'monopole_bottom', 'monopole_left', 'monopole_right',
                      'dipole_vertical', 'dipole_horizontal', 'monopole', 'dipole',
                      'half', 'third', 'quadrant', 'tripole_top', 'tripole',
                      'tripole_bottom_right', 'tripole_bottom_left']

# Define system parameters for our system
_system_params_default = {
                             'pixel_count': (2580, 2180),
                             'objective_numerical_aperture': 0.25,
                             'reflection_mode': False,
                             'objective_magnification': 10.2,
                             'system_magnification': 1.0,
                             'camera_pixel_size': 6.5e-6,
                             'camera_is_color': False,
                             'camera_readout_time': 0.032,  # seconds
                             'camera_quantum_efficency': 0.55,
                             'camera_max_counts': 65535,
                             'camera_dark_current': 0.9,
                             'camera_readout_noise': 3.85,
                             'illumination_wavelength': 0.53e-6,
                             'illumination_source_position_list_na': []
                         }


def getDefaultSystemParams(**kwargs):
    """Returns a dict of default optical system parameters.

    Parameters
    ----------
    **kwargs :
        key-value pairs which will over-ride default parameters

    Returns
    -------
    system_parameters: dict
        Dict containing common system parameters
    """

    params = copy.deepcopy(_system_params_default)
    for key in kwargs:
        if key in params:
            params[key] = kwargs[key]

    return params


def genDpcSourceList(shape,
                     dpc_type_list=('top', 'bottom', 'left', 'right'),
                     angle=0,
                     objective_numerical_aperture=0.25,
                     illumination_inner_na=0.0,
                     camera_pixel_size=6.5e-6,
                     objective_magnification=10,
                     system_magnification=1.0,
                     illumination_wavelength=0.53e-6,
                     center=True,
                     dtype=None,
                     backend=None,
                     **kwargs):
    """Generate list of rasterized DPC source patterns.

    Parameters
    ----------
    shape : tuple
        Shape of measurements

    dpc_type_list : list of str, optional
        List of escriptors of DPC types. Can be one of:
             "top": Standard half-circle, top orientation
             "bottom": Standard half-circle, bottom orientation
             "left": Standard half-circle, left orientation
             "right": Standard half-circle, right orientation
             "monopole, Monopole source, top orientation
             "monopole_top": Monopole source, top orientation
             "monopole_bottom": Monopole source, bottom orientation
             "monopole_left": Monopole source, left orientation
             "monopole_right, Monopole source, right orientation
             "dipole_vertical": Dipole source, vertical orientation
             "dipole_horizontal": Dipole source, horizontal orientation
             "dipole": Dipole source, vertical orientation
             "tripole": Tripole Source, spaced radially with the first above the origin
             "tripole_top": Tripole Source, spaced radially with the first above the origin
             "tripole_bottom_left": Tripole Source, spaced radially at bottom left point
             "tripole_bottom_right": Tripole Source, spaced radially at bottom right point
             "half": Source with 180 degree coverage, set orientation with angle kwarg
             "third": Source with 120 degree coverage, set orientation with angle kwarg
             "quadrant": Source with 90 degree coverage, set orientation with angle kwarg

    angle : float, optional
        Angle of source. Used only with "half", "third", or "quadrant" orientations
    objective_numerical_aperture : float, optional
        Objective numerical aperture
    illumination_inner_na : float, optional
        Inner NA cutoff of source
    camera_pixel_size : float, optional
        Pixel size of camera, usually microns
    objective_magnification : float, optional
        Magnification of objective. Total magnification is calculated as
        the product of this and system_magnification keyword argument.
    system_magnification : float, optional
        Magnification of system (not including objective). Total
        magnification is calculated as the product of this and
        system_magnification keyword argument.
    illumination_wavelength : float, optional
        Wavelength of illuminaiton source, usually microns
    center : bool, optional
        Whether to center (fftshift) the output
    dtype : str, optional
        Desired datatype of the source list (valid options are provided by
        llops.valid_datatypes)
    backend : str, optional
        Desired backend of the source list (valid options are provided by
        llops.valid_backends)

    """

    # Parse source types
    if type(dpc_type_list) not in (tuple, list):
        dpc_type_list = [dpc_type_list]

    # Ensure all source types are valid
    for dpc_type in dpc_type_list:
        assert dpc_type.lower() in valid_source_types

    # Generate inner source
    source_inner = imaging.pupil(shape, camera_pixel_size=camera_pixel_size,
                                         objective_magnification=objective_magnification,
                                         system_magnification=system_magnification,
                                         illumination_wavelength=illumination_wavelength,
                                         objective_numerical_aperture=illumination_inner_na,
                                         center=center, dtype=dtype, backend=backend)

    # Generate outer source
    source_outer = imaging.pupil(shape,
                                         camera_pixel_size=camera_pixel_size,
                                         objective_magnification=objective_magnification,
                                         system_magnification=system_magnification,
                                         illumination_wavelength=illumination_wavelength,
                                         objective_numerical_aperture=objective_numerical_aperture,
                                         center=center, dtype=dtype, backend=backend)

    # Get base source to filter for each source
    source_base = source_outer - source_inner

    # Generate coordinates
    yy, xx = yp.grid(shape)

    # Generate radius of pupil in px
    effective_pixel_size = camera_pixel_size / system_magnification / objective_magnification
    ps_fourier = 1 / effective_pixel_size / np.asarray(shape)
    pupil_radius_px = int(np.round((objective_numerical_aperture / illumination_wavelength) / ps_fourier)[0])

    # Helper funtcion for a rotated monopole
    def rotated_monopole(source_base, angle=0):
        from llops.geometry import rotation_matrix_2d
        led_pattern = yp.zeros_like(source_base)

        # Generate tempalate point and rotate
        template_point = (0, -int(pupil_radius_px) + 2) # x,y
        point = np.round(rotation_matrix_2d(np.deg2rad(0 + angle)).dot(template_point))

        # Assign point
        led_pattern[int(point[1] - shape[0] // 2), int(point[0] - shape[1] // 2)] = 1

        # Return
        return led_pattern

    # Generate LED source patterns
    source_list = []
    for dpc_type in dpc_type_list:
        if dpc_type.lower() == 'half':
            led_pattern = source_base * (yy * yp.sin(np.deg2rad(angle)) >= xx * yp.cos(np.deg2rad(angle)))
        elif dpc_type.lower() == 'top':
            led_pattern = source_base * (yy > 0)
        elif dpc_type.lower() == 'bottom':
            led_pattern = source_base * (yy < 0)
        elif dpc_type.lower() == 'left':
            led_pattern = source_base * (xx < 0)
        elif dpc_type.lower() == 'right':
            led_pattern = source_base * (xx > 0)
        elif dpc_type.lower() == 'monopole':
            led_pattern = rotated_monopole(source_base, 0 + angle)
        elif dpc_type.lower() == 'monopole_left':
            led_pattern = rotated_monopole(source_base, 270 + angle)
        elif dpc_type.lower() == 'monopole_right':
            led_pattern = rotated_monopole(source_base, 90 + angle)
        elif dpc_type.lower() == 'monopole_top':
            led_pattern = rotated_monopole(source_base, 0 + angle)
        elif dpc_type.lower() == 'monopole_bottom':
            led_pattern = rotated_monopole(source_base, 180 + angle)
        elif dpc_type.lower() == 'dipole':
            led_pattern = rotated_monopole(source_base, 0 + angle) + rotated_monopole(source_base, 180 + angle)
        elif dpc_type.lower() == 'dipole_vertical':
            led_pattern = rotated_monopole(source_base, 0 + angle) + rotated_monopole(source_base, 180 + angle)
        elif dpc_type.lower() == 'dipole_horizontal':
            led_pattern = rotated_monopole(source_base, 90 + angle) + rotated_monopole(source_base, 270 + angle)
        elif dpc_type.lower() == 'tripole':
            led_pattern = rotated_monopole(source_base, 0 + angle)
        elif dpc_type.lower() == 'tripole_top':
            led_pattern = rotated_monopole(source_base, 0 + angle)
        elif dpc_type.lower() == 'tripole_bottom_right':
            led_pattern = rotated_monopole(source_base, 120 + angle)
        elif dpc_type.lower() == 'tripole_bottom_left':
            led_pattern = rotated_monopole(source_base, 240 + angle)
        elif dpc_type.lower() == 'half':
            led_pattern = source_base
            led_pattern *= (yy * yp.sin(np.deg2rad(angle + 180)) <= xx * yp.cos(np.deg2rad(angle + 180)))
            led_pattern *= (yy * yp.sin(np.deg2rad(angle)) >= xx * yp.cos(np.deg2rad(angle)))
        elif dpc_type.lower() == 'third':
            led_pattern = source_base
            led_pattern *= (yy * yp.sin(np.deg2rad(angle + 120)) <= xx * yp.cos(np.deg2rad(angle + 120)))
            led_pattern *= (yy * yp.sin(np.deg2rad(angle)) >= xx * yp.cos(np.deg2rad(angle)))
        elif dpc_type.lower() == 'quadrant':
            led_pattern = source_base
            led_pattern *= (yy * yp.sin(np.deg2rad(angle + 90)) <= xx * yp.cos(np.deg2rad(angle + 90)))
            led_pattern *= (yy * yp.sin(np.deg2rad(angle)) >= xx * yp.cos(np.deg2rad(angle)))
        else:
            raise ValueError('Invalid dpc type %s' % dpc_type)

        # Append to list
        source_list.append(led_pattern)

    # Return
    if len(source_list) == 1:
        return source_list[0]
    else:
        return source_list


def genDpcSourcePositionList(dpc_type_list=('top', 'bottom', 'left', 'right'),
                             objective_numerical_aperture=0.25,
                             angle=0.0,
                             illumination_inner_na=0.0,
                             illumination_source_position_list_na=[],
                             **kwargs):
    """Generate list of DPC source patterns as illumination intensities of a
       discrete source.

    Parameters
    ----------
    shape : tuple
        Shape of measurements

    dpc_type_list : list of str, optional
        List of escriptors of DPC types. Can be one of:
             "top": Standard half-circle, top orientation
             "bottom": Standard half-circle, bottom orientation
             "left": Standard half-circle, left orientation
             "right": Standard half-circle, right orientation
             "monopole_top": Monopole source, top orientation
             "monopole_bottom": Monopole source, bottom orientation
             "monopole_left": Monopole source, left orientation
             "monopole_right, Monopole source, right orientation
             "dipole_top": Dipole source, top orientation
             "dipole_bottom": Dipole source, bottom orientation
             "dipole_left": Dipole source, left orientation
             "dipole_right, Dipole source, right orientation
             "half": Source with 180 degree coverage, set orientation with angle kwarg
             "third": Source with 120 degree coverage, set orientation with angle kwarg
             "quadrant": Source with 90 degree coverage, set orientation with angle kwarg

    illumination_source_position_list_na : list of list, optional
        List of source positions formatted as NA_y, NA_x tuples
    angle : float, optional
        Angle of source. Used only with "half", "third", or "quadrant" orientations
    objective_numerical_aperture : float, optional
        Objective numerical aperture
    illumination_inner_na : float, optional
        Inner NA cutoff of source

    Returns
    -------

    led_pattern_list : list
        List of led intensities for each LED coordinate in illumination_source_position_list_na
    """

    # Parse angle
    if angle != 0.0:
        raise NotImplementedError

    # Parse inner illumination na
    if illumination_inner_na != 0.0:
        raise NotImplementedError

    # Get LEDs within numerical aperture
    illumination_source_position_list_na = np.asarray(illumination_source_position_list_na)
    bf_led_mask = [np.sqrt(src[0] ** 2 + src[1] ** 2) <= objective_numerical_aperture for src in illumination_source_position_list_na]

    # Generate LED patterns
    led_pattern_list = []
    for dpc_type in dpc_type_list:
        if dpc_type.lower() == 'top':
            led_pattern = yp.cast(bf_led_mask, 'float32')
            led_pattern *= yp.cast(illumination_source_position_list_na[:, 0] >= 0, 'float32')
        elif dpc_type.lower() == 'bottom':
            led_pattern = yp.cast(bf_led_mask, 'float32')
            led_pattern *= yp.cast(illumination_source_position_list_na[:, 0] <= 0, 'float32')
        elif dpc_type.lower() == 'left':
            led_pattern = yp.cast(bf_led_mask, 'float32')
            led_pattern *= yp.cast(illumination_source_position_list_na[:, 1] <= 0, 'float32')
        elif dpc_type.lower() == 'right':
            led_pattern = yp.cast(bf_led_mask, 'float32')
            led_pattern *= yp.cast(illumination_source_position_list_na[:, 1] >= 0, 'float32')

        # Append to list
        led_pattern_list.append(led_pattern)

    # Return
    return led_pattern_list

def plotWotfCoverage(Hi_list, Hr_list, figsize=None):
    import matplotlib.pyplot as plt

    coverage_real = sum(yp.abs(Hr) for Hr in Hr_list)
    coverage_imag = sum(yp.abs(Hi) for Hi in Hi_list)

    plt.figure(figsize=figsize)
    plt.subplot(121)
    plt.imshow(coverage_real)
    plt.title('$H_r$ Coverage')
    plt.subplot(122)
    plt.imshow(coverage_imag)
    plt.title('$H_i$ Coverage')


def plotWotfList(Hr_list, Hi_list, labels=None, **kwargs):
    """Plots pairs of WOTFs.

    Parameters
    ----------
    Hr_list : list of arrays
        List of real WOTFs
    Hi_list : list of arrays
        List of real WOTFs
    source_labels : list, optional
        List of str which label each source used to generate Hr_list and Hi_list


    Returns
    -------

    """

    # Update labels for WOTFs
    if labels is not None:
        labels = [l + ' $H_r$' for l in labels] + [l + ' $H_i$' for l in labels]

    # Create plot
    yp.listPlotFlat(Hr_list + [yp.imag(Hi) for Hi in Hi_list], labels, max_width=4)


def plotSourceList(led_source_list,
                   illumination_source_position_list_na=0.25,
                   objective_numerical_aperture=None,
                   labels=None,
                   background_color='k',
                   zoom=True,
                   **kwargs):
    """Plot list of source patterns.

    Parameters
    ----------
    led_pattern_list : list
        List of of normalized LED intensities, one float per LED. Ordering
        corresponds to the same ordering in illumination_source_position_list_na.
    illumination_source_position_list_na : list of list, optional
        List of source positions formatted as NA_y, NA_x tuples
    objective_numerical_aperture : float, optional
        Objective numerical aperture
    source_labels : list, optional
        List of str which label each source
    background_color : str, optional
        Background color for source positions
    zoom : bool, optional
        Whether to zoom into used LED positions


    Returns
    -------

    """

    # Plot sources
    yp.listPlotFlat(led_source_list, labels)


def plotSourcePositionList(led_pattern_list,
                           illumination_source_position_list_na,
                           objective_numerical_aperture=None,
                           labels=None,
                           background_color='k',
                           zoom=True,
                           **kwargs):
    """Plot list of source patterns.

    Parameters
    ----------
    led_pattern_list : list
        List of of normalized LED intensities, one float per LED. Ordering
        corresponds to the same ordering in illumination_source_position_list_na.
    illumination_source_position_list_na : list of list, optional
        List of source positions formatted as NA_y, NA_x tuples
    objective_numerical_aperture : float, optional
        Objective numerical aperture
    source_labels : list, optional
        List of str which label each source
    background_color : str, optional
        Background color for source positions
    zoom : bool, optional
        Whether to zoom into used LED positions


    Returns
    -------

    """

    # Convert source position list to numpy array
    illumination_source_position_list_na = np.asarray(illumination_source_position_list_na)

    # Generate figure
    plt.figure(figsize=(14, 4))

    # Plot DPC patterns
    for index, pattern in enumerate(led_pattern_list):

        plt.subplot(141 + index)
        plt.scatter(illumination_source_position_list_na[:, 1], illumination_source_position_list_na[:, 0], c=background_color)
        plt.scatter(illumination_source_position_list_na[:, 1], illumination_source_position_list_na[:, 0], c=pattern)
        plt.axis('square')

        if zoom:
            plt.xlim((-1.5 * objective_numerical_aperture, 1.5 * objective_numerical_aperture))
            plt.ylim((-1.5 * objective_numerical_aperture, 1.5 * objective_numerical_aperture))
        else:
            plt.xlim((-1, 1))
            plt.ylim((-1, 1))

        if objective_numerical_aperture:
            circle = plt.Circle((0, 0), objective_numerical_aperture, color='r', fill=False, linewidth=4)
            plt.gca().add_artist(circle)

        if labels:
            plt.title(labels[index])


def genMeasurementNonLinear(field,
                            led_pattern,
                            illumination_source_position_list_na=[],
                            objective_numerical_aperture=0.25,
                            objective_magnification=10,
                            system_magnification=1.0,
                            camera_pixel_size=6.5e-3,
                            illumination_wavelength=0.53e-6,
                            **kwargs):
    """Generates DPC measurements from a led pattern and complex field using a
       non-linear (partially-coherent) forward model.

    Parameters
    ----------

    field : array
        Complex field to use for generating measurements
    led_pattern : list
        List of led patterns (intensities for each led in illumination_source_position_list_na)
    illumination_source_position_list_na : list of list, optional
        List of source positions formatted as NA_y, NA_x tuples
    objective_numerical_aperture : float, optional
        Objective numerical aperture
    objective_magnification : float, optional
        Magnification of objective. Total magnification is calculated as
        the product of this and system_magnification keyword argument.
    system_magnification : float, optional
        Magnification of system (not including objective). Total
        magnification is calculated as the product of this and
        system_magnification keyword argument.
    camera_pixel_size : float, optional
        Pixel size of camera, usually microns
    illumination_wavelength : float, optional
        Wavelength of illuminaiton source, usually microns

    Returns
    -------
    intensity : array
        The simulated intensity using a non-linear forward model.

    """

    # Generate pupil
    pupil = imaging.pupil(yp.shape(field),
                                  camera_pixel_size=camera_pixel_size,
                                  objective_magnification=objective_magnification,
                                  system_magnification=system_magnification,
                                  illumination_wavelength=illumination_wavelength,
                                  objective_numerical_aperture=objective_numerical_aperture,
                                  center=True,
                                  dtype=yp.getDatatype(field),
                                  backend=yp.getBackend(field))

    # Calculate Effective Pixel Size
    effective_pixel_size = camera_pixel_size / objective_magnification / system_magnification

    # Generate real-space grid
    yy, xx = yp.grid(yp.shape(field), effective_pixel_size)

    intensity = yp.zeros_like(field)
    for led_power, led_na in zip(led_pattern, illumination_source_position_list_na):
        
        # Only process non-zero LED powers
        if led_power > 0.0:

            # Generate Illumination
            illumination = np.exp(-1j * 2 * np.pi * (led_na[1] / illumination_wavelength * xx + led_na[0] / illumination_wavelength * yy))

            # Generate intensity
            intensity = intensity + led_power * yp.abs(yp.iFt(yp.Ft(illumination * field) * pupil)) ** 2.0

    return intensity


def genMeasurementsLinear(field,
                          led_pattern_list,
                          illumination_source_position_list_na=[],
                          objective_numerical_aperture=0.25,
                          objective_magnification=10,
                          system_magnification=1.0,
                          camera_pixel_size=6.5e-3,
                          illumination_wavelength=0.53e-6,
                          **kwargs):
    """Generates DPC measurements from a led pattern and complex field using a
       linear (WOTF-based) forward model.

    Parameters
    ----------

    field : array
        Complex field to use for generating measurements
    led_pattern : list
        List of led patterns (intensities for each led in illumination_source_position_list_na)
    illumination_source_position_list_na : list of list, optional
        List of source positions formatted as NA_y, NA_x tuples
    objective_numerical_aperture : float, optional
        Objective numerical aperture
    objective_magnification : float, optional
        Magnification of objective. Total magnification is calculated as
        the product of this and system_magnification keyword argument.
    system_magnification : float, optional
        Magnification of system (not including objective). Total
        magnification is calculated as the product of this and
        system_magnification keyword argument.
    camera_pixel_size : float, optional
        Pixel size of camera, usually microns
    illumination_wavelength : float, optional
        Wavelength of illuminaiton source, usually microns

    Returns
    -------
    intensity : array
        The simulated intensity using a linear forward model.

    """

    # Generate WOTFs
    Hr_list, Hi_list = genWotfsFromSourcePositionList(yp.shape(field),
                                                      led_pattern_list,
                                                      illumination_wavelength=illumination_wavelength,
                                                      illumination_source_position_list_na=illumination_source_position_list_na,
                                                      objective_magnification=objective_magnification,
                                                      objective_numerical_aperture=objective_numerical_aperture,
                                                      system_magnification=system_magnification,
                                                      camera_pixel_size=camera_pixel_size)

    # Generate Intensity
    intensity_list = []
    for (Hr, Hi) in zip(Hr_list, Hi_list):
        intensity_list.append(yp.real(1 + yp.iFt(Hr * yp.Ft(yp.abs(field)) + Hi * yp.Ft(yp.angle(field)))))

    return intensity_list


def rasterizeSourcePositionList(shape,
                                led_pattern,
                                illumination_source_position_list_na=[],
                                objective_numerical_aperture=0.25,
                                objective_magnification=10,
                                system_magnification=1.0,
                                camera_pixel_size=6.5e-3,
                                illumination_wavelength=0.53e-6,
                                dtype=None, backend=None, **kwargs):
    """Convert a LED pattern (list of led intensities corresponding to each
       position in illumination_source_position_list_na) to a rasterized source.

    Parameters
    ----------

    shape : tuple
        Desired dimensions of rasterized source
    led_pattern : list
        List of led patterns (intensities for each led in illumination_source_position_list_na)
    illumination_source_position_list_na : list of list, optional
        List of source positions formatted as NA_y, NA_x tuples
    objective_numerical_aperture : float, optional
        Objective numerical aperture
    objective_magnification : float, optional
        Magnification of objective. Total magnification is calculated as
        the product of this and system_magnification keyword argument.
    system_magnification : float, optional
        Magnification of system (not including objective). Total
        magnification is calculated as the product of this and
        system_magnification keyword argument.
    camera_pixel_size : float, optional
        Pixel size of camera, usually microns
    illumination_wavelength : float, optional
        Wavelength of illuminaiton source, usually microns
    dtype : str, optional
        Desired datatype of the source list (valid options are provided by
        llops.valid_datatypes)
    backend : str, optional
        Desired backend of the source list (valid options are provided by
        llops.valid_backends)

    Returns
    -------
    intensity : array
        The simulated intensity using a linear forward model.

    """

    # If we're passed a list, iterate over each source position list and return a list of rasterized sources.
    if np.ndim(np.asarray(led_pattern)) == 2:
        rasterized_source_list = []
        for _pattern in led_pattern:
            rasterized_source_list.append(rasterizeSourcePositionList(shape,
                                                                      _pattern,
                                                                      illumination_source_position_list_na=illumination_source_position_list_na,
                                                                      objective_numerical_aperture=objective_numerical_aperture,
                                                                      objective_magnification=objective_magnification,
                                                                      system_magnification=system_magnification,
                                                                      camera_pixel_size=camera_pixel_size,
                                                                      illumination_wavelength=illumination_wavelength,
                                                                      dtype=dtype, backend=backend, **kwargs))

        # Return this list
        return rasterized_source_list

    # Calculate Effective Pixel Size
    effective_pixel_size = camera_pixel_size / objective_magnification / system_magnification

    # Generate fourier-space grid
    kyy, kxx = yp.grid(shape, 1 / effective_pixel_size / np.asarray(shape))

    # Generate source
    source = yp.zeros(shape, dtype, backend)
    for led_power, led_na in zip(led_pattern, illumination_source_position_list_na):

        # Calculate distance from illumination point
        dist = (kyy - led_na[0] / illumination_wavelength) ** 2 + (kxx - led_na[1] / illumination_wavelength) ** 2
        # Assign the correct position in the source
        source[yp.argmin(dist)] = led_power

    return source


def invert(measurement_list, Hr_list, Hi_list, reg_real=1e-8, reg_imag=1e-8):
    """Perform direct DPC inversion using Tikhonov regularization.

    Parameters
    ----------
    measurement_list : list
        List of measurements to invert
    Hr_list : list of arrays
        List of real WOTFs
    Hi_list : list of arrays
        List of real WOTFs
    reg_real : float, optional
        Tikhonov regularization parameter to apply to Hr term
    reg_imag : float, optional
        Tikhonov regularization parameter to apply to Hi term

    Returns
    -------
    field : array
        Recovered complex field
    """


    # Build AHA
    AHA_11 = sum([yp.abs(Hr) ** 2 for Hr in Hr_list]) + reg_real
    AHA_12 = sum([yp.conj(Hr) * Hi for Hr, Hi in zip(Hr_list, Hi_list)])
    AHA_21 = sum([yp.conj(Hi) * Hr for Hr, Hi in zip(Hr_list, Hi_list)])
    AHA_22 = sum([yp.abs(Hi) ** 2 for Hi in Hi_list]) + reg_imag
    AHA = [AHA_11, AHA_12, AHA_21, AHA_22]

    # Calculate determinant
    determinant = AHA[0] * AHA[3] - AHA[1] * AHA[2]

    # Generate measurement list in Fourier domain
    measurement_list_fourier = np.asarray([yp.Ft(yp.dcopy(meas)) for meas in measurement_list])

    # Generate AHy
    AH_y_real = sum([yp.conj(Hr) * y_f for Hr, y_f in zip(Hr_list, measurement_list_fourier)])
    AH_y_imag = sum([yp.conj(Hi) * y_f for Hi, y_f in zip(Hi_list, measurement_list_fourier)])
    AHy = [AH_y_real, AH_y_imag]

    # Solve for absorption
    absorption = yp.real(yp.iFt((AHA[3] * AHy[0] - AHA[1] * AHy[1]) / determinant))

    # Solve for phase
    phase = yp.real(yp.iFt((AHA[0] * AHy[1] - AHA[2] * AHy[0]) / determinant))

    # Generate complex field
    field = absorption + 1.0j * phase

    # Return
    return field


def genWotfsFromSourceList(source_list, pupil=None,
                           objective_numerical_aperture=0.25,
                           objective_magnification=10,
                           system_magnification=1.0,
                           camera_pixel_size=6.5e-3,
                           illumination_wavelength=0.53e-6,
                           **kwargs):
    """Generate list of WOTFs from a list of LED patterns (list of led
       intensities corresponding to each position in illumination_source_position_list_na).

    Parameters
    ----------

    source_list : list
        List of rasterized (2D) sources
    pupil : array, optional
        A pupil to use, if desired
    objective_numerical_aperture : float, optional
        Objective numerical aperture
    objective_magnification : float, optional
        Magnification of objective. Total magnification is calculated as
        the product of this and system_magnification keyword argument.
    system_magnification : float, optional
        Magnification of system (not including objective). Total
        magnification is calculated as the product of this and
        system_magnification keyword argument.
    camera_pixel_size : float, optional
        Pixel size of camera, usually microns
    illumination_wavelength : float, optional
        Wavelength of illuminaiton source, usually microns

    Returns
    -------
    wotf_list : list
        List of tuples containing arrays for (Hr, Hi) respectively

    """

    # Generate pupil
    if pupil is None:
        pupil = imaging.pupil(yp.shape(source_list[0]),
                                      objective_numerical_aperture=objective_numerical_aperture,
                                      objective_magnification=objective_magnification,
                                      system_magnification=system_magnification,
                                      camera_pixel_size=camera_pixel_size,
                                      illumination_wavelength=illumination_wavelength,
                                      dtype=yp.getDatatype(source_list[0]),
                                      backend=yp.getBackend(source_list[0]))

    # Generate WOTFs and return
    return genWotfPair(source_list, pupil, illumination_wavelength)


def genWotfsFromSourcePositionList(shape,
                                   led_pattern_list,
                                   illumination_source_position_list_na=[],
                                   objective_numerical_aperture=0.25,
                                   objective_magnification=10,
                                   system_magnification=1.0,
                                   camera_pixel_size=6.5e-3,
                                   illumination_wavelength=0.53e-6,
                                   dtype=None, backend=None, **kwargs):
    """Generate list of WOTFs from a list of LED patterns (list of led
       intensities corresponding to each position in illumination_source_position_list_na).

    Parameters
    ----------

    shape : tuple
        Desired dimensions of rasterized source
    led_pattern : list
        List of led patterns (intensities for each led in illumination_source_position_list_na)
    illumination_source_position_list_na : list of list, optional
        List of source positions formatted as NA_y, NA_x tuples
    objective_numerical_aperture : float, optional
        Objective numerical aperture
    objective_magnification : float, optional
        Magnification of objective. Total magnification is calculated as
        the product of this and system_magnification keyword argument.
    system_magnification : float, optional
        Magnification of system (not including objective). Total
        magnification is calculated as the product of this and
        system_magnification keyword argument.
    camera_pixel_size : float, optional
        Pixel size of camera, usually microns
    illumination_wavelength : float, optional
        Wavelength of illuminaiton source, usually microns
    dtype : str, optional
        Desired datatype of the source list (valid options are provided by
        llops.valid_datatypes)
    backend : str, optional
        Desired backend of the source list (valid options are provided by
        llops.valid_backends)

    Returns
    -------
    wotf_list : list
        List of tuples containing arrays for (Hr, Hi) respectively

    """

    # Generate sources
    source_list = [rasterizeSourcePositionList(shape, led_pattern,
                                               illumination_source_position_list_na=illumination_source_position_list_na,
                                               objective_numerical_aperture=objective_numerical_aperture,
                                               objective_magnification=objective_magnification,
                                               system_magnification=system_magnification,
                                               camera_pixel_size=camera_pixel_size,
                                               illumination_wavelength=illumination_wavelength,
                                               dtype=dtype, backend=backend, **kwargs) for led_pattern in led_pattern_list]

    # Generate pupil
    pupil = imaging.pupil(shape,
                                  objective_numerical_aperture=objective_numerical_aperture,
                                  objective_magnification=objective_magnification,
                                  system_magnification=system_magnification,
                                  camera_pixel_size=camera_pixel_size,
                                  illumination_wavelength=illumination_wavelength,
                                  dtype=dtype,
                                  backend=backend)

    # Generate WOTFs and return
    return genWotfPair(source_list, pupil, illumination_wavelength)

# Generate Hu and Hp
def genWotfPair(source, pupil, illumination_wavelength, **kwargs):
    """Function which generates the Weak-Object Transfer Functions (WOTFs) for absorption (Hu) and phase (Hp) given a source and pupul list

    Parameters
    ----------
        source : array
            ndarray of source in the Fourier domain
        pupil : array
            ndarray of source in the Fourier domain
        illumination_wavelength : float, optional
            Wavelength of illuminaiton source, usually microns


    Returns
    -------
    wotf_pair : tuple
        Pair of arrays corresponding to (Hr, Hi) WOTFs
    """

    # If a list of sources is provided, return a list of WOTF pairs
    if np.ndim(source) == 3:
        # Repeat pupil if necessary
        if np.ndim(pupil) == 2:
            pupil = [pupil] * len(source)

        # Generate WOTF pairs
        Hr_list, Hi_list = [], []
        for _source, _pupil in zip(source, pupil):
            Hr, Hi = genWotfPair(_source, _pupil, illumination_wavelength)
            Hr_list += [Hr]
            Hi_list += [Hi]

        # Return
        return Hr_list, Hi_list

    # Calculate source power
    DC = yp.sum((yp.abs(pupil) ** 2 * source))

    # Ensure source is non-zeros
    if DC == 0:
        print("WARNING: Source must be non-zero")

    # Generate WOTF
    M = yp.Ft(source * pupil) * yp.conj(yp.Ft(pupil))
    Hu = 2 * yp.iFt(np.real(M)) / DC
    Hp = 1j * 2 * yp.iFt(1j * np.imag(M)) / DC

    # Cast both as complex
    Hu = yp.astype(Hu, 'complex32')
    Hp = yp.astype(Hp, 'complex32')

    # Return
    return (Hu, Hp)

# Default illumination parameters
illumination_power_dict = {'VAOL': 1000.42,
                           'APTF_RED': 6830.60,        # quasi-dome, red
                           'APTF_GREEN': 392759.57,    # quasi-dome, green
                           'APTF_BLUE': 360655.74}     # quasi-dome, blue


def singularValuesFromWotfList(Hr_list, Hi_list, eps=None, support=None, method='direct'):
    """Returns a list of the singular values squared which are > epsilon.

        Parameters
        ----------
        Hr_list : list of arrays
            List of real WOTFs
        Hi_list : list of arrays
            List of real WOTFs
        eps : float, optional
            Threshold for determining support, if support argument is not provided
        support : array, optional
            Binary array indicating the support from which to return singular values
        method : str, optional
            Method for calculating singular values, can be 'direct', 'svd', or 'eig'. Direct should be used for all purposes except comparison as it does not form the full matrix.


        Raises
        ------
        ValueError

        Returns
        -------
        singular_values : array
            List of singular values, sorted largest to smallest
    """

    # Parse threshold
    if eps is None:
        eps = 1e-4

    # Contraction helper function
    def contract(x):
        return yp.sum(x, axis=0)[0, :]

    # Calculate support
    if support is None:
        support_Hr = yp.squeeze(yp.sum(yp.abs(np.asarray(Hr_list)), axis=0) > eps)
        support_Hi = yp.squeeze(yp.sum(yp.abs(np.asarray(Hi_list)), axis=0) > eps)
        support = support_Hr * support_Hi

    # Crop Hr and Hi list to support
    Hr_list_crop = [Hr[support] for Hr in Hr_list]
    Hi_list_crop = [Hi[support] for Hi in Hi_list]

    # Convert lists to array
    Hr = yp.asarray(Hr_list_crop)
    Hi = yp.asarray(Hi_list_crop)

    # Build AHA
    AHA = [contract(Hr.conj() * Hr), contract(Hr.conj() * Hi),
           contract(Hi.conj() * Hr), contract(Hi.conj() * Hi)]

    if method == 'direct':
        # Calculate trace and determinant
        tr = AHA[0] + AHA[3]
        det = AHA[0] * AHA[3] - AHA[1] * AHA[2]

        # Generate eigenvalues
        lambda_1 = tr / 2 + np.sqrt(tr ** 2 / 4 - det)
        lambda_2 = tr / 2 - np.sqrt(tr ** 2 / 4 - det)

        # Append to list
        # This equals singular values squared (sigma_i^2 in appendix a), in paper
        singular_values_squared = np.abs(np.append(lambda_1, lambda_2))
    elif method == 'svd':
        A = np.hstack((np.vstack([np.diag(Hr) for Hr in Hr_list_crop]),
                       np.vstack([np.diag(Hi) for Hi in Hi_list_crop])))

        U, S, V = np.linalg.svd(A)
        singular_values_squared = S ** 2

    elif method == 'eig':
        AHA_full = np.hstack((np.vstack((np.diag(AHA[0]), np.diag(AHA[1]))),
                              np.vstack((np.diag(AHA[2]), np.diag(AHA[3])))))

        singular_values_squared, _ = np.linalg.eig(AHA_full)

    # Return
    return np.asarray(sorted(np.abs(singular_values_squared), reverse=True))


def dnfFromWotfList(Hr_list, Hi_list, eps=None, support=None, method='direct'):
    """Calculate DNF from WOTF List.

    Parameters
    ----------
    Hr_list : list of arrays
        List of real WOTFs
    Hi_list : list of arrays
        List of real WOTFs
    eps : float, optional
        Threshold for determining support, if support argument is not provided
    support : array, optional
        Binary array indicating the support from which to return singular values
    method : str, optional
        Method for calculating singular values, can be 'direct', 'svd', or 'eig'. Direct should be used for all purposes except comparison as it does not form the full matrix.


    Raises
    ------
    ValueError

    Returns
    -------
    dnf : float
        Deconvolution noise factor
    """

    # Calculate singular values
    sigma_squared = singularValuesFromWotfList(Hr_list, Hi_list, eps=eps,
                                               support=support, method=method)

    # Calculate DNF
    f_squared = yp.abs(yp.sum(1 / sigma_squared) / len(sigma_squared))

    # Deal with nan values
    if yp.isnan(f_squared):
        f_squared = np.inf

    # Return
    return np.real(yp.sqrt(f_squared))


def dnfFromSourcePositionList(led_pattern_list, shape=(64, 64), pupil=None,
                              eps=None, support=None,
                              illumination_source_position_list_na=[],
                              **system_params):
    """Calculate DNF from persecribed source pattern.

    Parameters
    ----------
    led_pattern_list : list
        List of of normalized LED intensities, one float per LED. Ordering
        corresponds to the same ordering in illumination_source_position_list_na.
    shape : tuple, optional
        Shape to use for generating rasterized sources
    eps : float, optional
        Threshold for determining support, if support argument is not provided
    support : array, optional
        Binary array indicating the support from which to return singular values
    method : str, optional
        Method for calculating singular values, can be 'direct', 'svd', or 'eig'. Direct should be used for all purposes except comparison as it does not form the full matrix.

    Raises
    ------
    ValueError

    Returns
    -------
    dnf : float
        Deconvolution noise factor
    """

    # Generate Hi and Hr
    from . import genWotfsFromSourcePositionList
    Hr_list, Hi_list = genWotfsFromSourcePositionList(shape, led_pattern_list, **system_params)

    # Calculate DNF and return
    return dnfFromWotfList(Hr_list, Hi_list, eps=eps, support=support)


def dnfFromSourceList(source_list, pupil=None,
                      eps=None, support=None, **system_params):
    """Calculate DNF from persecribed source pattern."""

    # Generate Hi and Hr
    from . import genWotfsFromSourceList
    Hr_list, Hi_list = genWotfsFromSourceList(source_list, pupil=pupil, **system_params)

    # Calculate DNF and return
    return dnfFromWotfList(Hr_list, Hi_list, eps=eps, support=support)


def genRandomIlluminationPattern(method='binary', gamma=None, pattern_type=None,
                                 minimum_numerical_aperture=0,
                                 illumination_source_position_list_na=[],
                                 objective_numerical_aperture=0.25,
                                 dtype=None, backend=None, **kwargs):

    # Get LEDs within numerical aperture
    illumination_source_position_list_na = np.asarray(illumination_source_position_list_na)
    brightfield_led_indicies = [index for index, src in enumerate(illumination_source_position_list_na)
                                if (np.sqrt(src[0] ** 2 + src[1] ** 2) <= objective_numerical_aperture and np.sqrt(src[0] ** 2 + src[1] ** 2) > minimum_numerical_aperture)]

    # Parse pattern type
    if not pattern_type:
        pattern_type = 'full'
    else:
        assert pattern_type.lower() in ['full', 'top', 'bottom', 'left', 'right']

    # Filter based on pattern type
    if pattern_type.lower() == 'top':
        pattern_indicies = [index for index in brightfield_led_indicies if illumination_source_position_list_na[index, 1] > 0]
    elif pattern_type.lower() == 'bottom':
        pattern_indicies = [index for index in brightfield_led_indicies if illumination_source_position_list_na[index, 1] < 0]
    elif pattern_type.lower() == 'left':
        pattern_indicies = [index for index in brightfield_led_indicies if illumination_source_position_list_na[index, 0] < 0]
    elif pattern_type.lower() == 'right':
        pattern_indicies = [index for index in brightfield_led_indicies if illumination_source_position_list_na[index, 0] > 0]
    elif pattern_type.lower() == 'full':
        pattern_indicies = brightfield_led_indicies

    # Calculate number of LEDs
    led_count = len(pattern_indicies)

    # Initialize default gamma
    if not gamma:
        gamma = led_count // 2

    # Initialize LED pattern
    led_pattern = yp.zeros(len(illumination_source_position_list_na), dtype=dtype, backend=backend)

    # Generate blur kernel
    if method == 'random_phase' or method == "binary":
        # Calculate number of binary LEDs which are illumiunated
        illuminated_led_count = gamma // 2 * 2

        # Generate LED indicies used
        indicies = [pattern_indicies[index] for index in np.random.choice(led_count, replace=False, size=illuminated_led_count).tolist()]

        # Assign
        for index in indicies:
            led_pattern[index] = 1.0
    elif method == 'random' or method == "grayscale":

        # Generate random LED illumination values
        brightfield_led_intensity = np.random.uniform(size=led_count)

        # Assign
        for intensity, index in zip(brightfield_led_intensity, pattern_indicies):
            led_pattern[index] = intensity
    else:
        assert False, "method " + method + " unrecognized"

    # Return LED Pattern
    return led_pattern


def supportMask(shape, min_na=None, max_na=None, camera_pixel_size=6.5e-6,
                   objective_magnification=10, system_magnification=1.0,
                   illumination_wavelength=0.53e-6,
                   objective_numerical_aperture=0.25,
                   center=True, dtype=None, backend=None, **kwargs):

    # Parse minimum NA
    if min_na is None:
        min_na = 0.0

    # Parse maximum NA
    if max_na is None:
        max_na = 2 * objective_numerical_aperture

    # Generate support donut
    pupil_inner = comptic.imaging.pupil(shape, camera_pixel_size=camera_pixel_size,
                           objective_magnification=objective_magnification,
                           system_magnification=system_magnification,
                           illumination_wavelength=illumination_wavelength,
                           objective_numerical_aperture=min_na,
                           center=center, dtype=dtype, backend=backend)

    # Generate support donut
    pupil_outer = comptic.imaging.pupil(shape, camera_pixel_size=camera_pixel_size,
                           objective_magnification=objective_magnification,
                           system_magnification=system_magnification,
                           illumination_wavelength=illumination_wavelength,
                           objective_numerical_aperture=max_na,
                           center=center, dtype=dtype, backend=backend)

    return (pupil_outer - pupil_inner) > 0


def optimizeSourceRandom(shape, method='binary', minimum_na=0.05,
                         maximum_na=None, source_type_list=None, gamma=None,
                         iteration_count=100, minimum_numerical_aperture=0, **system_params):

    # Parse source_type_list
    if source_type_list is None:
        source_type_list = ['top', 'bottom', 'left', 'right']

    # Generate support
    support = genSupportMask(shape, min_na=minimum_na, max_na=maximum_na, **system_params, dtype='int')

    # Generate pupil
    pupil = comptic.imaging.pupil(shape, **system_params)

    # Generate candidate patterns
    candidate_pattern_list = []
    for _ in yp.display.progressBar(range(iteration_count), name='Illumination Patterns Generated'):
        pattern_list = [genRandomIlluminationPattern(method=method,
                                                     gamma=gamma,
                                                     pattern_type=name,
                                                     minimum_numerical_aperture=minimum_numerical_aperture,
                                                     **system_params) for name in source_type_list]

        # Calculate DNF
        f = dnfFromSourcePositionList(pattern_list, shape, support=support, pupil=pupil, **system_params)

        # Append to List
        candidate_pattern_list.append((f, pattern_list))

    # Sort candidate patterns
    candidate_pattern_list_sorted = sorted(candidate_pattern_list, key=lambda x: x[0], reverse=False)

    # Return
    return candidate_pattern_list_sorted[0]


def illuminanceToPhotonPixelRate(illuminance,
                                 objective_numerical_aperture=1.0,
                                 illumination_wavelength=0.55e-6,
                                 camera_pixel_size=6.5e-6,
                                 objective_magnification=1,
                                 system_magnification=1,
                                 sample_quantum_yield=1.,
                                 **kwargs):

    """
    Function which converts source illuminance and microscope parameters to
    photons / px / s.

    Based heavily on the publication:
    "When Does Computational Imaging Improve Performance?,"
    O. Cossairt, M. Gupta and S.K. Nayar,
    IEEE Transactions on Image Processing,
    Vol. 22, No. 2, pp. 447–458, Aug. 2012.

    However, this function implements the same result for
    microscopy, replacing f/# with NA, removing reflectance,
    and including magnification.

    Args:
     exposure_time: Integration time, s
     source_illuminance: Photometric source illuminance, lux
     numerical_aperture: System numerical aperture
     pixel_size: Pixel size of detector, um
     magnification: Magnification of imaging system

    Returns:
      Photon counts at the camera.
    """

    # Conversion factor from radiometric to photometric cordinates
    # https://www.thorlabs.de/catalogPages/506.pdf
    K = 1 / 680

    # Planck's constant
    # h_bar = 6.626176e-34
    h_bar = 1.054572e-34

    # Speed of light
    c = 2.9979e8

    # Constant term
    const = K * illumination_wavelength / h_bar / c

    # Calculate photon_pixel_rate
    photon_pixel_rate = sample_quantum_yield * const * (objective_numerical_aperture ** 2) * illuminance * (camera_pixel_size / (system_magnification * objective_magnification)) ** 2

    # Return
    return photon_pixel_rate


def photonPixelRateToNoiseComponents(photon_pixel_rate,
                                     exposure_time,
                                     dnf=1,
                                     camera_quantum_efficency=0.6,
                                     camera_dark_current=0.9,
                                     camera_ad_conversion=0.46,
                                     pulse_time=None,
                                     camera_readout_noise=2.5,
                                     camera_max_counts=65535,
                                     **kwargs):

    """
    Function which calculates the variance of signal dependent noise and signal
    independent noise components.

    Args:
        photon_pixel_rate: Number of photons per pixel per second
        exposure_time: Integration time, s
        dnf: Deconvolution noise factor as specified in Agrawal et. al.
        camera_quantum_efficency: QE of camera, in [0,1]
        camera_dark_current: Dark current from datasheet, electrons / s
        camera_readout_noise: Readout noise from datasheet, electrons
        camera_max_counts: Maximum discrete bins of camera, usually 255 or 65535.

    Returns:
        (noise_var_dependent, noise_var_independent)
    """

    # Return zero if either photon_pixel_rate or exposure_time are zero
    if photon_pixel_rate * exposure_time == 0:
        print("Photon pixel rate or exposure time are zero")
        return 0, 0, 0

    # Signal term
    if pulse_time is None:
        signal_mean_counts = (photon_pixel_rate * camera_quantum_efficency * exposure_time)
    else:
        signal_mean_counts = (photon_pixel_rate * camera_quantum_efficency * pulse_time)

    # Ensure the camera isnt saturating
    if signal_mean_counts > camera_max_counts * camera_ad_conversion:
        print("Exceeding camera max counts")
        return 0, 0, 0

    # Signal-independent noise term
    noise_independent_e = (camera_readout_noise / camera_ad_conversion)

    # Signal-dependent noise term
    noise_dependent_e = np.sqrt(signal_mean_counts + camera_dark_current * exposure_time / camera_ad_conversion)

    # Return
    return signal_mean_counts, noise_dependent_e, noise_independent_e


def photonPixelRateToSnr(photon_pixel_rate, exposure_time,
                         dnf=1,
                         camera_quantum_efficency=0.6,
                         camera_dark_current=0.9,
                         camera_ad_conversion=0.46,
                         pulse_time=None,
                         camera_readout_noise=2.5,
                         camera_max_counts=65535, debug=False, **kwargs):
    """
    Function which converts deconvolution noise factor to signal to noise ratio.
    Uses equations from https://www.photometrics.com/resources/learningzone/signaltonoiseratio.php and the dnf from the Agrawal and Raskar 2009 CVPR paper found here: http://ieeexplore.ieee.org/document/5206546/
    Default values are for the PCO.edge 5.5 sCMOS camera (https://www.pco.de/fileadmin/user_upload/pco-product_sheets/pco.edge_55_data_sheet.pdf)

    Args:
        photon_pixel_rate: Number of photons per pixel per second
        exposure_time: Integration time, s
        dnf: Deconvolution noise factor as specified in Agrawal et. al.
        camera_quantum_efficency: QE of camera, in [0,1]
        camera_dark_current: Dark current from datasheet, electrons / s
        pulse_time: Amount of time the illumination is pulsed, s
        camera_readout_noise: Readout noise from datasheet, electrons
        camera_max_counts: Maximum discrete bins of camera, usually 255 or 65535.

    Returns:
        The Estimated SNR.
    """

    # Call component function
    result = photonPixelRateToNoiseComponents(photon_pixel_rate, exposure_time,
                                              dnf=dnf,
                                              camera_quantum_efficency=camera_quantum_efficency,
                                              camera_dark_current=camera_dark_current,
                                              pulse_time=pulse_time,
                                              camera_readout_noise=camera_readout_noise,
                                              camera_ad_conversion=camera_ad_conversion,
                                              camera_max_counts=camera_max_counts, **kwargs)
    signal, noise_independent, noise_dependent = result

    if debug:
        print('DNF: %g, signal: %g, independent noise: %g, dependent noise: %g' % tuple([dnf] + list(result)))

    # Return SNR
    if dnf * noise_independent * noise_dependent == 0:
        return 0
    else:
        return signal / (dnf * np.sqrt(noise_independent ** 2 + noise_dependent ** 2))


def calcSnrFromDpcSourceList(source_list, exposure_time, per_led_illuminance=1000,
                             support=None, **system_params):
    """Calculate SNR given a source list."""

    # Generate total support
    pupil = comptic.imaging.pupil(yp.shape(source_list[0]), **system_params)
    total_support = yp.sum(pupil > 0)

    # Calculate the illuminance per pixel based on number of LEDs
    led_count = sum([np.sqrt(src[0] ** 2 + src[1] ** 2) <= system_params['objective_numerical_aperture'] for src in system_params['illumination_source_position_list_na']])
    total_illuminance = led_count * per_led_illuminance
    illuminance_per_pixel = total_illuminance / yp.sum(total_support)

    # Calculate illuminance per measurement
    illuminance_list = [illuminance_per_pixel * yp.sum(source) for source in source_list]

    # Calculate photon pixel rate
    photon_pixel_rate_list = [illuminanceToPhotonPixelRate(illuminance, **system_params) for illuminance in illuminance_list]
    photon_pixel_rate = yp.mean(photon_pixel_rate_list)

    # Calculate the DNF of a DPC inversion using these four patterns
    f = dnfFromSourceList(source_list, **system_params, support=support, eps=1e-5)

    # Calculate Expected SNR
    return photonPixelRateToSnr(photon_pixel_rate, exposure_time, dnf=f, **system_params)