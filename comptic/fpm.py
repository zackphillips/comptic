import copy
import numpy as np
import llops as yp
import matplotlib.pyplot as plt
from .. import ledarray

# Define system parameters for our system
_system_params_default = {
                             'pixel_count': (2580, 2180),
                             'objective_numerical_aperture': 0.25,
                             'objective_magnification': 10.2,
                             'system_magnification': 1.0,
                             'camera_pixel_size': 6.5e-6,
                             'camera_is_color': False,
                             'illumination_wavelength': 0.53e-6,
                             'illumination_source_position_list_na': ledarray.getPositionsNa('quasi-dome')
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


def plotLedPattern(illumination_source_position_list_na,
                   led_intensity_list=None,
                   objective_numerical_aperture=None,
                   labels=None,
                   background_color='k',
                   figsize=(6, 6),
                   zoom=False,
                   **kwargs):
    """Plot list of source patterns.

    Parameters
    ----------
    led_intensity_list : list
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

    # Parse led_intensity_list
    if led_intensity_list is None:
        led_intensity_list = [[1.0, 1.0, 1.0]] * len(illumination_source_position_list_na)
    else:
        assert len(illumination_source_position_list_na) == len(led_intensity_list)

    # Convert source position list to numpy array
    illumination_source_position_list_na = np.asarray(illumination_source_position_list_na)

    # Generate figure
    plt.figure(figsize=figsize)

    plt.scatter(illumination_source_position_list_na[:, 1], illumination_source_position_list_na[:, 0], c=background_color)
    plt.scatter(illumination_source_position_list_na[:, 1], illumination_source_position_list_na[:, 0], c=led_intensity_list)
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
