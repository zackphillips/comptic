import copy

# Define system parameters for our system
_system_parameters = {
                        'default':{
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
                         }


def parameters(system_name='default', **kwargs):
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

    # Ensure system name is in parameter record
    assert system_name in _system_parameters, "%s not in system parameter record (available systems: %s)" % (system_name, str(_system_parameters))

    params = copy.deepcopy(_system_parameters[system_name])
    for key in kwargs:
        if key in params:
            params[key] = kwargs[key]

    return params
