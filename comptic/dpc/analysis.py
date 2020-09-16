import numpy as np
import copy
import matplotlib.pyplot as plt
import llops as yp
import comptic

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
    from pydpc import wotfsFromSourcePositionList
    Hr_list, Hi_list = wotfsFromSourcePositionList(shape, led_pattern_list, **system_params)

    # Calculate DNF and return
    return dnfFromWotfList(Hr_list, Hi_list, eps=eps, support=support)


def dnfFromSourceList(source_list, pupil=None,
                      eps=None, support=None, **system_params):
    """Calculate DNF from persecribed source pattern."""

    # Generate Hi and Hr
    from pydpc import wotfsFromSourceList
    Hr_list, Hi_list = wotfsFromSourceList(source_list, pupil=pupil, **system_params)

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
