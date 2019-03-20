
__all__ = ['wotfs', 'wotfsFromNa']

import numpy as np
import llops as yp
from llops.fft import Ft, iFt
import llops.operators as ops
import operator

# Generate Hu and Hp
def wotf(source, pupil, illumination_wavelength, **kwargs):
    """Function which generates the Weak-Object Transfer Functions (WOTFs) for absorption (Hu) and phase (Hp) given a source and pupul list

    Args:
        source_list: 3D ndarray of sources in the Fourier domain, where first dimensions are k_x, k_y and third is the number of transfer functions to generate
        pupil_list: 3D ndarray of pupil functions in the Fourier domain, where first dimensions are k_x, k_y and third is the number of transfer functions to generate
        lambda_list: list of wavelengths corresponding to the third dimension in sourceList and pupilList
        shifted_output: Whether to perform a fftshift on the output before returning

    Returns:
        A 2D numpy array where the first dimension is the number of LEDs loaded and the second is (Na_x, NA_y)
    """

    DC = yp.sum((np.abs(pupil) ** 2 * source))

    M = yp.Ft(source * pupil) * yp.conj(yp.Ft(pupil))

    Hu = 2 * yp.iFt(np.real(M)) / DC
    Hp = 1j * 2 * yp.iFt(1j * np.imag(M)) / DC

    # Cast both as complex
    Hu = yp.astype(Hu, 'complex32')
    Hp = yp.astype(Hp, 'complex32')

    # Return
    return (Hu, Hp)


def wotfsFromNa(image_size, led_pattern_list_na, pixel_size_eff, system_na, wavelength, dtype=np.complex64):
    """Function which generates WOTF given source coordinates in NA

    Args:
        image_size: The size of the WOTF to calculate
        led_pattern_list_na: List of source positions [NA_x, NA_y] to use
        pixel_size_eff: Effective pixel size of the system
        system_na: NA of the system
        wavelength: Wavelength of the system
    Returns:
        Hu_list: A list of amplitude transfer functions for illuminating with each LED.
        Hp_list: A list of phase transfer functions for illuminating with each LED.
    """

    def getMinIdx(myList):
        min_index, min_value = min(enumerate(myList), key=operator.itemgetter(1))
        return(min_index)

    M = image_size[0]
    N = image_size[1]
    led_pattern_list_na = np.asarray(led_pattern_list_na)

    # Generate spatial frequency coordinates and support
    dkx = 1 / (N * pixel_size_eff)
    dky = 1 / (M * pixel_size_eff)
    kyy, kxx = yp.grid(image_size, (dky, dkx))

    # Generate pupil
    pupil = np.sqrt(kxx * kxx + kyy * kyy) < (system_na / wavelength)

    # Generate WOTF
    Hu_list = []
    Hp_list = []
    source_list = []
    for index, led_pattern in enumerate(led_pattern_list_na):

        # Put points in source
        source = np.zeros(image_size, dtype=dtype)
        for led_position in led_pattern:
            # Find nearest index of this NA coordinate
            coordinate = getMinIdx((led_position[0] / wavelength - kxx.reshape(-1))
                                   ** 2 + (led_position[1] / wavelength - kyy.reshape(-1)) ** 2)
            ind = np.unravel_index(coordinate, image_size)
            source[ind[0], ind[1]] = 1

        # Now that we have the source, generate the corresponding Hu and Hp transfer functions for this single LED
        # Hu, Hp = genWotfs(np.asarray(source), np.asarray(pupil), wavelength)
        Hu, Hp = wotfs(np.asarray(source), np.asarray(pupil), wavelength)
        Hu_list.append(Hu)
        Hp_list.append(Hp)
        source_list.append(source)

    return (Hu_list, Hp_list, source_list, pupil)
