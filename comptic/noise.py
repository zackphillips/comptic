import numpy as np
import llops as yp
from llops.decorators import real_valued_function, numpy_function
import copy

valid_noise_models = ['gaussian', 'poisson', 'saltandpepper', 'speckle']

@numpy_function
def add(signal, type='gaussian', **kwargs):
    """ Add noise to a measurement"""
    if type == 'gaussian':
        snr = kwargs.get('snr', 1.0)
        signal_mean = yp.abs(yp.mean(signal))
        noise_gaussian = np.random.normal(0, signal_mean / snr, yp.shape(signal))
        return signal + noise_gaussian

    elif type == 'poisson':
        noise_poisson = np.random.poisson(np.real(signal))
        return signal + noise_poisson

    elif type == 'saltpepper' or type == 'saltandpepper':
        salt_pepper_ratio = kwargs.get('ratio', 0.5)
        salt_pepper_saturation = kwargs.get('saturation', 0.004)
        output = yp.changeBackend(signal, 'numpy')

        # Salt mode
        num_salt = np.ceil(salt_pepper_saturation * signal.size * salt_pepper_ratio)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in signal.shape]
        output[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(salt_pepper_saturation * signal.size * (1. - salt_pepper_ratio))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in signal.shape]
        output[tuple(coords)] = 0
        return yp.cast_like(output, signal)

    elif type == 'speckle':
        noise_speckle = yp.randn(signal.shape)
        return signal + signal * noise_speckle


def snr(signal, noise_roi=None, signal_roi=None, debug=False):
    """ Calculate the imaging signal to noise ratio (SNR) of a signal """
    # Reference: https://en.wikipedia.org/wiki/Signal-to-noise_ratio_(imaging)

    # Calculate signal mean, using ROI if provided
    signal_mean = yp.mean(signal) if signal_roi is None else yp.mean(signal[signal_roi.slice])

    # Calculate noise standard deviation, using ROI if provided
    signal_std = yp.std(signal) if noise_roi is None else yp.std(signal[noise_roi.slice])

    if debug:
        print('Mean is %g, std is %g' % (yp.scalar(signal_mean), yp.scalar(signal_std)))

    return yp.scalar(signal_mean / signal_std)


def psnr(signal, noise_roi=None, signal_roi=None):
    """ Calculate the peak signal to noise ratio (SNR) of a signal """
    # Reference: https://en.wikipedia.org/wiki/Signal-to-noise_ratio

    # Calculate full power of signal
    signal_power = yp.sum(yp.abs(signal) ** 2) if signal_roi is None else yp.sum(yp.abs(signal[noise_roi.slice]) ** 2)

    # Calculate noise standard deviation, using ROI if provided
    signal_var = yp.std(signal) ** 2 if noise_roi is None else yp.std(signal[noise_roi.slice]) ** 2

    return yp.log10(signal_power / signal_var)


def cnr(signal, noise_roi=None, signal_roi=None):
    """ Calculate the imaging contrast to noise ratio (CNR) of an image """
    # Reference: https://en.wikipedia.org/wiki/Contrast-to-noise_ratio

    # Calculate signal mean, using ROI if provided
    signal_contrast = yp.abs(yp.max(signal) - yp.min(signal)) if signal_roi is None else yp.abs(yp.max(signal[noise_roi.slice]) - yp.min(signal[noise_roi.slice]))

    # Calculate noise standard deviation, using ROI if provided
    signal_std = yp.std(signal) if noise_roi is None else np.std(signal[noise_roi.slice])

    return (signal_contrast / signal_std)


def dnfFromConvolutionKernel(h):
    """Calculate the deconvolution noise factor (DNF) of N-dimensional
       convolution operator, given it's kernel."""
    if len(h) == 0:
        return 0
    else:
        # Normalize
        h = copy.deepcopy(h) / yp.scalar(yp.sum(h))

        # Take fourier transform intensity
        h_fft = yp.Ft(h)
        sigma_h = yp.abs(h_fft) ** 2

        # Calculate DNF
        return np.sqrt(1 / len(h) * np.sum(1 / sigma_h))


def calcCondNumFromKernel(x):
    if len(x) == 0:
        return 0
    else:
        # x = x / np.sum(x)
        x_fft = np.fft.fft(x)
        sigma_x = np.abs(x_fft)
        return np.max(sigma_x) / np.min(sigma_x)
