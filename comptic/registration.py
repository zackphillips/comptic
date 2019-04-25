
import skimage
import skimage.feature
import numpy as np
import llops as yp
import llops.operators as ops
import llops.filter as filters
import scipy as sp

# Feature-based registration imports
from skimage.feature import ORB, match_descriptors, plot_matches
from skimage.measure import ransac
from skimage.transform import EuclideanTransform


def _preprocessForRegistration(image0, image1, methods=['pad'], **kwargs):

    # Ensure methods argument is a list
    if type(methods) not in (list, tuple):
        methods = [methods]

    # Perform 'reflection', which pads an object with antisymmetric copies of itself
    if 'pad' in methods:

        # Get pad factor
        pad_factor = kwargs.get('pad_factor', 2)

        # Get pad value
        pad_type = kwargs.get('pad_type', 'reflect')

        # Generate pad operator which pads the object to 2x it's size
        pad_size = [sp.fftpack.next_fast_len(int(pad_factor * s)) for s in yp.shape(image0)]

        # Perform padding
        image0 = yp.pad(image0, pad_size, pad_value=pad_type, center=True)
        image1 = yp.pad(image1, pad_size, pad_value=pad_type, center=True)

    # Normalize to the range [0,1] (Always do if we're filtering)
    if 'normalize' in methods:
        image0 = filters._normalize(image0)
        image1 = filters._normalize(image1)

    # Sobel filtering
    if 'sobel' in methods:
        image0 = filters.sobel(image0)
        image1 = filters.sobel(image1)

    # Gaussian filtering
    if 'gaussian' in methods:
        image0 = filters.gaussian(image0, sigma=1)
        image1 = filters.gaussian(image1, sigma=1)

    # High-pass filtering (using gaussian)
    if 'highpass' in methods:
        image0 = filters.gaussian(image0, sigma=2) - filters.gaussian(image0, sigma=4)
        image1 = filters.gaussian(image1, sigma=2) - filters.gaussian(image1, sigma=4)

    # Roberts filtering
    if 'roberts' in methods:
        image0 = filters.roberts(image0)
        image1 = filters.roberts(image1)

    # Scharr filtering
    if 'scharr' in methods:
        image0 = filters.scharr(image0)
        image1 = filters.scharr(image1)

    # Prewitt filtering
    if 'prewitt' in methods:
        image0 = filters.prewitt(image0)
        image1 = filters.prewitt(image1)

    # Canny filtering
    if 'canny' in methods:
        image0 = filters.canny(image0, sigma=kwargs.get('sigma', 1), low_threshold=kwargs.get('low_threshold', 0.01), high_threshold=kwargs.get('high_threshold', 0.05))
        image1 = filters.canny(image1, sigma=kwargs.get('sigma', 1), low_threshold=kwargs.get('low_threshold', 0.01), high_threshold=kwargs.get('high_threshold', 0.05))

    return image0, image1


def registerImage(image0, image1, method='xc', axis=None,
                  preprocess_methods=['reflect'], debug=False, **kwargs):

    # Perform preprocessing
    if len(preprocess_methods) > 0:
        image0, image1 = _preprocessForRegistration(image0, image1, preprocess_methods, **kwargs)

    # Parameter on whether we can trust our registration
    trust_ratio = 1.0

    if method in ['xc' or 'cross_correlation']:

        # Get energy ratio threshold
        trust_threshold = kwargs.get('energy_ratio_threshold', 1.5)

        # Pad arrays for optimal speed
        pad_size = tuple([sp.fftpack.next_fast_len(s) for s in yp.shape(image0)])

        # Perform padding
        if pad_size is not yp.shape(image0):
            image0 = yp.pad(image0, pad_size, pad_value='edge', center=True)
            image1 = yp.pad(image1, pad_size, pad_value='edge', center=True)

        # Take F.T. of measurements
        src_freq, target_freq = yp.Ft(image0, axes=axis), yp.Ft(image1, axes=axis)

        # Whole-pixel shift - Compute cross-correlation by an IFFT
        image_product = src_freq * yp.conj(target_freq)
        # image_product /= abs(src_freq * yp.conj(target_freq))
        cross_correlation = yp.iFt(image_product, center=False, axes=axis)

        # Take sum along axis if we're doing 1D
        if axis is not None:
            axis_to_sum = list(range(yp.ndim(image1)))
            del axis_to_sum[axis]
            cross_correlation = yp.sum(cross_correlation, axis=axis_to_sum)

        # Locate maximum
        shape = yp.shape(src_freq)
        maxima = yp.argmax(yp.abs(cross_correlation))
        midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

        shifts = np.array(maxima, dtype=np.float64)
        shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

        # If its only one row or column the shift along that dimension has no
        # effect. We set to zero.
        for dim in range(yp.ndim(src_freq)):
            if shape[dim] == 1:
                shifts[dim] = 0

        # If energy ratio is too small, set all shifts to zero
        trust_metric = yp.scalar(yp.max(yp.abs(cross_correlation) ** 2) / yp.mean(yp.abs(cross_correlation) ** 2))

        # Determine if this registraition can be trusted
        trust_ratio = trust_metric / trust_threshold

    elif method == 'orb':

        # Get user-defined mean_residual_threshold if given
        trust_threshold = kwargs.get('mean_residual_threshold', 40.0)

        # Get user-defined mean_residual_threshold if given
        orb_feature_threshold = kwargs.get('orb_feature_threshold', 25)

        match_count = 0
        fast_threshold = 0.05
        while match_count < orb_feature_threshold:
            descriptor_extractor = ORB(n_keypoints=500, fast_n=9,
                                       harris_k=0.1,
                                       fast_threshold=fast_threshold)

            # Extract keypoints from first frame
            descriptor_extractor.detect_and_extract(np.asarray(image0).astype(np.double))
            keypoints0 = descriptor_extractor.keypoints
            descriptors0 = descriptor_extractor.descriptors

            # Extract keypoints from second frame
            descriptor_extractor.detect_and_extract(np.asarray(image1).astype(np.double))
            keypoints1 = descriptor_extractor.keypoints
            descriptors1 = descriptor_extractor.descriptors

            # Set match count
            match_count = min(len(keypoints0), len(keypoints1))
            fast_threshold -= 0.01

            if fast_threshold == 0:
                raise RuntimeError('Could not find any keypoints (even after shrinking fast threshold).')

        # Match descriptors
        matches = match_descriptors(descriptors0, descriptors1, cross_check=True)

        # Filter descriptors to axes (if provided)
        if axis is not None:
            matches_filtered = []
            for (index_0, index_1) in matches:
                point_0 = keypoints0[index_0, :]
                point_1 = keypoints1[index_1, :]
                unit_vec = point_0 - point_1
                unit_vec /= np.linalg.norm(unit_vec)

                if yp.abs(unit_vec[axis]) > 0.99:
                    matches_filtered.append((index_0, index_1))

            matches_filtered = np.asarray(matches_filtered)
        else:
            matches_filtered = matches

        # Robustly estimate affine transform model with RANSAC
        model_robust, inliers = ransac((keypoints0[matches_filtered[:, 0]],
                                        keypoints1[matches_filtered[:, 1]]),
                                       EuclideanTransform, min_samples=3,
                                       residual_threshold=2, max_trials=100)

        # Note that model_robust has a translation property, but this doesn't
        # seem to be as numerically stable as simply averaging the difference
        # between the coordinates along the desired axis.

        # Apply match filter
        matches_filtered = matches_filtered[inliers, :]

        # Process keypoints
        if yp.shape(matches_filtered)[0] > 0:

            # Compute shifts
            difference = keypoints0[matches_filtered[:, 0]] - keypoints1[matches_filtered[:, 1]]
            shifts = (yp.sum(difference, axis=0) / yp.shape(difference)[0])
            shifts = np.round(shifts[0])

            # Filter to axis mask
            if axis is not None:
                _shifts = [0, 0]
                _shifts[axis] = shifts[axis]
                shifts = _shifts

            # Calculate residuals
            residuals = yp.sqrt(yp.sum(yp.abs(keypoints0[matches_filtered[:, 0]] + np.asarray(shifts) - keypoints1[matches_filtered[:, 1]]) ** 2))

            # Define a trust metric
            trust_metric = residuals / yp.shape(keypoints0[matches_filtered[:, 0]])[0]

            # Determine if this registration can be trusted
            trust_ratio = 1 / (trust_metric / trust_threshold)
            print('===')
            print(trust_ratio)
            print(trust_threshold)
            print(trust_metric)
            print(shifts)
        else:
            trust_metric = 1e10
            trust_ratio = 0.0
            shifts = np.asarray([0, 0])

    elif method == 'optimize':

        # Create Operators
        L2 = ops.L2Norm(yp.shape(image0), dtype='complex64')
        R = ops.PhaseRamp(yp.shape(image0), dtype='complex64')
        REAL = ops.RealFilter((2, 1), dtype='complex64')

        # Take Fourier Transforms of images
        image0_f, image1_f = yp.astype(yp.Ft(image0), 'complex64'), yp.astype(yp.Ft(image1), 'complex64')

        # Diagonalize one of the images
        D = ops.Diagonalize(image0_f)

        # Form objective
        objective = L2 * (D * R * REAL - image1_f)

        # Solve objective
        solver = ops.solvers.GradientDescent(objective)
        shifts = solver.solve(iteration_count=1000, step_size=1e-8)

        # Convert to numpy array, take real part, and round.
        shifts = yp.round(yp.real(yp.asbackend(shifts, 'numpy')))

        # Flip shift axes (x,y to y, x)
        shifts = np.fliplr(shifts)

        # TODO: Trust metric and trust_threshold
        trust_threshold = 1
        trust_ratio = 1.0

    else:
        raise ValueError('Invalid Registration Method %s' % method)

    # Mark whether or not this measurement is of good quality
    if not trust_ratio > 1:
        if debug:
            print('Ignoring shift with trust metric %g (threshold is %g)' % (trust_metric, trust_threshold))
        shifts = yp.zeros_like(np.asarray(shifts)).tolist()

    # Show debugging figures if requested
    if debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        plt.subplot(131)
        plt.imshow(yp.abs(image0))
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(yp.abs(image1))
        plt.title('Trust ratio: %g' % (trust_ratio))
        plt.axis('off')
        plt.subplot(133)
        if method in ['xc' or 'cross_correlation']:
            if axis is not None:
                plt.plot(yp.abs(yp.squeeze(cross_correlation)))
            else:
                plt.imshow(yp.abs(yp.fftshift(cross_correlation)))
        else:
            plot_matches(plt.gca(), yp.real(image0), yp.real(image1), keypoints0, keypoints1, matches_filtered)
        plt.title(str(shifts))
        plt.axis('off')

    # Return
    return shifts, trust_ratio


def register_roi_list(measurement_list, roi_list, axis=None,
                      use_overlap_region=True, debug=False,
                      preprocess_methods=['highpass', 'normalize'],
                      use_mean_offset=False, replace_untrusted=False,
                      tolerance=(200, 200), force_2d=False,
                      energy_ratio_threshold=1.5, method='xc'):
    """
    Register a list of overlapping ROIs
    """

    # Loop over frame indicies
    offsets = []
    trust_mask = []

    # Parse and set up axis definition
    if axis is not None and force_2d:
        _axis = None
    else:
        _axis = axis

    # Loop over frames
    rois_used = []
    for frame_index in range(len(measurement_list)):

        # Get ROIs
        roi_current = roi_list[frame_index]
        frame_current = measurement_list[frame_index]

        # Determine which rois overlap
        overlapping_rois = [(index, roi) for (index, roi) in enumerate(rois_used) if roi.overlaps(roi_current)]

        # Loop over overlapping ROIs
        if len(overlapping_rois) > 0:

            local_offset_list = []
            for index, overlap_roi in overlapping_rois:
                # Get overlap regions
                overlap_current, overlap_prev = yp.roi.getOverlapRegion((frame_current, measurement_list[index]),
                                                                        (roi_current, roi_list[index]))

                # Perform registration
                _local_offset, _trust_metric = registerImage(overlap_current,
                                                             overlap_prev,
                                                             axis=_axis,
                                                             method=method,
                                                             preprocess_methods=preprocess_methods,
                                                             pad_factor=1.5,
                                                             pad_type=0,
                                                             energy_ratio_threshold=energy_ratio_threshold,
                                                             sigma=0.1,
                                                             debug=False)

                # Deal with axis definitions
                if axis is not None and force_2d:
                    local_offset = [0] * len(_local_offset)
                    local_offset[axis] = _local_offset[axis]
                else:
                    local_offset = _local_offset

                # Filter to tolerance
                for ax in range(len(local_offset)):
                    if abs(local_offset[ax]) > tolerance[ax]:
                        local_offset[ax] = 0
                # local_offset = np.asarray([int(min(local_offset[i], tolerance[i])) for i in range(len(local_offset))])
                # local_offset = np.asarray([int(max(local_offset[i], -tolerance[i])) for i in range(len(local_offset))])

                # Append offset to list
                if _trust_metric > 1.0:
                    local_offset_list.append(local_offset)
                    if debug:
                        print('Registered with trust ratio %g' % _trust_metric)
                else:
                    if debug:
                        print('Did not register with trust ratio %g' % _trust_metric)

            # Append offset to list
            if len(local_offset_list) > 0:
                offsets.append(tuple((np.round(yp.mean(np.asarray(local_offset_list), axis=0)[0]).tolist())))
                trust_mask.append(True)
            else:
                offsets.append((0, 0))
                trust_mask.append(False)

        else:
            offsets.append((0, 0))
            trust_mask.append(True)

        # Store thir ROI in rois_used
        rois_used.append(roi_current)

    # Convert offsets to array and reverse diretion
    offsets = -1 * np.array(offsets)

    if not any(trust_mask):
        print('WARNING: Did not find any good registration values! Returning zero offset.')
        offsets = [np.asarray([0, 0])] * len(offsets)
    else:
        # Take mean of offsets if desired
        if use_mean_offset:
            # This flag sets all measurements to the mean of trusted registration
            offsets = np.asarray(offsets)
            trust_mask = np.asarray(trust_mask)
            offsets[:, 0] = np.mean(offsets[trust_mask, 0])
            offsets[:, 1] = np.mean(offsets[trust_mask, 1])
            offsets = offsets.tolist()
        elif replace_untrusted:
            # This flag replaces untrusted measurements with the mean of all trusted registrations
            offsets = np.asarray(offsets)
            trust_mask = np.asarray(trust_mask)
            trust_mask_inv = np.invert(trust_mask)
            offsets[trust_mask_inv, 0] = np.round(np.mean(offsets[trust_mask, 0]))
            offsets[trust_mask_inv, 1] = np.round(np.mean(offsets[trust_mask, 1]))
            offsets = offsets.tolist()

    # Convert to numpy array
    offsets = np.asarray(offsets)

    # Determine aggrigate offsets
    aggrigate_offsets = [offsets[0]]
    for offset_index in range(len(offsets) - 1):
        aggrigate_offsets.append(sum(offsets[slice(0, offset_index + 2)]).astype(np.int).tolist())

    # Return the recovered offsets
    return aggrigate_offsets


def register_translation(src_image, target_image, upsample_factor=1,
                         energy_ratio_threshold=2, space="real"):
    """
    Efficient subpixel image translation registration by cross-correlation.
    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.
    Parameters
    ----------
    src_image : ndarray
        Reference image.
    target_image : ndarray
        Image to register.  Must be same dimensionality as ``src_image``.
    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel.  Default is 1 (no upsampling)
    space : string, one of "real" or "fourier", optional
        Defines how the algorithm interprets input data.  "real" means data
        will be FFT'd to compute the correlation, while "fourier" data will
        bypass FFT of input data.  Case insensitive.
    return_error : bool, optional
        Returns error and phase difference if on,
        otherwise only shifts are returned
    Returns
    -------
    shifts : ndarray
        Shift vector (in pixels) required to register ``target_image`` with
        ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)
    error : float
        Translation invariant normalized RMS error between ``src_image`` and
        ``target_image``.
    phasediff : float
        Global phase difference between the two images (should be
        zero if images are non-negative).
    References
    ----------
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008). :DOI:`10.1364/OL.33.000156`
    .. [2] James R. Fienup, "Invariant error metrics for image reconstruction"
           Optics Letters 36, 8352-8357 (1997). :DOI:`10.1364/AO.36.008352`
    """
    # images must be the same shape
    if yp.shape(src_image) != yp.shape(target_image):
        raise ValueError("Error: images must be same size for "
                         "register_translation")

    # only 2D data makes sense right now
    if yp.ndim(src_image) > 3 and upsample_factor > 1:
        raise NotImplementedError("Error: register_translation only supports "
                                  "subpixel registration for 2D and 3D images")

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        src_freq = yp.Ft(src_image)
        target_freq = yp.Ft(target_image)
    else:
        raise ValueError("Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument.")

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = yp.shape(src_freq)
    image_product = src_freq * yp.conj(target_freq)
    cross_correlation = yp.iFt(image_product, center=False)

    # Locate maximum
    maxima = yp.argmax(yp.abs(cross_correlation))
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    # if upsample_factor > 1:
    #     # Initial shift estimate in upsampled grid
    #     shifts = np.round(shifts * upsample_factor) / upsample_factor
    #     upsampled_region_size = np.ceil(upsample_factor * 1.5)
    #     # Center of output array at dftshift + 1
    #     dftshift = np.fix(upsampled_region_size / 2.0)
    #     upsample_factor = np.array(upsample_factor, dtype=np.float64)
    #     normalization = (src_freq.size * upsample_factor ** 2)
    #     # Matrix multiply DFT around the current shift estimate
    #     sample_region_offset = dftshift - shifts*upsample_factor
    #     cross_correlation = _upsampled_dft(image_product.conj(),
    #                                        upsampled_region_size,
    #                                        upsample_factor,
    #                                        sample_region_offset).conj()
    #     cross_correlation /= normalization
    #     # Locate maximum and map back to original pixel grid
    #     maxima = np.array(np.unravel_index(
    #                           np.argmax(np.abs(cross_correlation)),
    #                           cross_correlation.shape),
    #                       dtype=np.float64)
    #     maxima -= dftshift
    #
    #     shifts = shifts + maxima / upsample_factor

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(yp.ndim(src_freq)):
        if shape[dim] == 1:
            shifts[dim] = 0

    # If energy ratio is too small, set all shifts to zero
    energy_ratio = yp.max(yp.abs(cross_correlation) ** 2) / yp.sum(yp.abs(cross_correlation) ** 2) * yp.prod(yp.shape(cross_correlation))
    if energy_ratio < energy_ratio_threshold:
        print('Ignoring shift with energy ratio %g (threshold is %g)' % (energy_ratio, energy_ratio_threshold))
        shifts = yp.zeros_like(shifts)

    return shifts
