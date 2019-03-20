import numpy as np
import llops as yp


def genBayerCouplingMatrix(image_stack_rgb, pixel_offsets=(0, 0)):
    """Generate bayer coupling matrix from measurements."""
    bayer_coupling_matrix = np.zeros((4, 3), dtype=image_stack_rgb[0].dtype)

    for color_index, frame in enumerate(image_stack_rgb):
        bayer_coupling_matrix[0, color_index] = np.mean(frame[pixel_offsets[0]::2, pixel_offsets[1]::2])
        bayer_coupling_matrix[1, color_index] = np.mean(frame[pixel_offsets[0]::2, pixel_offsets[1] + 1::2])
        bayer_coupling_matrix[2, color_index] = np.mean(frame[pixel_offsets[0] + 1::2, pixel_offsets[1]::2])
        bayer_coupling_matrix[3, color_index] = np.mean(frame[pixel_offsets[0] + 1::2, pixel_offsets[1] + 1::2])

    return(bayer_coupling_matrix)


def demosaic(frame,
             order='grbg',
             bayer_coupling_matrix=None,
             debug=False,
             white_balance=False):

    # bayer_coupling_matrix = None
    # bgrg: cells very green
    # rggb: slight gteen tint

    """Demosaic a frame"""
    frame_out = yp.zeros((int(yp.shape(frame)[0] / 2), int(yp.shape(frame)[1] / 2), 3), yp.getDatatype(frame), yp.getBackend(frame))

    if bayer_coupling_matrix is not None:
        frame_vec = yp.zeros((4, int(yp.shape(frame)[0] * yp.shape(frame)[1] / 4)), yp.getDatatype(frame), yp.getBackend(frame))

        # Cast bayer coupling matrix
        bayer_coupling_matrix = yp.cast(bayer_coupling_matrix,
                                        yp.getDatatype(frame),
                                        yp.getBackend(frame))

        # Define frame vector
        for bayer_pattern_index in range(4):
            pixel_offsets = (0, 0)
            if bayer_pattern_index is 3:
                img_sub = frame[pixel_offsets[0]::2, pixel_offsets[1]::2]
            elif bayer_pattern_index is 1:
                img_sub = frame[pixel_offsets[0]::2, pixel_offsets[1] + 1::2]
            elif bayer_pattern_index is 2:
                img_sub = frame[pixel_offsets[0] + 1::2, pixel_offsets[1]::2]
            elif bayer_pattern_index is 0:
                img_sub = frame[pixel_offsets[0] + 1::2, pixel_offsets[1] + 1::2]
            frame_vec[bayer_pattern_index, :] = yp.dcopy(yp.vec(img_sub))
            if debug:
                print("Channel %d mean is %g" % (bayer_pattern_index, yp.scalar(yp.real(yp.sum(img_sub)))))

        # Perform demosaic using least squares
        result = yp.linalg.lstsq(bayer_coupling_matrix, frame_vec)

        result -= yp.amin(result)
        result /= yp.amax(result)
        for channel in range(3):
            values = result[channel]
            frame_out[:, :, channel] = yp.reshape(values, ((yp.shape(frame_out)[0], yp.shape(frame_out)[1])))
            if white_balance:
                frame_out[:, :, channel] -= yp.amin(frame_out[:, :, channel])
                frame_out[:, :, channel] /= yp.amax(frame_out[:, :, channel])
        return frame_out
    else:
        frame_out = yp.zeros((int(yp.shape(frame)[0] / 2), int(yp.shape(frame)[1] / 2), 3),
                             dtype=yp.getDatatype(frame), backend=yp.getBackend(frame))

        # Get color order from order variable
        b_index = order.find('b')
        r_index = order.find('r')
        g1_index = order.find('g')

        # Get g2 from intersection of sets
        g2_index = set(list(range(4))).difference({b_index, r_index, g1_index}).pop()
        #  +-----+-----+
        #  |  0  |  1  |
        #  +-----+-----|
        #  |  2  |  3  |
        #  +-----+-----|

        if debug:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(frame[:12, :12])

        r_start = (int(r_index in [2, 3]), int(r_index in [1, 3]))
        g1_start = (int(g1_index in [2, 3]), int(g1_index in [1, 3]))
        g2_start = (int(g2_index in [2, 3]), int(g2_index in [1, 3]))
        b_start = (int(b_index in [2, 3]), int(b_index in [1, 3]))

        frame_out[:, :, 0] = frame[r_start[0]::2, r_start[1]::2]
        frame_out[:, :, 1] = (frame[g1_start[0]::2, g1_start[1]::2] + frame[g2_start[0]::2, g2_start[1]::2]) / 2.0
        frame_out[:, :, 2] = frame[b_start[0]::2, b_start[1]::2]

        # normalize
        frame_out /= yp.max(frame_out)

        # Perform white balancing if desired
        if white_balance:
            clims = []
            for channel in range(3):
                clims.append(yp.max(frame_out[:, :, channel]))
                frame_out[:, :, channel] /= yp.max(frame_out[:, :, channel])

        # Return frame
        return frame_out


def bayerCouplingMatrix(camera_label):
    """Returns bayer coupling matrix for predefined """
    if camera_label == 'pco':
        # PCO Edge 5.5
        # return [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]]
        return [[1.00000000, 0.02134979, 0.04296954],
                [0.28029678, 0.09363176, 0.10655828],
                [0.26754523, 0.09336441, 0.11066270],
                [0.26399771, 0.03291212, 0.74810859]]
    elif camera_label == 'optimos':
        # QImaging Optimos
        return [[0.21998870, 0.28237841, 0.26123319],
                [0.12497684, 0.07170586, 0.60060606],
                [1.00000000, 0.03582582, 0.02325407],
                [0.23741785, 0.27927917, 0.25053755]]
    else:
        raise ValueError('Invalid camera label %s' % camera_label)
