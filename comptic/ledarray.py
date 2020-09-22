"""
Copyright 2018 Waller Lab
The University of California, Berkeley

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import os
import json

# Default image directory (relative path)
led_position_json_filename = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'resources/led_positions.json')

# Load image dictionary
with open(led_position_json_filename) as f:
    _led_positions_dict = json.load(f)


def getAvailableLedArrays():
    """Get list of available LED arrays."""
    return tuple(_led_positions_dict.keys())


def getPositionsNa(device_name):
    """Get positions in the format [na_x, na_y]."""
    if device_name in _led_positions_dict:
        led_positons = [(pos['x'], pos['y'], pos['z']) for pos in _led_positions_dict[device_name]]
        return cartToNa(led_positons)
    else:
        raise ValueError('%s is not a valid device name')


def getPositionsCart(device_name):
    """Get positions in the format [x, y, z]."""
    if device_name in _led_positions_dict:
        return [(pos['x'], pos['y'], pos['z']) for pos in _led_positions_dict[device_name]]
    else:
        raise ValueError('%s is not a valid device name')


def getBoardIndicies(device_name):
    """Get positions in the format [board_index]."""
    if device_name in _led_positions_dict:
        return [pos['board_index'] for pos in _led_positions_dict[device_name]]
    else:
        raise ValueError('%s is not a valid device name')


def getPositions(device_name):
    """Get positions in the format [index, x, y, z, board_index]."""
    if device_name in _led_positions_dict:
        return [(pos['index'], pos['x'], pos['y'], pos['z'], pos['board_index']) for pos in _led_positions_dict[device_name]]
    else:
        raise ValueError('%s is not a valid device name')


def cartToNa(point_list_cart, z_offset=0):
    """Function which converts a list of cartesian points to numerical aperture (NA)

    Args:cd 
        point_list_cart: List of (x,y,z) positions relative to the sample (origin)
        z_offset : Optional, offset of LED array in z, mm

    Returns:
        A 2D numpy array where the first dimension is the number of LEDs loaded and the second is (Na_x, NA_y)
    """
    point_list_cart = np.asarray(point_list_cart) if np.ndim(point_list_cart) == 2 else np.asarray([point_list_cart])
    yz = np.sqrt(point_list_cart[:, 1] ** 2 + (point_list_cart[:, 2] + z_offset) ** 2)
    xz = np.sqrt(point_list_cart[:, 0] ** 2 + (point_list_cart[:, 2] + z_offset) ** 2)

    result = np.zeros((np.size(point_list_cart, 0), 2))
    result[:, 0] = np.sin(np.arctan(point_list_cart[:, 0] / yz))
    result[:, 1] = np.sin(np.arctan(point_list_cart[:, 1] / xz))

    return(result)


def reloadLedPositionsFile():
    """Reload the LED positions .json file from the disk."""
    global _led_positions_dict
    with open(led_position_json_filename) as f:
        _led_positions_dict = json.load(f)
