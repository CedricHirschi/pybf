"""
   Copyright (C) 2020 ETH Zurich. All rights reserved.

   Author: Sergei Vostrikov, ETH Zurich

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import numpy as np
from scipy.signal.windows import tukey, hamming, hann

def calc_fov_receive_apodization(num_of_elements, 
                                 elements_coords, 
                                 pixels_coords,
                                 alpha_fov_degree=45,
                                 channel_reduction=None):
    """
    Compute field-of-view (dynamic aperture) receive apodization weights for each imaging pixel.

    Parameters
    ----------
    num_of_elements : int
        Number of transducer elements.
    elements_coords : ndarray [2 x n_elements]
        Array of element coordinates (x, z) or similar.
    pixels_coords : ndarray [2 x n_pixels]
        Array of pixel coordinates (x, z).
    alpha_fov_degree : float
        Field-of-view aperture expansion angle in degrees.
    channel_reduction : int or None
        Optional number of channels to keep active (centered).

    Returns
    -------
    apod_weights : ndarray [n_pixels x n_elements]
        Apodization weights for each pixel and element.
    """

    n_elements = elements_coords.shape[1]
    n_pixels = pixels_coords.shape[1]
    elements_x = elements_coords[0, :]

    apod_weights = np.zeros((n_pixels, n_elements), dtype=np.float32)

    # Geometry-based aperture for each pixel
    tan_alpha = np.tan(np.radians(alpha_fov_degree / 2))
    delta_x = pixels_coords[1, :] * tan_alpha  # half-aperture width at given depth

    # Compute effective aperture bounds
    x_aperture_min = pixels_coords[0, :] - delta_x
    x_aperture_max = pixels_coords[0, :] + delta_x

    for n in range(n_pixels):
        # Determine active elements inside FOV aperture
        active_mask = np.logical_and(elements_x >= x_aperture_min[n],
                                     elements_x <= x_aperture_max[n])
        active_indices = np.where(active_mask)[0]
        Na = len(active_indices)

        if Na == 0:
            continue  # nothing active for this pixel

        # Choose window depending on number of active elements
        if Na <= 2:
            # Rectangular (all ones)
            win = np.ones(Na)
        elif Na == 3:
            # Gentle taper: [e, 1, e]
            e = 0.88
            win = np.array([e, 1.0, e])
        elif 4 <= Na <= 6:
            # Mild Tukey or Hamming, normalized to max=1
            win = tukey(Na, alpha=0.25)
            if win.max() != 0:
                win /= win.max()
        else:
            # Na >= 8 → use normal window (Hann by default)
            win = hann(Na)
            if win.max() != 0:
                win /= win.max()

        # Assign weights to active elements
        apod_weights[n, active_indices] = win.astype(np.float32)

    # Optional channel reduction (center mask)
    if channel_reduction is not None:
        channel_mask = np.zeros(n_elements, dtype=np.float32)
        start_i = int(np.ceil((n_elements - channel_reduction) / 2))
        stop_i = int(start_i + channel_reduction)
        channel_mask[start_i:stop_i] = 1
        apod_weights *= channel_mask

    return apod_weights.astype(np.float32)
