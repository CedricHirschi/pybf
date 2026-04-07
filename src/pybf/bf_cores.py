"""
Copyright (C) 2020 ETH Zurich. All rights reserved.

Author: Sergei Vostrikov, ETH Zurich
        Cedric Hirschi, ETH Zurich

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
from numba import jit, prange

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import warp as wp


# Initialize Warp
wp.init()


# Perform delay and sum operation with numba
# Input: rf_data_in of shape (n_samples x n_elements)
# delays_idx of shape (n_modes x n_elements x n_points)
# apod_weights of shape (n_points x n_elements)
@jit(nopython=True, parallel=True, nogil=True)
def delay_and_sum_numba(rf_data_in, delays_idx, apod_weights=None):
    n_elements = rf_data_in.shape[1]
    n_modes = delays_idx.shape[0]
    n_points = delays_idx.shape[2]

    # Allocate array
    das_out = np.zeros((n_modes, n_points), dtype=np.complex64)

    # Iterate over modes, points, elements
    for i in range(n_modes):
        for j in prange(n_points):  # type: ignore
            for k in range(n_elements):
                if delays_idx[i, k, j] <= rf_data_in.shape[0] - 1:
                    if apod_weights is None:
                        das_out[i, j] += rf_data_in[delays_idx[i, k, j], k]
                    else:
                        das_out[i, j] += (
                            rf_data_in[delays_idx[i, k, j], k] * apod_weights[j, k]
                        )

    return das_out


# Perform delay and sum operation with numpy
# Input: rf_data_in of shape (n_samples x n_elements)
# delays_idx of shape (n_modes x n_elements x n_points)
def delay_and_sum_numpy(rf_data_in, delays_idx, apod_weights=None):
    n_elements = rf_data_in.shape[1]
    n_modes = delays_idx.shape[0]
    n_points = delays_idx.shape[2]

    # Add one zero sample for data array (in the end)
    rf_data_shape = rf_data_in.shape
    rf_data = np.zeros((rf_data_shape[0] + 1, rf_data_shape[1]), dtype=np.complex64)
    rf_data[: rf_data_shape[0], : rf_data_shape[1]] = rf_data_in

    # If delay index exceeds the input data array dimensions,
    # write -1 (it will point to 0 element)
    delays_idx[delays_idx >= rf_data_shape[0] - 1] = -1

    # Choose the right samples for each channel and point
    # using numpy fancy indexing

    # Create array for fancy indexing of channels
    # of size (n_modes x n_points x n_elements)
    # The last two dimensions are transposed to fit the rf_data format
    fancy_idx_channels = np.arange(0, n_elements)
    fancy_idx_channels = np.tile(fancy_idx_channels, (n_modes, n_points, 1))

    # Create array for fancy indexing of samples
    # of size (n_modes x n_points x n_elements)
    # The last two dimensions are transposed to fit the rf_data format
    fancy_idx_samples = np.transpose(delays_idx, axes=[0, 2, 1])

    # Make the delay and sum operation by selecting the samples
    # using fancy indexing,
    # multiplying by apodization weights (optional)
    # and then summing them up along the last axis

    if apod_weights is None:
        das_out = np.sum(rf_data[fancy_idx_samples, fancy_idx_channels], axis=-1)
    else:
        das_out = np.sum(
            np.multiply(rf_data[fancy_idx_samples, fancy_idx_channels], apod_weights),
            axis=-1,
        )

    # Output shape: (n_modes x n_points)
    return das_out


if HAS_CUPY:

    def delay_and_sum_cupy(rf_data_in, delays_idx, apod_weights=None):
        """
        GPU-accelerated Delay-and-Sum using CuPy.

        Parameters
        ----------
        rf_data_in : array_like, shape (n_samples, n_elements)
            Input RF data on the host.
        delays_idx : array_like of ints, shape (n_modes, n_elements, n_points)
            Delay indices per mode/element/point.
        apod_weights : array_like, shape (n_points, n_elements), optional
            Apodization weights.

        Returns
        -------
        out : ndarray, shape (n_modes, n_points)
            Beamformed output on the host.
        """
        # Move inputs to GPU
        rf = cp.asarray(rf_data_in, dtype=cp.complex64)
        delays = cp.asarray(delays_idx, dtype=cp.int32)

        # Dimensions
        n_samples, n_elements = rf.shape
        n_modes, _, n_points = delays.shape

        # 1) Pad one zero row
        rf = cp.vstack(
            [rf, cp.zeros((1, n_elements), dtype=rf.dtype)]
        )  # → (n_samples+1, n_elements)

        # 2) Clip out-of-bounds delays to point at that last zero row
        safe_idx = cp.clip(delays, 0, n_samples)  # (n_modes, n_elements, n_points)

        # 3) Broadcast RF array to (n_modes, n_samples+1, n_elements)
        arr = cp.broadcast_to(rf[None, :, :], (n_modes, rf.shape[0], n_elements))

        # 4) Transpose indices to (n_modes, n_points, n_elements)
        idx = safe_idx.transpose(0, 2, 1)

        # 5) Gather delayed samples: (n_modes, n_points, n_elements)
        samples = cp.take_along_axis(arr, idx, axis=1)

        # 6) Apply apodization if provided
        if apod_weights is not None:
            # Move and align weights: (1, n_points, n_elements)
            w = cp.asarray(apod_weights, dtype=samples.dtype)[None, :, :]
            samples = samples * w

        # 7) Sum across elements → (n_modes, n_points)
        out_gpu = samples.sum(axis=2)

        # 8) Bring result back to host
        return cp.asnumpy(out_gpu)


# Warp kernel for Delay-and-Sum (DAS)
@wp.kernel
def das_warp_kernel(
    rf_real: wp.array2d(dtype=float),  # (n_samples, n_elements) # type: ignore
    rf_imag: wp.array2d(dtype=float),  # (n_samples, n_elements) # type: ignore
    delays: wp.array3d(dtype=int),  # (n_modes, n_elements, n_points) # type: ignore
    apo: wp.array2d(dtype=float),  # (n_points, n_elements) # type: ignore
    out_real: wp.array2d(dtype=float),  # (n_modes, n_points) # type: ignore
    out_imag: wp.array2d(dtype=float),  # (n_modes, n_points) # type: ignore
    n_samples: int,
    n_elements: int,
    use_apo: bool,
):
    # Get 2D thread indices
    mode, pt = wp.tid()  # type: ignore

    acc_r = float(0.0)
    acc_i = float(0.0)

    # Sum over elements
    for ele in range(n_elements):
        idx = delays[mode, ele, pt]
        if 0 <= idx < n_samples:
            r = rf_real[idx, ele]
            i = rf_imag[idx, ele]
            # Use explicit if since warp kernels disallow ternary expressions
            w = 1.0
            if use_apo:
                w = apo[pt, ele]
            acc_r += r * w
            acc_i += i * w

    out_real[mode, pt] = acc_r
    out_imag[mode, pt] = acc_i


def delay_and_sum_warp(rf_data_in, delays_idx, apod_weights=None):
    """
    Warp-accelerated DAS.

    rf_data_in : (n_samples, n_elements) complex64 on host
    delays_idx : (n_modes, n_elements, n_points) int32 on host
    apod_weights: (n_points, n_elements) float32 on host or None
    """
    # --- host->device
    rf_real = wp.array(rf_data_in.real.astype(np.float32))
    rf_imag = wp.array(rf_data_in.imag.astype(np.float32))
    delays = wp.array(delays_idx.astype(np.int32))

    n_modes, n_elements, n_points = delays_idx.shape
    n_samples, _ = rf_data_in.shape

    if apod_weights is None:
        apo = wp.zeros((n_points, n_elements), dtype=float)
        use_apo = False
    else:
        apo = wp.array(apod_weights.astype(np.float32))
        use_apo = True

    # allocate outputs on device
    out_r = wp.zeros((n_modes, n_points), dtype=float)
    out_i = wp.zeros((n_modes, n_points), dtype=float)

    # launch 2D grid: (modes x points)
    wp.launch(
        kernel=das_warp_kernel,
        dim=(n_modes, n_points, 1),
        inputs=[
            rf_real,
            rf_imag,
            delays,
            apo,
            out_r,
            out_i,
            n_samples,
            n_elements,
            use_apo,
        ],
    )
    wp.synchronize()

    # device->host and recombine
    real = out_r.numpy().astype(np.float32)
    imag = out_i.numpy().astype(np.float32)
    return (real + 1j * imag).astype(np.complex64)
