"""
   Copyright (C) 2020 ETH Zurich. All rights reserved.

   Author: Sergei Vostrikov, ETH Zurich
        Wolfgang Boettcher, ETH Zurich

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

# Basic libraries
import numpy as np
from os.path import dirname, abspath
import sys
import time

# Import pybf modules
from pybf.pybf.io_interfaces import ImageSaver
from pybf.pybf.image_settings import ImageSettings
from pybf.pybf.delay_calc import calc_propagation_delays
from pybf.pybf.delay_calc import convert_time_to_samples
from pybf.pybf.apodization import calc_fov_receive_apodization
from pybf.pybf.signal_processing import demodulate_decimate
from pybf.pybf.signal_processing import interpolate_modulate
from pybf.pybf.signal_processing import filter_band_pass
from pybf.pybf.signal_processing import hilbert_interpolate

from pybf.pybf.bf_cores import delay_and_sum_numba, delay_and_sum_numpy
from pybf.scripts.visualize_image_dataset import visualize_image_dataset

# Constants
LATERAL_PIXEL_DENSITY_DEFAULT = 5
ALPHA_FOV_APOD_ANGLE_DEFAULT = 50
DB_RANGE_DEFAULT = 40
IMAGE_RESOLUTION_DEFAULT = [100, 100]

class BFCartesianRealTime():

    def __init__(self,
                 f_sampling,
                 tx_strategy,
                 transducer_obj,
                 decimation_factor,
                 interpolation_factor,
                 image_res,
                 img_config_obj,
                 db_range=DB_RANGE_DEFAULT,
                 start_time=None,
                 correction_time_shift=None,
                 alpha_fov_apod=ALPHA_FOV_APOD_ANGLE_DEFAULT,
                 bp_filter_params=None,
                 envelope_detector='I_Q',
                 picmus_dataset=False,
                 channel_reduction=None,
                 is_inherited=False,
                 use_coherence_factor=False,   # NEW
                 cf_gamma=1.5):                # NEW

        # Existing init code unchanged ------------------------------
        self._transducer = transducer_obj
        lateral_pixel_density = LATERAL_PIXEL_DENSITY_DEFAULT
        self._img_config = img_config_obj
        self._image_res = image_res
        self._pixels_coords = self._img_config.get_pixels_coords(image_res[0], image_res[1])

        print('Delays precalculation...')
        self._tx_strategy = tx_strategy
        self._rx_delays, self._tx_delays = calc_propagation_delays(self._tx_strategy,
                                                                   self._transducer.num_of_elements,
                                                                   self._transducer.elements_coords,
                                                                   self._pixels_coords,
                                                                   self._transducer.speed_of_sound,
                                                                   simulation_flag=picmus_dataset)

        self._f_sampling = f_sampling
        self._decimation_factor = decimation_factor
        self._interpolation_factor = interpolation_factor
        self._f_sampling_proc = self._f_sampling / self._decimation_factor * self._interpolation_factor

        self._start_time = 0 if start_time is None else start_time
        self._correction_time_shift = 0 if correction_time_shift is None else correction_time_shift

        self._rx_delays_samples = convert_time_to_samples(self._rx_delays,
                                                          self._f_sampling_proc,
                                                          self._start_time,
                                                          self._correction_time_shift)
        self._tx_delays_samples = convert_time_to_samples(self._tx_delays,
                                                          self._f_sampling_proc, 0, 0)

        if is_inherited is False:
            print('Apodization precalculation...')
            self._apod = calc_fov_receive_apodization(int(self._transducer.num_of_elements),
                                                      self._transducer.elements_coords,
                                                      self._pixels_coords,
                                                      alpha_fov_degree=alpha_fov_apod,
                                                      channel_reduction=channel_reduction)

        self._bp_filter_params = bp_filter_params
        self._envelope_detector = envelope_detector

        # NEW parameters
        self._use_coherence_factor = use_coherence_factor
        self._cf_gamma = cf_gamma
        # ------------------------------------------------------------

    # unchanged preprocessing function -------------------------------
    def _preprocess_data(self, rf_data):
        if self._bp_filter_params is not None:
            rf_data_filt = filter_band_pass(rf_data.astype(np.float32),
                                            self._f_sampling,
                                            self._bp_filter_params[0],
                                            self._bp_filter_params[1],
                                            self._bp_filter_params[2])
        else:
            rf_data_filt = rf_data

        if self._envelope_detector == 'I_Q':
            rf_data_IQ = demodulate_decimate(rf_data_filt,
                                             self._f_sampling,
                                             self._transducer.f_central_hz,
                                             self._decimation_factor)
            rf_data_proc = interpolate_modulate(rf_data_IQ,
                                                self._f_sampling / self._decimation_factor,
                                                self._transducer.f_central_hz,
                                                self._interpolation_factor)
        elif self._envelope_detector == 'hilbert':
            rf_data_proc = hilbert_interpolate(rf_data_filt, self._f_sampling, self._interpolation_factor)
        return rf_data_proc
    # ----------------------------------------------------------------

    def beamform(self, rf_data, numba_active=False):

        print('Beamforming...')
        print(' ')
        start_time = time.time()

        acqs_to_process = [x for x in range(self._tx_delays_samples.shape[0])]
        das_out = np.zeros((len(acqs_to_process), self._pixels_coords.shape[1]), dtype=np.complex128)

        if len(rf_data.shape) == 2 and len(acqs_to_process) == 1:
            rf_data_reshaped = rf_data.reshape((1, rf_data.shape[0], rf_data.shape[1]))
        elif len(rf_data.shape) == 3 and len(acqs_to_process) == rf_data.shape[0]:
            rf_data_reshaped = rf_data
        else:
            print('Input data shape ', rf_data.shape, ' is incorrect.')

        for i in acqs_to_process:

            rf_data_proc = self._preprocess_data(rf_data_reshaped[i, :, :])
            rf_data_proc_trans = np.transpose(rf_data_proc)

            delays_samples = self._rx_delays_samples + self._tx_delays_samples[i, :]

            # --- Normal DAS beamforming
            if numba_active:
                das_current = delay_and_sum_numba(rf_data_proc_trans,
                                                  delays_samples.reshape(1, delays_samples.shape[0], -1),
                                                  apod_weights=self._apod)
            else:
                das_current = delay_and_sum_numpy(rf_data_proc_trans,
                                                  delays_samples.reshape(1, delays_samples.shape[0], -1),
                                                  apod_weights=self._apod)

            # --- Optional Coherence Factor weighting ----------------
            if self._use_coherence_factor:
                # run the same DAS core but with |RF| instead of complex RF
                if numba_active:
                    das_abs = delay_and_sum_numba(np.abs(rf_data_proc_trans),
                                                  delays_samples.reshape(1, delays_samples.shape[0], -1),
                                                  apod_weights=self._apod)
                else:
                    das_abs = delay_and_sum_numpy(np.abs(rf_data_proc_trans),
                                                  delays_samples.reshape(1, delays_samples.shape[0], -1),
                                                  apod_weights=self._apod)

                cf = np.abs(das_current) / (das_abs + 1e-12)
                das_current *= cf ** self._cf_gamma
            # ---------------------------------------------------------

            das_out[i, :] = das_current

        das_out_compound = np.sum(das_out[acqs_to_process, :], axis=0)

        print('Time of execution: %s seconds' % (time.time() - start_time))
        return das_out_compound.reshape(self._image_res[1], self._image_res[0])
