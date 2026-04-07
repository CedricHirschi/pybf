"""
Copyright (C) 2025 ETH Zurich. All rights reserved.

Authors:
    - Sergei Vostrikov, ETH Zurich
    - Cedric Hirschi, ETH Zurich

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

import logging
import warnings
from typing import Optional

import h5py
import numpy as np

from .transducer import Transducer
from .hardware import Hardware


def get_dataset(h5_node: h5py.File | h5py.Group, path: str) -> h5py.Dataset:
    assert path in h5_node, f"'{path}' is not found in the dataset"
    node = h5_node[path]
    assert isinstance(node, h5py.Dataset), f"'{path}' should be a dataset"
    return node


def get_group(h5_node: h5py.File | h5py.Group, path: str) -> h5py.Group:
    assert path in h5_node, f"'{path}' is not found in the dataset"
    node = h5_node[path]
    assert isinstance(node, h5py.Group), f"'{path}' should be a group"
    return node


def get_array(h5_node: h5py.File | h5py.Group, path: str) -> np.ndarray:
    return np.asarray(get_dataset(h5_node, path)[()])


def get_scalar(h5_node: h5py.File | h5py.Group, path: str):
    value = get_dataset(h5_node, path)[()]
    if isinstance(value, np.ndarray):
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


class DataLoader:
    def __init__(self, path_to_dataset: str):
        # Load hdf5 dataset
        self._file = h5py.File(path_to_dataset, "r")

        # Calculate number of frames
        self._rf_data = get_group(self._file, "data/rf_data")
        self._num_of_frames = len(list(self._rf_data.keys()))

        # Calculate number of acquisitions per frame
        self._frame_1 = get_group(self._rf_data, "frame_1")
        self._num_of_acq_per_frame = len(list(self._frame_1.keys()))

        # Get real sampling of the data
        self._f_sampling = get_scalar(self._file, "data/f_sampling")

        # Get frames per second
        self._fps = get_scalar(self._file, "data/fps")

        # Check the type of the dataset: experimental or simulation
        # If sim_params group is empty then it is experimental data
        if "sim_params" in list(self._file.keys()):
            self._simulation_flag = True
        else:
            self._simulation_flag = False

        # Create Transducer object
        self._create_transducer_obj()

        # Create USHardware object
        self._create_hardware_obj()

    def close_file(self) -> None:
        self._file.close()

    def _create_transducer_obj(self) -> None:

        # Calculate bandwidth separately
        bw_hz = get_array(self._file, "trans_params/bandwidth") * get_scalar(
            self._file, "trans_params/f_central"
        )

        self._transducer = Transducer(
            num_of_x_elements=get_scalar(self._file, "trans_params/x_num_of_elements"),
            num_of_y_elements=get_scalar(self._file, "trans_params/y_num_of_elements"),
            x_pitch=get_scalar(self._file, "trans_params/x_pitch"),
            y_pitch=get_scalar(self._file, "trans_params/y_pitch"),
            x_width=get_scalar(self._file, "trans_params/x_width"),
            y_width=get_scalar(self._file, "trans_params/y_width"),
            f_central_hz=get_scalar(self._file, "trans_params/f_central"),
            bandwidth_hz=bw_hz,
        )

    def _create_hardware_obj(self) -> None:

        if self._simulation_flag:
            str_temp = "sim_params/"

            self._hardware = Hardware(
                f_sampling_hz=get_scalar(self._file, str_temp + "f_sim_hz"),
                excitation=get_array(self._file, str_temp + "excitation"),
                impulse_response=get_array(
                    self._file, str_temp + "electroacoustic_impulse_response"
                ),
                start_time_s=get_scalar(self._file, str_temp + "start_time"),
            )

        else:
            str_temp = "hardware_params/"

            self._hardware = Hardware(
                f_sampling_hz=get_scalar(self._file, str_temp + "f_sampling_hz"),
                start_time_s=get_scalar(self._file, str_temp + "start_time"),
                correction_time_shift_s=get_scalar(
                    self._file, str_temp + "correction_time"
                ),
            )

    # Get the RF data for the mth acquisition of nth frame
    def get_rf_data(self, n_frame, m_acq) -> np.ndarray:

        if n_frame >= self._num_of_frames or n_frame < 0:
            raise ValueError(
                f"{n_frame = } outside the range of available frames (0 - {self._num_of_frames - 1})"
            )

        if m_acq >= self._num_of_acq_per_frame or m_acq < 0:
            raise ValueError(
                f"{m_acq = } outside the range of available acquisitions per frame (0 - {self._num_of_acq_per_frame - 1})"
            )

        # Create a path to shot (in rf dataset m_acq starts from 1 (due to Matlab),
        # but in Python first acquisition correcponds to shot_1)
        # The same works for Frame number
        shot_path = (
            "data/rf_data/" + "frame_" + str(n_frame + 1) + "/shot_" + str(m_acq + 1)
        )

        # Check if all elements of the transducer were active or not
        shot_ds = get_dataset(self._file, shot_path)
        if self._transducer._active_elements is None:
            rf_data = shot_ds[()]
        else:
            # Select some channels
            rf_data = shot_ds[()][self.transducer._active_elements, :]

        return rf_data.astype(np.float32)

    # Get positions of the scatters
    def get_scatters_pos(self) -> np.ndarray:

        if self._simulation_flag:
            return get_array(self._file, "sim_params/scatters_data")
        else:
            raise ValueError(
                "Scatters positions are available only in simulation mode."
            )

    # Get TX strategy
    @property
    def tx_strategy(self) -> tuple[str, np.ndarray]:

        temp_path = "data/tx_mode"
        tx_group = get_group(self._file, temp_path)

        # Get the TX strategy name
        tx_strat_str = list(tx_group.keys())[0]

        # Get the TX strategy params
        params = get_array(self._file, temp_path + "/" + tx_strat_str)

        return (tx_strat_str, params)

    # Returns number of frames
    @property
    def num_of_frames(self) -> int:

        return self._num_of_frames

    # Returns number of acquisitions per frame
    @property
    def num_of_acq_per_frame(self) -> int:

        return self._num_of_acq_per_frame

    # Returns real sampling rate of rf data
    @property
    def f_sampling(self) -> float:

        return self._f_sampling

    # Returns fps that defines time delay between frames
    @property
    def fps(self) -> float:

        return self._fps

    # Returns an instance of transducer object
    @property
    def transducer(self) -> Transducer:

        return self._transducer

    # Returns an instance of hardware object
    @property
    def hardware(self) -> Hardware:

        return self._hardware

    # Simulation flag
    @property
    def simulation_flag(self) -> bool:

        return self._simulation_flag


# Class to save beamformed images
class ImageSaver:
    def __init__(self, path_to_dataset: str):
        # Read/write if exists, create otherwise (default)
        self._file = h5py.File(path_to_dataset, "w")
        self.close_file()

        self._file = h5py.File(path_to_dataset, "a")

        self.data_subgroup = self._file.create_group("beamformed_data")
        self.params_subgroup = self._file.create_group("params")

    def close_file(self) -> None:
        self._file.close()

    # Save the image data in a dataset according to the format
    # imgs_data has shape (n_images x n_x_points x n_y_points)
    def save_low_res_images(
        self,
        imgs_data: np.ndarray,
        frame_number: int,
        low_res_imgs_indices: Optional[list[int]] = None,
    ) -> None:
        name = "/beamformed_data/frame_" + str(frame_number)
        group = self._file.require_group(name)

        # save low resolution images
        # If the list of indices was provided then use it to name datasets
        if low_res_imgs_indices is None:
            low_res_imgs_indices = [i for i in range(imgs_data.shape[0])]
        else:
            if len(low_res_imgs_indices) != imgs_data.shape[0]:
                warnings.warn(
                    f"length of indices list ({len(low_res_imgs_indices)}) is not equal to data shape ({imgs_data.shape[0]})"
                )

        for m_shot in low_res_imgs_indices:
            _ = group.create_dataset(
                "low_res_image_" + str(m_shot), data=imgs_data[m_shot, :]
            )

    # Save the image data in a dataset according to the format
    # img_data has shape (n_x_points x n_y_points)
    def save_high_res_image(self, img_data: np.ndarray, frame_number: int) -> None:
        name = "/beamformed_data/frame_" + str(frame_number)
        group = self._file.require_group(name)

        # save high resolution images
        _ = group.create_dataset("high_res_image", data=img_data)

    # Save the image params in a dataset according to the format
    def save_params(
        self,
        pixels_coords: np.ndarray,
        image_size: np.ndarray,
        elements_coords: np.ndarray,
        fps: float,
    ) -> None:

        # Save pixels coords
        _ = self.params_subgroup.create_dataset(
            "pixels_coords_x_z", pixels_coords.shape, data=pixels_coords
        )

        # Save image resolutions
        _ = self.params_subgroup.create_dataset(
            "image_resolution", image_size.shape, data=image_size
        )

        # Save elements coords
        _ = self.params_subgroup.create_dataset(
            "elements_coords", elements_coords.shape, data=elements_coords
        )

        # Save fps
        _ = self.params_subgroup.create_dataset("fps", data=fps)

    # Save the simulation params in a dataset according to the format
    def save_simulation_params(self, scatters_coords: np.ndarray) -> None:

        # Create subgroup
        group = self._file.require_group("sim_params")

        # save pixels coords
        _ = group.create_dataset(
            "scatters_data", scatters_coords.shape, data=scatters_coords
        )


# Class to save beamformed images
class ImageLoader:
    def __init__(self, path_to_dataset: str):
        self.log = logging.getLogger(__name__)

        # Open for read
        self._file = h5py.File(path_to_dataset, "r")

        # Create data subgroup
        self._data_subgroup = get_group(self._file, "/beamformed_data")

        # Calculate number of frames
        frame_names_list = list(self._data_subgroup.keys())
        self._num_of_frames = len(frame_names_list)
        self.log.debug(f"number of available frames: {self._num_of_frames}")

        # Calculate indices of existing frames
        self._frames_indices = [
            int(filename.split("_")[-1]) for filename in frame_names_list
        ]

        self.log.debug(
            f"Available frame keys: {list(get_group(self._file, '/beamformed_data').keys())}"
        )

        # Calculate number of low resolution images per frame
        # Each folder containts low res images + 1 high resolution image
        lri_names_list = list(get_group(self._file, "/beamformed_data/frame_1").keys())

        self.log.debug(
            f"Available low resolution image keys for frame 1: {list(get_group(self._file, '/beamformed_data/frame_1').keys())}"
        )

        # Kick out high resolution image
        if "high_res_image" in lri_names_list:
            lri_names_list.remove("high_res_image")

        self._num_of_low_res_img_per_frame = len(lri_names_list)

        # Calculate indices of existing low resolution images
        self._lri_indices = [
            int(filename.split("_")[-1]) for filename in lri_names_list
        ]
        self.log.debug(
            f"number of available LRIs per frame: {self._num_of_low_res_img_per_frame}"
        )
        self.log.debug(f"Indices of available LRIs: {self._lri_indices}")

        # Check the type of the dataset: experimental or simulation
        # If sim_params group is empty then it is experimental data
        if "sim_params" in list(self._file.keys()):
            self._simulation_flag = True
        else:
            self._simulation_flag = False

    def close_file(self) -> None:
        self._file.close()

    # Get the Image data for the mth acquisition of nth frame
    def get_low_res_image(self, n_frame: int, m_low_res_img: int) -> np.ndarray:

        if n_frame not in self._frames_indices:
            raise ValueError(f"{n_frame = } is not available in the dataset")

        if m_low_res_img not in self._lri_indices:
            raise ValueError(f"{m_low_res_img = } is not available in the dataset")

        # Create a path to the image
        img_path = "frame_" + str(n_frame) + "/low_res_image_" + str(m_low_res_img)

        return get_array(self._file, "/beamformed_data/" + img_path)

    # Get the high resolution image
    def get_high_res_image(self, n_frame: int) -> np.ndarray:

        if n_frame not in self._frames_indices:
            raise ValueError(f"{n_frame = } is not available in the dataset")

        # Create a path to the image
        img_path = "frame_" + str(n_frame) + "/high_res_image"

        return get_array(self._file, "/beamformed_data/" + img_path)

    # Get positions of the scatters
    def get_scatters_coords(self) -> np.ndarray:

        if self._simulation_flag:
            return get_array(self._file, "sim_params/scatters_data")
        else:
            raise ValueError(
                "Scatters positions are available only in simulation mode."
            )

    # Get pixels coords
    def get_pixels_coords(self) -> np.ndarray:

        return get_array(self._file, "params/pixels_coords_x_z")

    # Get elements coords
    def get_elements_coords(self) -> np.ndarray:

        return get_array(self._file, "params/elements_coords")

    # Get elements coords
    def get_fps(self) -> float:

        return get_scalar(self._file, "params/fps")

    # Returns a list of found frames
    @property
    def frame_indices(self) -> list[int]:

        return self._frames_indices

    # Returns a list of found low resolution images
    # for a single frame
    @property
    def lri_indices(self) -> list[int]:

        return self._lri_indices
