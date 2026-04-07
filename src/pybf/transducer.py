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
from typing import Optional

import numpy as np


class Transducer:
    def __init__(
        self,
        num_of_x_elements: int = 1,
        num_of_y_elements: int = 1,
        x_pitch: float = 0,
        y_pitch: float = 0,
        x_width: float = 0,
        y_width: float = 0,
        f_central_hz: float = 0,
        bandwidth_hz: float = 0,
        active_elements: Optional[list[int]] = None,
        speed_of_sound: float = 1540,
    ):
        self.log = logging.getLogger(__name__)

        self._num_of_x_elements = num_of_x_elements
        self._num_of_y_elements = num_of_y_elements
        self._num_of_elements = num_of_y_elements * num_of_x_elements

        self._x_pitch = x_pitch
        self._y_pitch = y_pitch

        self._x_width = x_width
        self._y_width = y_width

        self._f_central_hz = f_central_hz
        self._bandwidth_hz = bandwidth_hz

        self._speed_of_sound = speed_of_sound

        # Calculate X, Y coordinates of transducer elements
        self._calc_elements_coords()

        # Check if active elements were specified
        # By default all the elements are active
        if active_elements is not None:
            self.set_active_elements(active_elements)
        else:
            self._active_elements = None

    # Returns x,y coords for the elements of transducer
    def _calc_elements_coords(self) -> None:

        # Calc x coords
        x_coords = np.arange(0, self._num_of_x_elements) * (self._x_pitch)
        x_coords = x_coords.reshape(
            -1,
        )
        # Put zero to the center of array
        x_coords = x_coords - (x_coords[-1] - x_coords[0]) / 2

        # Cals y coords
        y_coords = np.arange(0, self._num_of_y_elements) * (self._y_pitch)
        y_coords = y_coords.reshape(
            -1,
        )
        # Put zero to the center of array
        y_coords = y_coords - (y_coords[-1] - y_coords[0]) / 2

        self._elements_coords = np.transpose(
            np.dstack(np.meshgrid(x_coords, y_coords)).reshape(-1, 2)
        )

    # Set the active elements of the transducer by list of indices
    # Attention: elements numeration starts from 0
    def set_active_elements(self, active_elements: list[int]) -> None:

        self._active_elements = np.array(active_elements, dtype=int)
        self.log.debug(f"number of active elements = {len(active_elements)}")

        # Calculate X, Y coordinates of transducer elements
        self._calc_elements_coords()

        # Update number of elements
        self._num_of_elements = len(self._active_elements)

        # Update coordinates of activated elements
        self._elements_coords = self._elements_coords[:, self._active_elements]

    # Returns transducers elements coordinates
    @property
    def elements_coords(self) -> np.ndarray:

        return self._elements_coords

    # Returns number of transducers elements
    @property
    def num_of_elements(self) -> int:

        return self._num_of_elements

    # Returns indices of active transducer elements
    @property
    def active_elements(self) -> Optional[np.ndarray]:

        return self._active_elements

    # Returns central frequency
    @property
    def f_central_hz(self) -> float:

        return self._f_central_hz

    # Returns bandwidth
    @property
    def bandwidth_hz(self) -> float:

        return self._bandwidth_hz

    # Returns speed of sound
    @property
    def speed_of_sound(self) -> float:

        return self._speed_of_sound
