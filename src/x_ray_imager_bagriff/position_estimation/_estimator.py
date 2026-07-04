# Copyright (c) 2026 Brady Griffith
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""The PointEstimator base class, inherited by each implementation."""
import logging
from pathlib import Path
from typing import Union
import numpy as np
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)


class PointEstimator:
    """Base class for converting an x-ray observation into an energy/position.

    Attributes:
        short_name: A string of a name that should be used for the method
            implemented in each subclass.
        response: An NDArray[float] with the expected responses for each point.
        energies: An NDArray[float] with the energy of the points.
        positions: An NDArray[float] with the position of the points.
    """
    short_name = 'base'

    def __init__(self,
                 response: ArrayLike,
                 energies: ArrayLike,
                 positions: ArrayLike):
        """Loads the generic estimator.

        Args:
            response: Expected responses for an array of points.
                The array can be any shape that matches the energies
                and positions: (*shape of measurements, n detectors).
            points: Array  with axis 0 having eenergy, x and y of samples.
                Shape is (3, *shape of measurements)
        """
        # Check input array shapes
        response_shape = np.shape(response)

        if len(response_shape) < 2:
            raise ValueError("Provided response should be an array of "
                             "measurements, not 1-D.")

        if response_shape[-1] != 4:
            logger.warning("Measurements are expected to have 4 detectors, "
                           "but %s are provided.", response_shape[-1])

        energies_shape = np.shape(energies)

        if energies_shape != response_shape[:-1]:
            raise ValueError("Mismatch between response and energies shape, "
                             f"{response_shape} and {energies_shape}.")

        positions_shape = np.shape(positions)

        if positions_shape[0] != 2 or len(positions_shape) == 1:
            raise ValueError("The positions array must have two spacial "
                             f"dimensions, but {positions_shape[0]} provided.")

        if positions_shape[1:] != response_shape[:-1]:
            raise ValueError("Mismatch between response and positions shape, "
                             f"{response_shape} and {positions_shape}.")

        self._idx = None

        # Load arrays
        self.response = np.array(response, dtype=np.float64)\
                            .reshape((-1, response_shape[-1]))
        self.points = np.append(np.array(energies, dtype=np.float64)\
                                    .reshape((1, -1)),
                                np.array(positions, dtype=np.float64)\
                                    .reshape((2, -1)),
                                axis=0)

    def __call__(self, observations: ArrayLike) -> NDArray[np.double]:
        """See get_value()."""
        return self.get_value(observations)

    def get_value(self, observations: ArrayLike) -> NDArray[np.double]:
        """Estimate the x-ray energy/possition producing the observations.

        Args:
            observations: Measurements from the x-ray imager. Can be any
                shape, but the last dimmention must have n_detectors elements.
                Shape is (*any_measurements_shape, n_detectors)
        
        Returns:
            An array of the estimated energy, x, and y. Shape is
            (3, *any_measurements_shape).
        """
        logger.warning("Unimplemented base estimator is being called.")
        return np.full((3, *np.shape(observations)[:-1]), np.nan)

    def get_values_with_error(self, observations: ArrayLike
                              ) -> tuple[NDArray[np.double],
                                         NDArray[np.double]]:
        """Estimate x-ray energy/position with uncertainties.
        
        Args:
            observations: See get_values
        """
        logger.warning("Unimplemented base estimator is being called.")
        estimation = self.get_value(observations)
        error = np.zeros_like(estimation)
        return estimation, error

    def save_to(self, path: str|Path):
        """Save this estimator to a file to be reloaded later.

        This method should be undone by `load_from(path)`. If not overridden,
        the input response arrays will be saved in the "npz" format to be
        reloaded by the initializer. For subclasses where this is an expensive
        operation, this method could be used to save an intermediate result
        instead.

        Args:
            path: Path to save the estimator parameters. It's treated
                like a .npz file, but the extension isn't enforced.
        """
        np.savez(str(path), allow_pickle=False,
                 name=[self.short_name],
                 positions=self.positions,
                 energies=self.energies,
                 response=self.response)

    @classmethod
    def load_from(cls, path: str|Path):
        """Reloads the estimator from a file made using `save_to(path)`."""
        loaded = np.load(str(path))

        if loaded['name'][0] != cls.short_name:
            raise ValueError(f"Loaded array {loaded['name'][0]} is not a "
                             f"{cls.short_name} estimator.")

        return cls(loaded['response'],
                   loaded['energies'],
                   loaded['positions'])

    @property
    def energies(self) -> NDArray[np.float64]:
        """Returns the energy component of points."""
        return self.points[0]

    @property
    def positions(self) -> NDArray[np.float64]:
        """Returns the energy component of points."""
        return self.points[1:]
