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

import logging
from typing import override, Optional
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.neighbors import KDTree
from x_ray_imager_bagriff.position_estimation import (
    PointEstimator,
    anger_basis
)
from scipy.stats import poisson

logger = logging.getLogger(__name__)


class PointLookup(PointEstimator):
    """A generic base for estimator methods using closest calibration."""
    short_name = 'base_lookup'

    @override
    def __init__(self,
                 response: ArrayLike,
                 energies: ArrayLike,
                 positions: ArrayLike):
        super().__init__(response, energies, positions)

        self._idx = None  # Stores index looked up from last request.
        self._weights = None

    def lookup_index(self,
                     observations: ArrayLike
                     ) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        """Find the closest calibration points and their weight.

        Using any lookup method, grab the index some number of points. Also
        return a weight for each point. All weights for an observation should
        sum to one.
        
        Both arrays should have shape (*any_measurements_shape, n_indices).

        Args:
             observations: See get_value.
                Shape should be (*any_measurements_shape, n_detectors).

        Returns:
            A tuple of two arrays. The first is a set of indices for close
            calibration points from each measurement. The second is the
            relative weight for each index.
        """
        logger.warning("PointLookup lookup_index called but not implemented.")

        self._idx = np.zeros((*np.shape(observations)[:-1], 1), dtype=np.intp)
        self._weights = np.zeros_like(self._idx, dtype=np.float64)

        return self._idx, self._weights

    @override
    def get_value(self, observations):
        """Estimate using a weighted set of close calibration points."""
        idx, weights = self.lookup_index(observations)

        if not np.allclose(np.sum(weights, axis=-1), 1):
            logger.warning('Some weights did not add up to one.')

        weights = np.repeat(weights[np.newaxis, ...], 3, axis=0)
        print(weights)

        return np.average(self.points[:, idx], weights=weights, axis=-1)

    @override
    def get_values_with_error(self, observations: ArrayLike
                              ) -> tuple[NDArray[np.double],
                                         NDArray[np.double]]:
        prediction = self.get_value(observations)
        assert self._idx is not None and self._weights is not None

        # Calculate weighted variance
        d2 = np.abs(self.points[:, self._idx] - prediction[..., np.newaxis])**2
        var = np.sum(self._weights * d2, axis=1) \
            / np.sum(self._weights, axis=-1)
        error = np.sqrt(var)

        return prediction, error


class TreeLookup(PointLookup):
    """Point lookup using a KDTree."""
    short_name = 'kdtree'

    def __init__(self,
                 response: ArrayLike,
                 energies: ArrayLike,
                 positions: ArrayLike,
                 k_lookup: Optional[int] = None):
        super().__init__(response, energies, positions)
        self._kdtree = KDTree(self.response,
                              leaf_size=128,
                              metric='euclidean')

        if k_lookup is None:
            # Find a very low amplitude point which will have high variance.
            # See how many points are needed to grab a roughly 3 sigma radius.
            amplitude = np.sum(self.response, axis=-1)
            i = np.argmin(amplitude)

            k_lookup = int(self._kdtree.query_radius([self.response[i]],
                                                     3*np.sqrt(amplitude[i]/4),
                                                     count_only=True)[0])
            logger.info('Lookup set to %s points.', k_lookup)

        self.k_lookup = k_lookup

    @override
    def lookup_index(self, observations):
        """Find closes indices in the KDtree."""
        observations = np.array(observations, dtype=np.int32)

        self._idx = self._kdtree.query(observations, 64,
                                       return_distance=False,
                                       sort_results=False)
        k = np.repeat(observations[..., np.newaxis, :],
                      np.shape(self._idx)[-1], axis=-2)
        mu = self.response[self._idx]
        p = np.prod(poisson.pmf(k, mu), axis=-1)

        logger.debug("Identifying observations: %s", observations)
        logger.debug("Identified calibration points: %s",
                     self.points[:, self._idx])
        logger.debug("Probabilities: %s", p)

        incomplete_set = np.min(p, axis=-1) > 0.01 * np.max(p, axis=-1)
        if np.any(incomplete_set):
            logger.warning("Some measurements fit calibration poorly.")
            logger.info("Max match at points: %s",
                        observations[incomplete_set])

        self._weights = p / np.sum(p, axis=-1)[..., np.newaxis]

        return self._idx, self._weights
