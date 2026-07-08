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

"""Anger imager style portioning algorithms."""
import logging
import numpy as np
from numpy.typing import ArrayLike, NDArray
from x_ray_imager_bagriff.position_estimation import PointEstimator

logger = logging.getLogger(__name__)


def anger_basis(X: ArrayLike  # pylint: disable=invalid-name
                ) -> tuple[NDArray[np.double],
                           NDArray[np.double],
                           NDArray[np.double]]:
    """Simple Anger imager positioning algorithm.

    Example::

        x = sum(detectors_plus_x) - sum(detectors_minus_x)
        y = sum(detectors_plus_y) - sum(detectors_minus_y)
        an then normalized by the sum of all detectore
    
    It is assumed here that the detectors are numbered::

            +y
        (2) | (1)
        ----+---- +x
        (3) | (0)
    
    Args:
        X: Array of measurements. Shape is ``(n events, n detectors)``.

    Returns:
        Tuple of three arrays. The amplitude, sum of all detectors.
        The Anger x position. The Anger y position.
    """
    n_detectors = np.shape(X)[1]
    if n_detectors != 4:
        # Project specific warning. Other imagers may need this removed.
        logger.warning('Expected 4 detectors, but got %s', n_detectors)

    amplitude = np.sum(X, axis=1, dtype=np.float64)

    # Exclude null events where the amplitudes are all zero.
    amplitude = np.where(amplitude < 1., np.nan, amplitude)

    x = np.dot(X, [1, 1, -1, -1]) / amplitude
    y = np.dot(X, [-1, 1, 1, -1]) / amplitude

    return amplitude, x, y


class AngerSimple(PointEstimator):
    """Anger algorithm based PointEstimator."""
    short_name = 'anger'

    def __init__(self, channels, energies, positions):
        amp, x, y = anger_basis(channels)

        self.a_x = np.max(np.abs(positions)) / np.max(np.abs(x))
        self.a_y = np.max(np.abs(positions)) / np.max(np.abs(y))
        self.a_e = np.mean(amp/energies)

    def get_value(self, channels, return_error=False):
        amp, x, y = anger_basis(channels)
        gain = 1/2

        d_amp = gain*np.sqrt(amp)
        d_s = d_amp/amp

        e = amp / self.a_e
        de = d_amp / self.a_e

        x = self.a_x * x
        dx = self.a_x*d_s
        y = self.a_y * y
        dy = self.a_y*d_s

        if return_error:
            return (e, x, y), (de, dx, dy)
        else:
            return (e, x, y)
