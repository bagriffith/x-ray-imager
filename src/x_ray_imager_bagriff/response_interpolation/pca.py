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
import numpy as np
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)

DEFAULT_COMPS = 5

DIRECTION = [[-1, -1, 1, 1],
             [1, -1, -1, 1]]


def flip_response(response: ArrayLike) -> NDArray[np.float64]:
    """Mirror the four detector responses to have the same orientation.
    
    ``flip_response`` changes to matrix so all detectors use the same
    positions. ``flip_position`` flips the values in that position array.

    Args:
        response: Detector response function on a symmetrical grid,
            n_pt across on each axis. Shape should be
            (*any_shape, n_pt, n_pt, n_det=4)
    """
    response = np.float64(response)
    assert response.shape[-1] == 4

    for i in range(4):
        response[..., i] = response[...,
                                    ::DIRECTION[0][i],
                                    ::DIRECTION[1][i],
                                    i]

    return response


def flip_position(positions):
    """Mirror the x/y values so all detectors have the same orientation.

    ``flip_response`` changes to matrix so all detectors use the same
    positions. ``flip_position`` flips the values in that position array.
    """
    return np.moveaxis(
        np.array([[det_sign * comp for det_sign in signs]
                   for comp, signs in zip(positions, DIRECTION)]),
                   1, -1)


def form_basis(response: ArrayLike
               ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Use PCA to create a basis based variation of tube response shape

    Args:
        response: Collection of responses used to determine the new basis.
        It should be a list of response grids. The shape should be
        (n_sets, nx, ny, 4). The grid should be square (n_x == n_y).

    Returns:
        Tuple of two arrays. The first is the set of PCA basis functions in
        order of singular value magnitude. The shape is is (n_sets, nx, ny).
        The second is an array of those singular values

        Not all of these components are useful. Truncate it to k components
        using ``basis[:k, :, :]``.
    """
    # Verify correct shape
    response_shape = np.shape(response)
    logger.debug('for_basis centers shape %s', response_shape)

    if (len(response_shape) != 4
        or response_shape[-1] != 4
        or response_shape[1] != response_shape[2]):
        raise ValueError(f'Incorrect shape for response, {response_shape}.')

    n_pts = response_shape[1]  # Number of x or y samples

    response = flip_response(response)
    data = np.moveaxis(response, -1, 1).reshape((-1, n_pts*n_pts))
    # data /= np.max(data, axis=1)[:, np.newaxis]  # Normalize the responses
    logger.debug('Shape of data for SVD: %s', data.shape)

    _, s, vh = np.linalg.svd(data, full_matrices=False)
    logger.info('Top singular values: %s', s[:10])
    logger.debug('Other singular values: %s', s[10:])
    return vh.reshape((-1, n_pts, n_pts)), s
