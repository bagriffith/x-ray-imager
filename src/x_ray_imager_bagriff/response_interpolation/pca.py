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

DEFAULT_COMPS = 5

flip_tube = {0: (False, True),
             1: (True, True),
             2: (True, False),
             3: (False, False)}


def flip(a):
    b = np.zeros_like(a)
    for tube_n in range(4):
        flip_y, flip_x = flip_tube[tube_n]
        b[:, :, tube_n] = a[::-1 if flip_x else 1,
                            ::-1 if flip_y else 1,
                            tube_n]
    return b


def form_basis(centers, terms=8):
    """Use PCA to create a basis based variation of tube response shape

    Args:
        centers: shape (nx, ny, 4, )
    """
    # TODO mirror tube
    logging.debug('%s components', terms)
    logging.debug('for_basis centers shape %s', centers.shape)
    assert (len(centers.shape) == 4) and centers.shape[2] == 4
    data = flip(centers)

    # for i in range(data.shape[3]):
    #     X, Y = np.meshgrid(np.arange(data.shape[0]),
    #                                  np.arange(data.shape[1]))
    #     nn_calibration.plot.grid(X, Y, data[:,:,:,i], label=f'pt{i}_pca')

    logging.debug('for_basis data shape %s', data.shape)

    data_flat = np.reshape(data, (data.shape[0]*data.shape[1],
                                  data.shape[2]*data.shape[3])).T

    # X, Y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
    # q = np.empty((data.shape[0], data.shape[1], 4))
    # for i in range(4):
    #     q[:, :, i] = np.reshape(data_flat[i, :],
    #                             (data.shape[0], data.shape[1]))
    # nn_calibration.plot.grid(X, Y, q, label=f'ptflat_pca')

    logging.debug('for_basis data_flat shape %s', data_flat.shape)

    _, s, vh = np.linalg.svd(data_flat, full_matrices=False)
    logging.info(s[:terms])
    return vh[:terms].T
