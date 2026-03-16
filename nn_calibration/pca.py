import numpy as np
import logging
import nn_calibration

logger = logging.getLogger(__name__)

DEFAULT_COMPS = 5

flip_tube = {0: (False, True),
             1: (True, True),
             2: (True, False),
             3: (False, False)}


def flip(a):
    b = np.zeros_like(a)
    for tube_n in range(4):
        flip_y, flip_x = flip_tube[tube_n]
        b[:, :, tube_n] = a[::-1 if flip_x else 1, ::-1 if flip_y else 1, tube_n]
    return b


def form_basis(centers, terms=8):
    """Use PCA to create a basis based variation of tube response shape

    Args:
        centers: shape (nx, ny, 4, )
    """
    # TODO mirror tube
    logger.debug('%s components', terms)
    logger.debug(f'for_basis centers shape {centers.shape}')
    assert (len(centers.shape) == 4) and centers.shape[2] == 4
    data = flip(centers)

    # for i in range(data.shape[3]):
    #     X, Y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
    #     nn_calibration.plot.grid(X, Y, data[:,:,:,i], label=f'pt{i}_pca')

    logger.debug(f'for_basis data shape {data.shape}')

    data_flat = np.reshape(data, (data.shape[0]*data.shape[1],
                                  data.shape[2]*data.shape[3])).T

    # X, Y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
    # q = np.empty((data.shape[0], data.shape[1], 4))
    # for i in range(4):
    #     q[:, :, i] = np.reshape(data_flat[i, :], (data.shape[0], data.shape[1]))
    # nn_calibration.plot.grid(X, Y, q, label=f'ptflat_pca')

    logger.debug(f'for_basis data_flat shape {data_flat.shape}')

    _, s, vh = np.linalg.svd(data_flat, full_matrices=False)
    logger.info(s[:terms])
    return vh[:terms].T
