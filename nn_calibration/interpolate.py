import logging
from scipy.interpolate import RegularGridInterpolator, interp1d
import numpy as np
import xraydb
import matplotlib.pyplot as plt
import nn_calibration

logger = logging.getLogger(__name__)

NA_I_DENSITY = 3.67

e_mu = np.logspace(0, 2.9, 512)
val_mu = xraydb.material_mu('NaI', 1e3*e_mu, NA_I_DENSITY, 'total')
na_i_mu = interp1d(e_mu, val_mu, 'nearest',
                   bounds_error=False,
                   fill_value='extrapolate')


class Interpolation:
    def __init__(self, energies, positions, centers):
        logger.info('s', centers.shape)
        max_error = 0.
        mean_error = 0.

        for i, e in enumerate(energies):
            max_error_e, mean_error_e = \
                self.validate(e, positions, centers[:,:,:,i])
            max_error = max(max_error, max_error_e)
            mean_error += mean_error_e/len(energies)

        logger.info('Interpolation Statistics:')
        logger.info(f'\tMean Error:{mean_error:.2f}')
        logger.info(f'\tMax Error:{max_error:.2f}')

    def __call__(self, *args, **kwargs):
        return self.values(*args, **kwargs)

    def values(self, energy, x, y):
        return

    def validate(self, energy, positions, centers):

        interp_centers = np.empty((*positions.shape[1:], 4))
        for idx in np.ndindex(positions.shape[1:]):
            interp_centers[idx] = self([energy], [positions[0,:,:][idx]], [positions[1,:,:][idx]])

        errors = interp_centers - centers
        nn_calibration.plot.grid_colormesh(positions[0, :, :],
                                           positions[1, :, :],
                                           errors,
                                           label=f'error_{energy:.1f}keV')

        x_hr = np.linspace(np.min(positions[0, :, :]),
                        np.max(positions[0, :, :]), 70)
        y_hr = np.linspace(np.min(positions[1, :, :]),
                        np.max(positions[1, :, :]), 70)

        X, Y = np.meshgrid(x_hr, y_hr)
        Z = np.empty((*X.shape, 4))
        for idx in np.ndindex(X.shape):
            Z[idx] = self([energy], [X[idx]], [Y[idx]])

        nn_calibration.plot.grid(x_hr, y_hr, Z, label=f'interp_{energy:.1f}keV')
        return np.max(np.abs(errors)), np.mean(np.abs(errors))


class CubicInterpolation(Interpolation):
    def __init__(self, energies, positions, centers):
        ind = np.argsort(energies)
        energies = np.array(energies)[ind]
        centers = np.moveaxis(centers[:, :, :, ind], 3, 0)
        positions = np.array(positions)
        logger.debug(energies)
        # TODO, test that grid is rectancular
        x = positions[0, 0, :].copy()
        y = positions[1, :, 0].copy()

        for i, e in enumerate(energies):
            X, Y = np.meshgrid(x, y)
            nn_calibration.plot.grid(X, Y, centers[i, :, :, :], label=f'{e}keV')

        self.grid_interp = RegularGridInterpolator([energies, x, y], centers,
                                                method='cubic', bounds_error=False, fill_value=None)   

    def values(self, energy, x, y):
        return self.grid_interp((energy, x, y))


class PCACleanedInterpolation(CubicInterpolation):
    def __init__(self, energies, positions, centers, basis):
        cleaned_centers = basis.T @ (basis @ centers)
        super().__init__(energies, positions, cleaned_centers)


class PCAEnergyInterpolation(Interpolation):
    def __init__(self, energies, positions, centers, basis):
        assert centers.shape[-2] == 4
        assert positions.shape[1:] == centers.shape[:2]
        assert positions.shape[1] * positions.shape[2] == basis.shape[0]
        energies = np.double(energies)
        # TODO, test that grid is rectancular
        x = positions[0, :, 0].copy()
        y = positions[1, 0, :].copy()

        basis_3d = np.reshape(basis, (len(x), len(y), -1))

        self.basis_interp = RegularGridInterpolator([x, y],
                                            basis_3d, method='linear',
                                            bounds_error=False, fill_value=None)

        # data_proj = basis.T @ np.reshape(centers, (-1, centers.shape[0]))
        flat_centers = np.reshape(nn_calibration.pca.flip(centers),
                                  (-1, *centers.shape[-2:]))
        data_proj = np.apply_along_axis(lambda v: np.dot(v, basis), 0, flat_centers)

        del flat_centers
        terms = basis_3d.shape[-1]

        A = np.vstack([self.calc_e_lin(energies), np.ones(len(energies))]).T
        y = np.empty(data_proj.shape)

        self.v1 = np.zeros((terms, 4))
        self.v2 = np.zeros((terms, 4))
        for tube in range(4):
            for i in range(terms):
                y[i, tube, :] = self.calc_y_lin(data_proj[i, tube, :], energies)
                self.v2[i, tube], self.v1[i, tube] = \
                    np.linalg.lstsq(A, y[i, tube, :], rcond=None)[0]
            label = f'pca_tube_{tube}'
            nn_calibration.plot.pca_interp(A[:, 0], y[:, tube, :],
                                           (self.v1[:, tube], self.v2[:, tube]),
                                           label)

        super().__init__(energies, positions, centers)

    def values(self, energy, x, y):
        shape_out = (*np.shape(energy), 4)
        energy = np.ravel(energy)
        x = np.ravel(x)
        y = np.ravel(y)
        e_var = self.calc_e_lin(energy)

        points = len(x)
        result = np.empty((points, 4))

        max_step = 500_000
        for i_l in range(0, points, max_step):
            i_r = min(points, i_l+max_step)
            # if i_l > 0:
            #     print(f'Running points {i_l} to {i_r}')
            term_weights = (np.multiply.outer(self.v1, energy[i_l:i_r])
                            + np.multiply.outer(self.v2, energy[i_l:i_r]*e_var[i_l:i_r]))

            pos_interp_term = np.empty((*self.v1.shape, i_r-i_l))
            for tube in range(4):
                flip_y, flip_x = nn_calibration.pca.flip_tube[tube]
                pos_interp_term[:, tube, :] = \
                    self.basis_interp(((-1. if flip_x else 1.)*x[i_l:i_r],
                                    (-1. if flip_y else 1.)*y[i_l:i_r])).T
            result[i_l:i_r] = np.sum(pos_interp_term * term_weights, axis=0).T
        return result.reshape(shape_out)

    def calc_y_lin(self, y, energies):
        return y/energies

    def calc_e_lin(self, energies):
        return energies


class PCADepthInterpolation(PCAEnergyInterpolation):
    def calc_e_lin(self, energies):
        # depth in mm: (10 mm/cm) / (mu in /cm)
        return 10./na_i_mu(energies)


# TODO, move short name into class
methods = {'cubic': CubicInterpolation,
           'pca_clean': PCACleanedInterpolation,
           'pca_energy_linear': PCAEnergyInterpolation,
           'pca_depth_linear': PCADepthInterpolation}

uses_basis = {'cubic': False,
              'pca_clean': True,
              'pca_energy_linear': True,
              'pca_depth_linear': True}

