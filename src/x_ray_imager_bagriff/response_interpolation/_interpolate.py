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

"""Interpolate between calibration source positions and energies."""
import logging
from typing import Optional
from scipy.interpolate import RegularGridInterpolator, make_interp_spline
import numpy as np
from numpy.typing import ArrayLike, NDArray
import xraydb
from x_ray_imager_bagriff.response_interpolation import pca
from x_ray_imager_bagriff.response_interpolation.plot import (
    GenericResponseDiagnostic)

logger = logging.getLogger(__name__)


class Interpolation:
    """Generic base to interpolate x-ray detector response and validate it.

    This is intended to produce the best estimate of what the mean response
    from the imager would be for a given x-ray position and energy. Specific
    implementation is left for inherited classes.
    """
    def __init__(self,
                 energies: ArrayLike,
                 positions: ArrayLike,
                 responses: ArrayLike,
                 input_diagnostic: Optional[GenericResponseDiagnostic] = None,
                 output_diagnostic: Optional[GenericResponseDiagnostic] = None,
                 error_diagnostic: Optional[GenericResponseDiagnostic] = None
                 ) -> None:
        """Initialize the interpolation.

        Args:
            energy: Array of energies used for the calibration points.
                Shape is (n_energies).
            positions: Array of the x and y position for each calibration
                point. It's expected that the same points are used for each
                energy. Shape should be (2, n_x_positions, n_y_positions)
            responses: Array of expected response for each x-ray energy and
                position used in calibration. Shape should be
                (n_energies, n_x_positions, n_y_positions, n_detectors)
            input_diagnostic: Diagnostic plot of the calibration responses.
            output_diagnostic: Diagnostic plot to use for the predicted output
                at each calibration point and the specified energy.
            error_diagnostic: Diagnostic plot of the difference between
                responses from calibration and the prediction.
        """
        self._check_shape(energies, positions, responses)
        energies = np.array(energies, dtype=np.double)
        positions = np.array(positions, dtype=np.double)
        responses = np.array(responses, dtype=np.double)

        if input_diagnostic is not None:
            for response, energy in zip(responses, energies):
                input_diagnostic.plot_diagnostic(response, positions)
                input_diagnostic.savefig(f'in-{energy:.1f}keV.png', dpi=300)

        max_error = 0.
        mean_error = 0.

        for response, energy in zip(responses, energies):
            # For each energy calibration, check how accurately it's reproduced
            max_error_e, mean_error_e = \
                self.validate(energy, positions, response,
                              error_diagnostic=error_diagnostic,
                              output_diagnostic=output_diagnostic)
            max_error = max(max_error, max_error_e)
            mean_error += mean_error_e/len(energies)

    def __call__(self, *args, **kwargs):
        return self.values(*args, **kwargs)

    def values(self,
               energy: ArrayLike,
               x: ArrayLike,
               y: ArrayLike
               ) -> NDArray[np.double]:
        """Estimate the response for an energy and position.

        Implementation is left for the specific subclass. It should be able
        to tolarate any array shape, provided it's the same for all three
        variables.

        Args:
            energy: Array of energies for each response to predict.
            x: Array of x position coordinate for each response to predict.
            y: Array of y position coordinate for each response to predict.
        """
        _ = energy, x, y
        logger.warning('Base class is used with no interpolation method.')
        return np.full_like(energy, np.nan)

    def validate(self,
                 energy: float,
                 positions: NDArray[np.double],
                 responses: NDArray[np.double],
                 output_diagnostic: Optional[GenericResponseDiagnostic] = None,
                 error_diagnostic: Optional[GenericResponseDiagnostic] = None
                 ) -> tuple[float, float]:
        """Check the predictions against calibration and plot diagnostics.

        Args:
            energy: Energy of the calibration source.
            positions: Array of calibration points.
            responses: Array of detector outputs for each calibration point.
            output_diagnostic: Diagnostic plot to use for the predicted output
                at each calibration point and the specified energy.
            error_diagnostic: Diagnostic plot of the difference between
                responses from calibration and the prediction.
        Returns:
            Tuple of the maximum and mean error between the calibration
            responses and the predicted output for the same energies and
            positions.
        """
        interp_centers = self(np.full(positions.shape[1:], energy),
                              positions[0, :, :],
                              positions[1, :, :])

        errors = interp_centers - responses

        x_hr = np.linspace(np.min(positions[0, :, :]),
                           np.max(positions[0, :, :]), 70)
        y_hr = np.linspace(np.min(positions[1, :, :]),
                           np.max(positions[1, :, :]), 70)

        if output_diagnostic is not None:
            x_mesh, y_mesh = np.meshgrid(x_hr, y_hr)
            e_mesh = np.full_like(x_mesh, energy)
            z_mesh = self(e_mesh, x_mesh, y_mesh)
            output_diagnostic.plot_diagnostic(z_mesh,
                                              
                                              
                                              np.array([x_mesh, y_mesh]))
            output_diagnostic.savefig(f'out-{energy:.1f}keV.png', dpi=300)

        if error_diagnostic is not None:
            error_diagnostic.plot_diagnostic(errors, positions)
            error_diagnostic.savefig(f'error-{energy:.1f}keV.png', dpi=300)

        max_error = np.max(np.abs(errors))
        mean_error = float(np.mean(np.abs(errors)))

        logger.info('Interpolation Statistics for %.1f keV:', energy)
        logger.info('  Mean Error: %.2f', mean_error)
        logger.info('  Max Error: %.2f', max_error)
        if mean_error > 4.0:
            # Somewhat arbitrary threshold. An error < 1.0 should be harmless.
            logger.warning('Mean error for %s keV is high: %.2f',
                           energy, mean_error)

        return max_error, mean_error

    @staticmethod
    def _check_shape(energy: ArrayLike,
                     positions: ArrayLike,
                     responses: ArrayLike):
        """Check the shapes of the input arrays.

        Args:
            See __init__().

        Raises:
            ValueError: Shapes of the three arrays do not agree.
        """
        energy_shape = np.shape(energy)
        logger.debug('%s init - energy shape: %s',
                     __class__, energy_shape)
        positions_shape = np.shape(positions)
        logger.debug('%s init - positions shape: %s',
                     __class__, positions_shape)
        responses_shape = np.shape(responses)
        logger.debug('%s init - responses shape: %s',
                     __class__, responses_shape)

        if len(energy_shape) != 1:
            raise ValueError(f'energy array is the wrong shape {energy_shape}')

        if len(positions_shape) != 3 or positions_shape[0] != 2:
            raise ValueError('positions array is the wrong shape '
                             f'{positions_shape}')

        if (len(responses_shape) != 4
            or
            responses_shape[:3] != (energy_shape[0], *positions_shape[1:])):
            raise ValueError('responses array is the wrong shape '
                             f'{responses_shape}')

        if responses_shape[3] != 4:
            # Project specific warning. Other imagers may need this removed.
            logger.warning('Expected 4 detectors, but got %s',
                           responses_shape[3])


class CubicInterpolation(Interpolation):
    def __init__(self, energies, positions, centers, **kwargs):
        ind = np.argsort(energies)
        energies = np.array(energies)[ind]
        centers = centers[ind, :, :, :]
        positions = np.array(positions)
        logging.debug(energies)
        # TODO, test that grid is rectancular
        x = positions[0, 0, :].copy()
        y = positions[1, :, 0].copy()

        self.grid_interp = \
            RegularGridInterpolator([energies, x, y], centers,
                                    method='cubic',
                                    bounds_error=False)
        super().__init__(energies, positions, centers, **kwargs)

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
        energies = np.ndarray(energies, dtype=np.float64)
        # TODO, test that grid is rectancular
        x = positions[0, :, 0].copy()
        y = positions[1, 0, :].copy()

        basis_3d = np.reshape(basis, (len(x), len(y), -1))

        self.basis_interp = \
            RegularGridInterpolator([x, y],
                                    basis_3d,
                                    method='linear',
                                    bounds_error=False)

        # data_proj = basis.T @ np.reshape(centers, (-1, centers.shape[0]))
        flat_centers = np.reshape(pca.flip(centers),
                                  (-1, *centers.shape[-2:]))

        data_proj = np.apply_along_axis(lambda v: np.dot(v, basis),
                                        0,
                                        flat_centers)

        del flat_centers
        terms = basis_3d.shape[-1]

        A = np.vstack([self.calc_e_lin(energies),
                       np.ones(len(energies))]).T
        y = np.empty(data_proj.shape)

        self.v1 = np.zeros((terms, 4))
        self.v2 = np.zeros((terms, 4))
        for tube in range(4):
            for i in range(terms):
                y[i, tube, :] = self.calc_y_lin(data_proj[i, tube, :],
                                                energies)
                self.v2[i, tube], self.v1[i, tube] = \
                    np.linalg.lstsq(A, y[i, tube, :], rcond=None)[0]

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
            term_weights = (
                np.multiply.outer(self.v1,
                                  energy[i_l:i_r])
                + np.multiply.outer(self.v2,
                                    energy[i_l:i_r] * e_var[i_l:i_r])
                )

            pos_interp_term = np.empty((*self.v1.shape, i_r-i_l))
            for tube in range(4):
                flip_y, flip_x = pca.flip_tube[tube]
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
    def __init__(self, energies, positions, centers,
                 basis, mu_interpolator=None):
        if mu_interpolator is None:
            # Default to NaI
            NA_I_DENSITY = 3.67  # g/cm^3

            e_mu = np.logspace(0, 2.9, 512)
            val_mu = xraydb.material_mu('NaI', 1e3*e_mu, NA_I_DENSITY, 'total')
            mu_interpolator = make_interp_spline(e_mu, val_mu, k=1)

        self.mu = mu_interpolator

        super().__init__(energies, positions, centers, basis)

    def calc_e_lin(self, energies):
        # depth in mm: (10 mm/cm) / (mu in /cm)
        return 10. / self.mu(energies)


# TODO, move short name into class
methods = {'cubic': CubicInterpolation,
           'pca_clean': PCACleanedInterpolation,
           'pca_energy_linear': PCAEnergyInterpolation,
           'pca_depth_linear': PCADepthInterpolation}

uses_basis = {'cubic': False,
              'pca_clean': True,
              'pca_energy_linear': True,
              'pca_depth_linear': True}
