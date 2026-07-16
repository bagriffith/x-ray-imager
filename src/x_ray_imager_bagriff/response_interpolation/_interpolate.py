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
from typing import Optional, override
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from numpy.typing import ArrayLike, NDArray
from x_ray_imager_bagriff.response_interpolation import pca
from x_ray_imager_bagriff.response_interpolation.plot import (
    GenericResponseDiagnostic)

logger = logging.getLogger(__name__)


class Interpolation:
    """Generic base to interpolate x-ray detector response and validate it.

    This is intended to produce the best estimate of what the mean response
    from the imager would be for a given x-ray position and energy. Specific
    implementation is left for inherited classes.

    Attributes:
        points:
        responses:
        positions:
        energies:
    """
    short_name = 'default'
    uses_basis = False

    def __init__(self,
                 energies: ArrayLike,
                 positions: ArrayLike,
                 responses: ArrayLike,
                 input_diagnostic: Optional[GenericResponseDiagnostic] = None
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
        """
        self._check_shape(energies, positions, responses)
        self.energies = np.array(energies, dtype=np.double)
        self.positions = np.array(positions, dtype=np.double)
        self.responses = np.array(responses, dtype=np.double)

        if input_diagnostic is not None:
            for response, energy in zip(self.responses, self.energies):
                input_diagnostic.plot_diagnostic(response, self.positions)
                input_diagnostic.savefig(f'in-{energy:.1f}'.replace('.', '_')
                                         + 'keV.png', dpi=300)

    @property
    def x(self):
        """TODO"""
        return self.positions[0]

    @property
    def y(self):
        """TODO"""
        return self.positions[1]

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
        x_mesh, y_mesh = np.meshgrid(*[np.linspace(-70, 70, 71)]*2)

        max_error = 0.
        mean_error = 0.

        for energy, response in zip(self.energies, self.responses):
            predicted_response = self(np.full_like(self.positions[0], energy),
                                      self.positions[0],
                                      self.positions[1])

            if output_diagnostic is not None:
                output_diagnostic.plot_diagnostic(
                    self(np.full_like(x_mesh, energy), x_mesh, y_mesh),
                    np.array([x_mesh, y_mesh], dtype=np.float64)
                )

                output_diagnostic.savefig(
                    f'out-{energy:.1f}'.replace('.', '_') + 'keV.png',
                    dpi=300
                )

            error = predicted_response - response

            if error_diagnostic is not None:
                error_diagnostic.plot_diagnostic(
                    error,
                    self.positions
                )

                error_diagnostic.savefig(
                    f'error-{energy:.1f}'.replace('.', '_') + 'keV.png',
                    dpi=300
                )

            max_error_pt = np.max(np.abs(error))
            max_error = max(max_error, max_error_pt)
            mean_error_pt = float(np.mean(np.abs(error)))
            mean_error += mean_error_pt / len(self.energies)

            logger.info('Interpolation Statistics for %.1f keV:', energy)
            logger.info('  Mean Error: %.2f', mean_error_pt)
            logger.info('  Max Error: %.2f', max_error_pt)

            if mean_error_pt > 4.0:
                # Somewhat arbitrary threshold. An error < 1.0 should be harmless.
                logger.warning('Mean error for %s keV is high: %.2f',
                               energy, mean_error_pt)

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


class LinearInterpolation(Interpolation):
    """Linearly interpolate responses in both energy and postion."""
    short_name = 'linear'
    uses_basis = False

    @override
    def __init__(self, *args, **kwargs):
        """Initialize the interpolator, creating a scipy interpolator."""
        super().__init__(*args, **kwargs)

        sorted_index = np.argsort(self.energies)
        self.points = self.energies[sorted_index, ...]
        self.responses = self.responses[sorted_index, :, :, :]
        logging.debug(self.energies)

        if not np.allclose(self.positions,
                           np.meshgrid(self.x[:, 0],
                                       self.y[0, :],
                                       indexing='ij')):
            raise ValueError('Calibrations must use an even grid.')

        self.grid_interp = \
            RegularGridInterpolator([self.energies,
                                     self.x[:, 0],
                                     self.y[0, :]],
                                    self.responses,
                                    method='linear',
                                    bounds_error=False,
                                    fill_value=None) # type: ignore

    @override
    def values(self, energy, x, y):
        prediction = self.grid_interp((energy, x, y))
        prediction[prediction < 0.] = 0
        return prediction


class BasisFilteredInterpolation(LinearInterpolation):
    """Linear interpolation, using responses projected into a given basis."""
    short_name = 'pca_clean'
    uses_basis = True

    def __init__(self,
                 energies: ArrayLike,
                 positions: ArrayLike,
                 responses: ArrayLike,
                 basis: ArrayLike,
                 **kwargs) -> None:
        """Initialize the interpolation, cleaning the responses array.

        Args:
            energy: Energy of the calibration source.
            positions: Array of calibration points.
            responses: Array of detector outputs for each calibration point.
            basis: Set of orthonormal basis vectors. Each vector should have
                the same shape as the position grid. The array shape is
                (n_components, n_x, n_y).
            input_diagnostic: Diagnostic plot of the calibration responses.
        """

        if np.shape(basis)[1:] != np.shape(positions)[1:]:
            raise ValueError('Basis must match position size.')

        basis_flat = np.array(basis, dtype=np.float64)\
            .reshape((np.shape(basis)[0], -1))

        # Check that the basis is orthogonal
        if not np.allclose(basis_flat @ basis_flat.T,
                           np.identity(basis_flat.shape[0])):
            raise ValueError("Basis must have independent, unit components.")

        responses = np.array(responses, dtype=np.float64)
        responses = np.moveaxis(responses, 0, -2)
        cleaned_responses = np.transpose(basis) @ np.matmul(basis, responses)
        cleaned_responses = np.moveaxis(cleaned_responses, -2, 0)

        super().__init__(energies, positions, cleaned_responses, **kwargs)


class PCAEnergyInterpolation(Interpolation):
    r"""Response function is an energy dependent sum of position functions.
    
    $$ \frac{x}{E} = v_0 + \sum_{i = 1}^n v_i E  $$
    """
    short_name = 'pca_energy_linear'
    uses_basis = True

    def __init__(self,
                 energies: ArrayLike,
                 positions: ArrayLike,
                 responses: ArrayLike,
                 basis: ArrayLike,
                 **kwargs
                 ) -> None:
        """Initialize, determining the energy least squared dependence."""
        super().__init__(energies, positions, responses, **kwargs)

        if np.shape(basis)[1:] != np.shape(positions)[1:]:
            raise ValueError('Basis must match position size.')

        self.basis = np.array(basis, dtype=np.float64)
        basis_flat = self.basis.reshape((self.basis.shape[0], -1))

        # Check that the basis is orthogonal
        if not np.allclose(basis_flat @ basis_flat.T,
                           np.identity(basis_flat.shape[0])):
            raise ValueError("Basis must have independent, unit components.")

        if not np.allclose(self.positions,
                           np.meshgrid(self.x[:, 0],
                                       self.y[0, :],
                                       indexing='ij')):
            raise ValueError('Calibrations must use an even grid.')

        self.position_interp = \
            RegularGridInterpolator([self.x[:, 0], self.y[0, :]],
                                    np.moveaxis(self.basis, 0, -1),
                                    method='cubic',
                                    bounds_error=False)

        n_pos = self.positions.shape[1]*self.positions.shape[2]
        n_comp = self.basis.shape[0]
        n_det = self.responses.shape[-1]
        n_e = len(self.energies)

       # Project the response provided into the basis provided
        response_flipped = pca.flip_response(self.responses)
        projected = np.matmul(
            self.basis.reshape((n_comp, n_pos)),
            np.moveaxis(response_flipped, 0, -2).reshape(n_pos, -1)
            ).reshape(n_comp*n_e, n_det)

        self.weights = np.empty((2, n_comp, n_det), dtype=np.float64)
        for det_i in range(n_det):
            # Use linear least squares to find the energy dependence.
            projected = np.matmul(
                    self.basis.reshape((n_comp, n_pos)),
                    response_flipped[:, :, :, det_i].reshape(-1, n_pos).T
                )
            for comp_j in range(n_comp):
                energy_matrix = np.vstack((np.ones_like(self.energies),
                               self.energies)).T
                self.weights[:, comp_j, det_i] = \
                    np.linalg.lstsq(energy_matrix,
                                    projected[comp_j] / self.energies,
                                    rcond=None)[0]

    @override
    def values(self, energy, x, y):
        shape = np.shape(energy)
        assert (np.shape(x) == shape) and (np.shape(y) == shape)

        energy = np.array(energy, dtype=np.float64)
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        basis = np.array(
            self.position_interp(
                np.moveaxis(pca.flip_position([x, y]), 0, -1)),
            dtype=np.float64
        )

        predicted_response = np.empty((*shape, 4), dtype=np.float64)

        for i in range(4):
            projected_response = np.array(
                np.outer(energy, self.weights[0, :, i])\
                + np.outer(energy**2, self.weights[1, :, i]),
                dtype=np.float64
            ).reshape((*shape, basis.shape[-1]))

            predicted_response[..., i] = \
                np.vecdot(basis[..., i, :],
                          projected_response,
                          axis=-1) # type: ignore

        return predicted_response


methods = {x.short_name: x for x in
           [LinearInterpolation,
            BasisFilteredInterpolation,
            PCAEnergyInterpolation]}
