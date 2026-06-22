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

"""Test the interpolation methods."""
import pytest
import numpy as np
from unittest import mock
from x_ray_imager_bagriff.response_interpolation import (
    plot, CubicInterpolation, Interpolation
)

# Tests will need to check internal functions
# pylint: disable=protected-access
# For pytest fixtures without warnings:
# pylint: disable=redefined-outer-name


class MockDiagnostic(plot.GenericResponseDiagnostic):
    def plot_diagnostic(self, *args, **kwargs):
        pass
    def savefig(self, *args, **kwargs):
        pass


@pytest.fixture(name="interpolation_data")
def interpolation_data():
    """Provides dummy data for Interpolation tests."""
    init_energies = np.array([10., 20.])
    init_positions = np.array([[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]])
    init_responses = np.zeros((2, 2, 2, 4))

    energy = 10.0
    positions_for_validate = np.array([
        [[0.1, 1.1],
         [2.1, 3.1]],
        [[4.1, 5.1],
         [6.1, 7.1]]
    ]) # shape (2, 2, 2)

    responses_for_validate = np.array([
        [[10., 10., 10., 10.], [11., 11., 11., 11.]],
        [[12., 12., 12., 12.], [13., 13., 13., 13.]]]
    ) # shape (2, 2, 4) - validate expects (n_x, n_y, n_detectors)

    interp_responses = np.array([
        [[10.5, 10.0, 10.5, 10.0], # for (0,0)
         [11.5, 11.0, 11.5, 11.0]], # for (0,1)
        [[12.5, 12.0, 12.5, 12.0], # for (1,0)
         [13.5, 13.0, 13.5, 13.0]]] # for (1,1)
    )
    return (init_energies, init_positions, init_responses, energy,
            positions_for_validate, responses_for_validate,
            interp_responses)


def test_interpolation_check_shape():
    """Checks that Interpolation._check_shape() checks arrays correctly."""
    good_energy = np.arange(5)
    good_positions = np.array(np.meshgrid(np.arange(8),
                                          np.arange(9),
                                          indexing='ij'))
    good_responses = np.zeros((5, 8, 9, 4))

    # Test good shapes (no error)
    Interpolation._check_shape(good_energy, good_positions, good_responses)

    # Test bad energy shape
    with pytest.raises(ValueError, match='energy array is the wrong shape'):
        Interpolation._check_shape(np.zeros((2, 5)),
                                   good_positions,
                                   good_responses)

    # Test bad positions shape (not 3D)
    with pytest.raises(ValueError, match='positions array is the wrong shape'):
        Interpolation._check_shape(good_energy,
                                   np.zeros([2, 3]),
                                   good_responses)

    # Test bad positions shape (first dim not 2)
    with pytest.raises(ValueError, match='positions array is the wrong shape'):
        Interpolation._check_shape(good_energy,
                                   np.zeros((1,8,9)),
                                   good_responses)

    # Test bad responses shape (not 4D)
    with pytest.raises(ValueError, match='responses array is the wrong shape'):
        Interpolation._check_shape(good_energy,
                                   good_positions,
                                   np.zeros((3, 2, 2)))

    # Test bad responses shape (dims don't match)
    with pytest.raises(ValueError, match='responses array is the wrong shape'):
        Interpolation._check_shape(good_energy,
                                   good_positions,
                                   np.zeros((5, 9, 8, 4)))


def test_interpolation_validate(interpolation_data):
    """Checks that Interpolation.validate() calculates errors correctly."""
    (init_energies, init_positions, init_responses, energy,
     positions_for_validate, responses_for_validate,
     interp_responses) = interpolation_data

    with mock.patch.object(Interpolation, 'values',
                          return_value=interp_responses):

        interpolator = Interpolation(init_energies,
                                     init_positions,
                                     init_responses)

        # Call validate
        max_error, mean_error = interpolator.validate(
            energy,
            positions_for_validate,
            responses_for_validate
        )

        assert max_error == pytest.approx(0.5)
        assert mean_error == pytest.approx(0.25)

def test_interpolation_init():
    """Check that init checks the array checks shape and calls diagnostics."""
    # Dummy data for Interpolation.__init__
    energies = np.array([10., 20.])
    positions = np.array([[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]])
    responses = np.zeros((2, 2, 2, 4))

    with (mock.patch.object(Interpolation, '_check_shape') as mock_check_shape,
          mock.patch.object(Interpolation, 'validate',
                            return_value=(0., 0.)) as mock_validate):
        Interpolation(energies, positions, responses)
        mock_check_shape.assert_called_once_with(energies,
                                                 positions,
                                                 responses)
        assert mock_validate.call_count == len(energies)


def test_interpolation_diagnostics(interpolation_data):
    """Checks that correct values are passed to each diagnostic plot."""
    (init_energies, init_positions, init_responses,
     _, _, _, interp_responses) = interpolation_data

    # Create mock diagnostic objects
    input_diag = MockDiagnostic()
    output_diag = MockDiagnostic()
    error_diag = MockDiagnostic()

    with (
        mock.patch.object(Interpolation, 'values',
                          return_value=interp_responses),
        mock.patch.object(input_diag, 'plot_diagnostic') as input_plot,
        mock.patch.object(input_diag, 'savefig') as input_save,
        mock.patch.object(output_diag, 'plot_diagnostic') as output_plot,
        mock.patch.object(output_diag, 'savefig') as output_save,
        mock.patch.object(error_diag, 'plot_diagnostic') as error_plot,
        mock.patch.object(error_diag, 'savefig') as error_save
    ):
        # Initialize Interpolation with diagnostics
        Interpolation(init_energies,
                      init_positions,
                      init_responses,
                      input_diagnostic=input_diag,
                      output_diagnostic=output_diag,
                      error_diagnostic=error_diag)

        # Each diagnostic is called once for each energy in init_energies
        for plot, save in [(input_plot, input_save),
                           (output_plot, output_save),
                           (error_plot, error_save)]:
            assert plot.call_count == len(init_energies)
            assert save.call_count == len(init_energies)


def test_cubic_interpolation():
    energy = np.linspace(50, 500, 19)
    x = np.linspace(-70, 70, 29)
    y = np.linspace(-70, 70, 29)
    E, X, Y = np.meshgrid(energy, x, y, indexing='ij')
    response = np.repeat(
        np.expand_dims((E / 25) * ((X + 100) + 2*(Y + 100)), 3),
        4, axis=3)

    sampled_energy = energy[::2]
    sampled_position = np.array(np.meshgrid(x[::2], y[::2]))
    sampled_response = response[::2, ::2, ::2, :]
    interpolator = CubicInterpolation(sampled_energy,
                                      sampled_position,
                                      sampled_response)

    assert pytest.approx(response, 1e-3) == interpolator.values(E, X, Y)
