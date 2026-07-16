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
from unittest import mock
import pytest
import numpy as np
from x_ray_imager_bagriff.response_interpolation import (
    plot, Interpolation,
    LinearInterpolation, PCAEnergyInterpolation
)

# Tests will need to check internal functions
# pylint: disable=protected-access
# For pytest fixtures without warnings:
# pylint: disable=redefined-outer-name


class MockDiagnostic(plot.GenericResponseDiagnostic):
    """Mocks a ResponseDiagnostic instance without plotting or saving."""
    def plot_diagnostic(self, *args, **kwargs):
        pass
    def savefig(self, *args, **kwargs):
        pass


@pytest.fixture(name="interpolation_data")
def minimal_data():
    """Provides dummy data for Interpolation tests."""
    init_energies = np.array([10.])
    init_positions = np.arange(8).reshape(2, 2, 2)
    init_responses = np.array([[
        [[10.0, 10.0, 10.0, 10.0], # for (0,0,0)
         [11.0, 11.0, 11.0, 11.0]], # for (0,0,1)
        [[12.0, 12.0, 12.0, 12.0], # for (0,1,0)
         [13.0, 13.0, 13.0, 13.0]]]] # for (0,1,1)
    )

    interp_responses = np.array([[
        [[10.5, 10.0, 10.5, 10.0], # for (0,0,0)
         [11.5, 11.0, 11.5, 11.0]], # for (0,0,1)
        [[12.5, 12.0, 12.5, 12.0], # for (0,1,0)
         [13.5, 13.0, 13.5, 13.0]]]] # for (0,1,1)
    )
    return (init_energies, init_positions, init_responses, interp_responses)


@pytest.fixture
def realistic_data():
    energy = np.linspace(50, 500, 19)
    x = np.linspace(-70, 70, 29)
    y = np.linspace(-70, 70, 29)
    e_mesh, x_mesh, y_mesh = np.meshgrid(energy, x, y, indexing='ij')
    response = np.repeat(
        np.expand_dims((e_mesh / 25) * ((x_mesh + 100) + 2*(y_mesh + 100)), 3),
        4, axis=3)
    
    sampled_energy = energy[::2]
    sampled_position = np.array(np.meshgrid(x[::2], y[::2], indexing='ij'))
    sampled_response = response[::2, ::2, ::2, :]
    return (e_mesh, x_mesh, y_mesh, response,
            sampled_energy, sampled_position, sampled_response)

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
    (init_energies, init_positions,
     init_responses, interp_responses) = interpolation_data

    with mock.patch.object(Interpolation, 'values',
                           return_value=interp_responses):

        interpolator = Interpolation(init_energies,
                                     init_positions,
                                     init_responses)

        # Call validate
        max_error, mean_error = interpolator.validate()

        assert max_error == pytest.approx(0.5)
        assert mean_error == pytest.approx(0.25)

def test_interpolation_init():
    """Check that init checks the array checks shape and calls diagnostics."""
    # Dummy data for Interpolation.__init__
    energies = np.array([10., 20.])
    positions = np.array([[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]])
    responses = np.zeros((2, 2, 2, 4))

    with (mock.patch.object(Interpolation, '_check_shape') as mock_check_shape):
        Interpolation(energies, positions, responses)
        mock_check_shape.assert_called_once_with(energies,
                                                 positions,
                                                 responses)


def test_interpolation_diagnostics(interpolation_data):
    """Checks that correct values are passed to each diagnostic plot."""
    (init_energies, init_positions,
     init_responses, interp_responses) = interpolation_data

    # Create mock diagnostic objects
    diagnostics = {s: MockDiagnostic() for s in
                   ['input_diagnostic',
                   'output_diagnostic',
                   'error_diagnostic']}

    with (
        mock.patch.object(Interpolation, 'values',
                          return_value=interp_responses),
        mock.patch.object(diagnostics['input_diagnostic'],
                          'plot_diagnostic') as input_plot,
        mock.patch.object(diagnostics['input_diagnostic'],
                          'savefig') as input_save,
        mock.patch.object(diagnostics['output_diagnostic'],
                          'plot_diagnostic') as output_plot,
        mock.patch.object(diagnostics['output_diagnostic'],
                          'savefig') as output_save,
        mock.patch.object(diagnostics['error_diagnostic'],
                          'plot_diagnostic') as error_plot,
        mock.patch.object(diagnostics['error_diagnostic'],
                          'savefig') as error_save
    ):
        # Initialize Interpolation with diagnostics
        interp = Interpolation(init_energies,
                               init_positions,
                               init_responses,
                               input_diagnostic=diagnostics['input_diagnostic'])

        interp.validate(output_diagnostic=diagnostics['output_diagnostic'],
                        error_diagnostic=diagnostics['error_diagnostic'])

        # Each diagnostic is called once for each energy in init_energies
        for plot, save in [(input_plot, input_save),
                           (output_plot, output_save),
                           (error_plot, error_save)]:
            assert plot.call_count == len(init_energies)
            assert save.call_count == len(init_energies)


def test_linear_interpolation(realistic_data):
    """Tests CubicInterpolation accurately interpolates a linear function."""
    (e_mesh, x_mesh, y_mesh, response,
     sampled_energy, sampled_position, sampled_response) = realistic_data

    interpolator = LinearInterpolation(sampled_energy,
                                       sampled_position,
                                       sampled_response)

    interpolated = interpolator.values(e_mesh, x_mesh, y_mesh)
    assert pytest.approx(response, 1e-3) == interpolated


def test_pca_interpolation(realistic_data):
    (e_mesh, x_mesh, y_mesh, response,
     sampled_energy, sampled_position, sampled_response) = realistic_data

    basis = [np.ones_like(x_mesh[0, ::2, ::2]),
             x_mesh[0, ::2, ::2],
             y_mesh[0, ::2, ::2]]
    basis = [x / np.linalg.norm(x) for x in basis]

    interpolator = PCAEnergyInterpolation(sampled_energy,
                                          sampled_position,
                                          sampled_response,
                                          basis)

    interpolated = interpolator.values(e_mesh, x_mesh, y_mesh)
    assert pytest.approx(response, 1e-3) == interpolated
