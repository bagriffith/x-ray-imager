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

"""Test PointEstimator, the base estimator class."""
import pytest
import numpy as np
from x_ray_imager_bagriff.position_estimation import (
    PointEstimator
)

# For pytest fixtures without warnings:
# pylint: disable=redefined-outer-name


@pytest.fixture
def example_response():
    """Creates example energy, position, and response arrays."""
    energies, x, y = np.meshgrid(np.arange(3),
                                 np.arange(5),
                                 np.arange(5),
                                 indexing='ij')
    positions = np.array([x, y])

    response = np.array([energies * (positions[0] + 2 * positions[1])
                         for i in range(4)])

    response = np.moveaxis(response, 0, -1)
    return response, energies, positions


def test_estimator_init_shape(example_response):
    """Test PointEstimator raises exceptions for incorrect shapes."""
    response, energies, positions = example_response
    response_1d = np.arange(300)
    with pytest.raises(ValueError,
                       match="Provided response should be an "
                              "array of measurements, not 1-D."):
        PointEstimator(response_1d, energies, positions)

    energies_wrong = np.ones((5, 15))  # Mismatch energy shape
    with pytest.raises(ValueError,
                       match="Mismatch between response and energies shape"):
        PointEstimator(response, energies_wrong, positions)

    positions_1d = np.arange(75)  # 1-D positions
    with pytest.raises(ValueError,
                       match="The positions array must have "
                             "two spacial dimensions,"):
        PointEstimator(response, energies, positions_1d)

    positions_wrong = np.ones((2, 5, 5))  # Mismatch energy shape
    with pytest.raises(ValueError,
                       match="Mismatch between response and positions shape"):
        PointEstimator(response, energies, positions_wrong)


def test_estimator_save(tmp_path, example_response):
    """Test PointEstimator can be saved and reloaded accurately."""
    response, energies, positions = example_response
    estimator = PointEstimator(response, energies, positions)

    file_path = tmp_path / "estimator.npz"
    estimator.save_to(file_path)
    loaded_estimator = PointEstimator.load_from(file_path)

    np.testing.assert_array_equal(estimator.response,
                                  loaded_estimator.response)
    np.testing.assert_array_equal(estimator.points[0],
                                  loaded_estimator.energies)
    np.testing.assert_array_equal(estimator.positions,
                                  loaded_estimator.positions)


def test_get_values_shape(example_response):
    """Test PointEstimator.get_values returns the expected shape."""
    response, energies, positions = example_response
    estimator = PointEstimator(response, energies, positions)

    observations_1d = np.arange(8).reshape((2, 4))
    observations_2d = np.arange(24).reshape((2, 3, 4))

    # Without error output.
    estimation_1d = estimator.get_value(observations_1d)
    assert np.shape(estimation_1d) == (3, 2)  # (energy, x, y)

    estimation_2d = estimator.get_value(observations_2d)
    assert np.shape(estimation_2d) == (3, 2, 3)

    # With error output
    estimation_1d_with_error, error_1d = \
        estimator.get_values_with_error(observations_1d)
    assert estimation_1d_with_error.shape == (3, 2)
    assert error_1d.shape == (3, 2)

    # Test with 2D observation and return_error=True
    estimation_2d_with_error, error_2d = \
        estimator.get_values_with_error(observations_2d)
    assert estimation_2d_with_error.shape == (3, 2, 3)
    assert error_2d.shape == (3, 2, 3)
