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
"""Test PCA untillites."""

import pytest
import numpy as np
from x_ray_imager_bagriff.response_interpolation.pca import (
    flip_response, flip_position, form_basis
)

# For pytest fixtures without warnings:
# pylint: disable=redefined-outer-name


@pytest.fixture
def example_data():
    """Construct a dataset out of vectors clearly identifiable to PCA."""
    # Construct an independent basis.
    np.random.seed(1)
    expected_basis = np.triu(np.ones((8, 9))) / np.sqrt(9)

    # Make each component independent from the previous ones.
    for i in range(1, np.shape(expected_basis)[0]):
        for j in range(i):
            expected_basis[i] -= \
                expected_basis[j] * np.dot(expected_basis[j],
                                           expected_basis[i])
        expected_basis[i] /= np.linalg.norm(expected_basis[i])

    # Use a random portion of each with a weight for each 10x smaller.
    data = np.diag(np.sqrt(9) * np.pow(10.0, -np.arange(8))/1.5) @ expected_basis
    proj = np.random.uniform(1, 2, size=8*8).reshape((8, 8))
    data = (proj @ data).reshape((2, 4, 3, 3))

    # Shape it the same way as a response array.
    data = np.moveaxis(data, 1, -1)
    data = flip_response(data)
    return data, expected_basis


def test_flip_response():
    """Test that ``flip_response`` correctly flips x and y components."""
    x, y = np.meshgrid(np.arange(3),
                       np.arange(3),
                       indexing='ij')

    x = np.repeat(x[..., np.newaxis], 4, axis=-1)
    y = np.repeat(y[..., np.newaxis], 4, axis=-1)

    x_flip = flip_response(x)
    y_flip = flip_response(y)

    for i in range(3):
        assert np.all(x_flip[:, i, 0] == [2, 1, 0])
        assert np.all(x_flip[:, i, 1] == [2, 1, 0])
        assert np.all(x_flip[:, i, 2] == [0, 1, 2])
        assert np.all(x_flip[:, i, 3] == [0, 1, 2])

        assert np.all(y_flip[i, :, 0] == [0, 1, 2])
        assert np.all(y_flip[i, :, 1] == [2, 1, 0])
        assert np.all(y_flip[i, :, 2] == [2, 1, 0])
        assert np.all(y_flip[i, :, 3] == [0, 1, 2])


def test_flip_position():
    """Test ``flip_position`` correctly mirrors x/y values."""
    x, y = np.meshgrid(np.arange(3),
                       np.arange(3),
                       indexing='ij')

    flipped = flip_position([x, y])

    assert flipped[0, :, :, 0] == pytest.approx(-x)
    assert flipped[1, :, :, 0] == pytest.approx(y)
    assert flipped[0, :, :, 1] == pytest.approx(-x)
    assert flipped[1, :, :, 1] == pytest.approx(-y)
    assert flipped[0, :, :, 2] == pytest.approx(x)
    assert flipped[1, :, :, 2] == pytest.approx(-y)
    assert flipped[0, :, :, 3] == pytest.approx(x)
    assert flipped[1, :, :, 3] == pytest.approx(y)


def test_form_basis(example_data):
    """Test that basis vectors used to create a test set are identified."""
    data, expected_basis = example_data

    predicted_basis, singular_values = form_basis(data)

    for i in range(8):
        # PCA should produce the expected unit vectors, down to a sign.
        # Their dot product should therefore be +/- 1.
        assert np.abs(np.dot(predicted_basis[i].flatten(),
                             expected_basis[i].flatten())) == \
                                pytest.approx(1, 0.01)

    # Verify each component gets much less significant
    assert np.all(singular_values[:-1] / singular_values[1:] > 5.0)
