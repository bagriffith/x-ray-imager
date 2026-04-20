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

import numpy as np
import pytest
from x_ray_imager_bagriff.position_estimation import anger_basis, AngerSimple


def test_anger_basis():
    points = np.array([[0, 100, 0, 0],
                       [0, 0, 50, 50],
                       [0, 50, 50, 0],
                       [25, 25, 25, 25]],
                      dtype=np.float64)

    expected_amplitude = pytest.approx(np.full(4, 100, np.float64))
    expected_x = pytest.approx(np.array([1, -1, 0, 0], dtype=np.float64))
    expected_y = pytest.approx(np.array([1, 0, 1, 0], dtype=np.float64))

    amplitude, x, y = anger_basis(points)
    assert expected_amplitude == amplitude
    assert expected_x == x
    assert expected_y == y


def test_anger_estimator():
    energies = np.array([60, 480], dtype=np.float64)
    positions = np.array([[-70, -70], [70, 70]], dtype=np.float64)
    channels = np.array([[0, 0, 0, 128], [0, 1024, 0, 0]], dtype=np.long)

    estimator = AngerSimple(channels, energies, positions)
    e, x, y = estimator.get_value(channels)
    assert pytest.approx(energies) == e
    assert pytest.approx(positions[:, 0]) == x
    assert pytest.approx(positions[:, 1]) == y
