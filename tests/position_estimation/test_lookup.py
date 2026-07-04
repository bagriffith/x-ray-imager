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

import pytest
import numpy as np
from x_ray_imager_bagriff.position_estimation import TreeLookup

# For pytest fixtures
# pylint: disable=redefined-outer-name


@pytest.fixture
def response():
    """Create an example detector response."""
    energy = np.linspace(50, 500, 19)
    x = np.linspace(-70, 70, 29)
    y = np.linspace(-70, 70, 29)
    e_mesh, x_mesh, y_mesh = np.meshgrid(energy, x, y, indexing='ij')

    value = (2/140) * e_mesh[..., np.newaxis] * (
                x_mesh[..., np.newaxis] @ [[1, 1, -1, -1]] + 70
                + 2 * (y_mesh[..., np.newaxis] @ [[-1, 1, 1, -1]] + 70))

    return [value, e_mesh, np.array([x_mesh, y_mesh])]


def test_lookup_tree(response):
    """Tests the tree lookup position estimator can identify correctly."""
    channels, energy, position = response
    pos_est = TreeLookup(channels, energy, position)
    channels_test = channels[1:-1:2, 1:-1:2, 1:-1:2, :]\
        .reshape((-1, channels.shape[-1]))
    energy_test = energy[1:-1:2, 1:-1:2, 1:-1:2]\
        .reshape((-1))
    position_test = position[:, 1:-1:2, 1:-1:2, 1:-1:2]\
        .reshape((2, -1))
    estimation = pos_est.get_value(channels_test)
    assert estimation[0] == pytest.approx(energy_test, abs=0.5)
    assert estimation[1:] == pytest.approx(position_test, abs=2.5)
