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
from x_ray_imager_bagriff.position_estimation import (
    LookupGradError,
    TreeLookup
)

# For pytest fixtures
# pylint: disable=redefined-outer-name


@pytest.fixture
def response():
    energy = np.linspace(50, 500, 19)
    x = np.linspace(-70, 70, 29)
    y = np.linspace(-70, 70, 29)
    E, X, Y = np.meshgrid(energy, x, y, indexing='ij')
    energy = np.arange(100, 1000, 10)
    value = np.repeat(np.expand_dims(E, axis=3), 4, axis=3) * \
        (np.repeat(np.expand_dims(X, axis=3), 4, axis=3)*[1, 1, -1, -1]
         + np.repeat(np.expand_dims(Y, axis=3), 4, axis=3)*[-1, 1, 1, -1]
         + 140)

    return [value, E, np.moveaxis(np.array([X, Y]), 0, -1)]


def test_lookup_tree(response):
    pos_est = TreeLookup(*response)
    channels, energy, position = response
    channels = channels.reshape((-1, 4))
    energy = energy.reshape((-1))
    position = position.reshape((-1, 2))
    e_out, x_out, y_out = pos_est.get_value(channels)
    assert e_out == pytest.approx(energy)
    assert x_out == pytest.approx(position[:, 0])
    assert y_out == pytest.approx(position[:, 1])
