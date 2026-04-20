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
from x_ray_imager_bagriff.response_interpolation import CubicInterpolation


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
