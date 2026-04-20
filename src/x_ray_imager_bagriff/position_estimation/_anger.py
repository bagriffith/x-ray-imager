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
from x_ray_imager_bagriff.position_estimation import PointEstimator


def anger_basis(points):
    # TODO Test shape
    amplitude = np.sum(points, axis=1, dtype=np.float64)
    amplitude[amplitude < 1.] = np.nan

    x = np.dot(points, [1, 1, -1, -1]) / amplitude
    y = np.dot(points, [-1, 1, 1, -1]) / amplitude

    return amplitude, x, y


class AngerSimple(PointEstimator):
    short_name = 'anger'

    def __init__(self, channels, energies, positions):
        amp, x, y = anger_basis(channels)

        self.a_x = np.max(np.abs(positions)) / np.max(np.abs(x))
        self.a_y = np.max(np.abs(positions)) / np.max(np.abs(y))
        self.a_e = np.mean(amp/energies)

    def get_value(self, channels, return_error=False):
        amp, x, y = anger_basis(channels)
        gain = 1/2

        d_amp = gain*np.sqrt(amp)
        d_s = d_amp/amp

        e = amp / self.a_e
        de = d_amp / self.a_e

        x = self.a_x * x
        dx = self.a_x*d_s
        y = self.a_y * y
        dy = self.a_y*d_s

        if return_error:
            return (e, x, y), (de, dx, dy)
        else:
            return (e, x, y)
