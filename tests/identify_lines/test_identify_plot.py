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
from x_ray_imager_bagriff.identify_lines import plot


def test_full_diagnostic(tmp_path):
    n_points = 1000
    means = (64, 128)
    np.random.seed(0)  # Keep the set consistent between tests
    example_set = np.concat(
        [np.random.randint(0, 256, size=(n_points, 4))] +
        [np.random.poisson(x, size=(n_points, 4)) for x in means]
    )
    cluster_labels = np.repeat(np.arange(len(means) + 1), n_points)

    fig = plot.FullDiagnostic()
    fig.plot_diagnostic(example_set, cluster_labels)
    plot_path = tmp_path / 'diagnostic.pdf'
    fig.savefig(plot_path)
    assert plot_path.exists()
