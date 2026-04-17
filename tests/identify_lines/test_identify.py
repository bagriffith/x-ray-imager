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
from sklearn.base import ClusterMixin
from x_ray_imager_bagriff.identify_lines import (
    find_centers, match_energy, source_identify_all,
    SourceParams
)


class MockCluster(ClusterMixin):
    """Mock scikit-learn clustering where labels are prespecified."""
    def __init__(self, cluster_labels) -> None:
        self.labels_ = cluster_labels
        super().__init__()

    def fit(self, x, **kwargs):
        """Fit, but do nothing."""
        _ = x
        _ = kwargs


def test_source_identify_all():
    """Test source line identification with preidentified clusters."""
    n_points = 10_000
    means = (256, 512)
    np.random.seed(0)  # Keep the set consistent between tests
    example_set = np.concat(
        [np.random.poisson(x, size=(n_points, 4)) for x in means]
    )

    cluster_labels = np.repeat(np.arange(len(means)), n_points)
    mock_cluster = MockCluster(cluster_labels)

    source = SourceParams(means)

    centers = source_identify_all(example_set, mock_cluster,
                                  source, gain_range=(1, 8))

    for mean, center in zip(means, centers):
        assert center == pytest.approx([mean]*4, 0.05)


def test_find_centers():
    """Tests `find_centers()` for a Poisson distribution."""
    n_points = 10_000
    mean = 256
    np.random.seed(0)  # Keep the set consistent between tests
    example_set = np.random.poisson(mean, size=(n_points, 4))
    example_labels = np.zeros(n_points, dtype=int)

    centers = find_centers(example_set, example_labels)

    assert centers == pytest.approx(np.full((1, 4), mean), 0.05)


def test_match_energy():
    """Tests `match_energy()` for evenly spaced centers."""
    example_centers = np.transpose([np.arange(100., 1000., 100.)]*4)
    example_energies = np.array([30., 80.])

    idx, g = match_energy(example_centers, example_energies, (20, 80))

    assert np.all(idx == [2, 7])
    assert g == pytest.approx(40.)
