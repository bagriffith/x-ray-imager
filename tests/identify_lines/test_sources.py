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
from x_ray_imager_bagriff.identify_lines\
    import SourceParams


def test_get_source():
    """Create and retreive a source."""
    example_energies = np.linspace(100, 400, 4) - 0.001
    source = SourceParams(example_energies, 'Ex123')
    source_fetched = SourceParams.get_source('Ex123')

    assert source == source_fetched
    assert source_fetched.energies == pytest.approx(example_energies)


def test_filter():
    """Test SourceParams excludes energies correctly.

    Energies less than half the min and twice the max should be excluded.
    """
    # Slightly above the integer steps to avoid points at the filter boundary.
    example_energies = np.linspace(100, 400, 4) - 0.001
    example_points = np.transpose([np.arange(10, 1000, 10)]*4)
    # For gain = 4.0, points in the range should be:
    filtered_correct = np.transpose([np.arange(50., 800., 10.)]*4)

    source = SourceParams(example_energies)
    source_filter = source.get_filter(example_points, gain=4.0)
    assert example_points[source_filter] == pytest.approx(filtered_correct)
