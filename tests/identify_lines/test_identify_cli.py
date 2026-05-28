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

from io import StringIO
from os import chdir
from click.testing import CliRunner
import numpy as np
import pandas as pd
import pytest
from x_ray_imager_bagriff.identify_lines import SourceParams
from x_ray_imager_bagriff.identify_lines._cli import cli

# For pytest fixtures
# pylint: disable=redefined-outer-name


@pytest.fixture
def example_events(tmp_path):
    """Create a CSV file with rows of imager events.
    
    Example follows the two energies of Am241 using a gain of 4/keV.
    """
    n_points = 2_000
    means = [4.0 * x*np.arange(3, 7) / 18
             for x in SourceParams.get_source('Am241').energies]
    np.random.seed(0)  # Keep the set consistent between tests
    example_set = np.concat(
        [np.random.poisson(np.repeat([x], n_points, axis=0))
         for x in means]
        + [np.random.randint(0, 256, size=(n_points, 4))]  # Background events
    )
    data_file = tmp_path / 'events.csv'
    np.savetxt(data_file, example_set,
               delimiter=',', fmt='%d',
               header='#T1,T2,T3,T4')
    return [data_file, np.array(means)]


@pytest.fixture
def example_points(example_events, tmp_path):
    """Create a CSV list of calibrations points with for `example_events`.
    
    Only contains a header and one calibration point, set at (-10, 10).
    """
    data_file, means = example_events

    points_csv = tmp_path / 'points.csv'
    with open(points_csv, 'w', encoding='utf-8') as f:
        f.write('"x","y","csv_path"\n'
                f'-10.0,10.0,"{data_file}"')

    return [points_csv, means]


def test_identify_point_cli(example_events, tmp_path):
    """Tests that the cli point command correctly loads."""
    events_csv, means = example_events
    runner = CliRunner()
    chdir(tmp_path)
    runner.isolated_filesystem(tmp_path)
    result = runner.invoke(cli, ['point', str(events_csv), 'Am241',
                                 '--gain', '1.0', '8.0',
                                 '--diagnostic', 'full'])
    assert result.exit_code == 0
    response = np.loadtxt(StringIO(result.stdout), delimiter=',')
    assert response.shape == means.shape
    assert response == pytest.approx(means, 0.1)
    assert (tmp_path / 'Am241-diagnostic.png').exists()


def test_identify_grid_cli(example_points, tmp_path):
    points_csv, means = example_points
    runner = CliRunner()
    runner.isolated_filesystem(tmp_path)
    out_path = tmp_path / 'out.csv'
    result = runner.invoke(cli, ['grid', str(points_csv), 'Am241',
                                 '--gain', '1.0', '8.0',
                                 '--output', str(out_path)])
    print(result.output)
    assert result.exit_code == 0
    df = pd.read_csv(out_path)
    assert df['x'].to_numpy() == pytest.approx([-10])
    assert df['y'].to_numpy() == pytest.approx([10])

    result_means = np.array(
        [[df[f'{energy:.1f} keV T{tube}'][0] for tube in range(4)]
         for energy in SourceParams.get_source('Am241').energies]
         )
    assert result_means == pytest.approx(means, 0.1)
