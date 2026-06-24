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

"""Tests for the identify_lines CLI."""
from io import StringIO
from unittest import mock
from os import chdir
from click.testing import CliRunner
import numpy as np
import pandas as pd
import pytest
from x_ray_imager_bagriff.identify_lines import SourceParams
from x_ray_imager_bagriff.identify_lines.plot import FullDiagnostic
from x_ray_imager_bagriff.identify_lines._cli import cli

# For pytest fixtures without warnings:
# pylint: disable=redefined-outer-name


@pytest.fixture
def example_events(tmp_path):
    """Create a CSV file with rows of imager events.
    
    Example follows the two energies of Am241 using a gain of 4/keV.
    """
    n_points = 2_000
    source = SourceParams.get_source('Am241')
    means = np.array([4.0 * x*np.arange(3, 7) / 18 for x in source.energies])
    np.random.seed(0)  # Keep the set consistent between tests
    example_set = np.concat(
        [np.random.poisson(np.repeat([x], n_points, axis=0))
         for x in means]
        + [np.random.randint(0, 256, size=(n_points, 4))]  # Background events
    )
    data_file = tmp_path / 'events.csv'
    np.savetxt(data_file, example_set,
               delimiter=',', fmt='%d',
               header='#d1,d2,d3,d4')

    return [data_file, means, source]


@pytest.fixture
def example_points(example_events, tmp_path):
    """Create a CSV list of calibrations points with for `example_events`.
    
    Only contains a header and one calibration point, set at (-10, 10).
    """
    data_file, means, source = example_events

    points_csv = tmp_path / 'points.csv'
    with open(points_csv, 'w', encoding='utf-8') as f:
        f.write('"x","y","csv_path"\n'
                f'-10.0,10.0,"{data_file}"')

    return [points_csv, means, source]


def test_identify_single_cli(example_events, tmp_path):
    """Tests that the cli single command correctly loads."""
    events_csv, means, source = example_events
    runner = CliRunner()
    chdir(tmp_path)
    runner.isolated_filesystem(tmp_path)
    with (mock.patch('x_ray_imager_bagriff.identify_lines._cli.find_lines',
                     return_value=means)
          as mock_find_lines):
        result = runner.invoke(cli, ['single', str(events_csv), source.name,
                                     '--gain', '1.0', '8.0',
                                     '--diagnostic', 'full'])
        mock_find_lines.assert_called_once()

    assert result.exit_code == 0
    response = np.loadtxt(StringIO(result.stdout), delimiter=',')
    assert response[:, 1:] == pytest.approx(means, abs=1e-5)
    assert response[:, 0] == pytest.approx(source.energies, abs=1e-1)


def test_identify_multiple_cli(example_points, tmp_path):
    """Tests that the cli multiple loads a listed csv and identifies lines."""
    points_csv, means, source = example_points
    runner = CliRunner()
    runner.isolated_filesystem(tmp_path)
    out_path = tmp_path / 'out.csv'
    with (mock.patch('x_ray_imager_bagriff.identify_lines._cli.find_lines',
                     return_value=means)
          as mock_find_lines):
        result = runner.invoke(cli, ['multiple', str(points_csv), source.name,
                                    '--gain', '1.0', '8.0',
                                    '--output', str(out_path)])
        mock_find_lines.assert_called_once()  # Only one set is provided

    print(result.output)
    assert result.exit_code == 0
    df = pd.read_csv(out_path)
    assert df['x'].to_numpy() == pytest.approx([-10])
    assert df['y'].to_numpy() == pytest.approx([10])

    # Construct array from columns
    result_means = np.array(
        [[df[f'{energy:.1f} keV d{i}'][0] for i in range(4)]
         for energy in source.energies]
         )
    assert result_means == pytest.approx(means, 0.1)
