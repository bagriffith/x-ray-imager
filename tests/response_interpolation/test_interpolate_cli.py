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

"""Test the interpolation CLI."""
import csv
import os
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from x_ray_imager_bagriff.response_interpolation import _cli

# For pytest fixtures without warnings:
# pylint: disable=redefined-outer-name


@pytest.fixture
def example_files(tmp_path):
    """Creates a set of calibration CSV files."""
    energies = np.array([20., 60., 250., 500.])
    n_pts = 5
    positions = np.array(np.meshgrid(np.linspace(-70, 70, n_pts),
                                     np.linspace(-70, 70, n_pts),
                                     indexing='ij')).reshape((2, n_pts*n_pts))

    # Simple output data
    response = np.empty((len(energies), n_pts*n_pts, 4), dtype=np.double)
    response_csv = list()
    for i, energy in enumerate(energies):
        for det_n in range(4):
            response[i, :, det_n] = 0.125 * energy * (det_n + 4) \
                * (positions[0] + 70. + 1.5*(positions[1] + 70.))

        # Write the example line to a file
        csv_path = tmp_path / (f'{energy:.1f} keV'.replace('.', '_')+'.csv')
        response_csv.append(str(csv_path))

        df_dict = {'x': positions[0],
                   'y': positions[1]}

        for det_n in range(4):
            df_dict[f'{energy:.1f} keV d{det_n}'] = response[i, :, det_n]

        df = pd.DataFrame(df_dict)
        with open(response_csv[-1], 'w') as f:
            df.to_csv(f,
                      index=False,
                      float_format='%.6f',
                      quoting=csv.QUOTE_NONNUMERIC)

    # Return paths to the files
    return response_csv, energies


def test_interpolate_cli_run(example_files, tmp_path):
    """Test a run of the CLI with correctly formatted calibration set."""
    response_csv_paths, energies = example_files
    runner = CliRunner()
    os.chdir(str(tmp_path))

    # Prepare arguments for the CLI
    cli_args = []
    for energy, path in zip(energies, response_csv_paths):
        cli_args.extend(['--line', str(energy), str(path)])

    output_path = tmp_path / 'grid.npz'
    cli_args.extend(['--output', str(output_path.absolute())])

    result = runner.invoke(_cli.cli, cli_args)
    assert result.exit_code == 0
    assert output_path.exists()

    # Check for diagnostic files.
    # The names of diagnostic files are derived from the diagnostic class names.
    # I'll assume they are saved as .png files in the current working directory.
    assert (tmp_path / 'in-20_0keV.png').exists()
    assert (tmp_path / 'out-20_0keV.png').exists()
    assert (tmp_path / 'error-20_0keV.png').exists()
