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

"""Command Line Interface to interpolate within a calibration grid."""
import logging
from pathlib import Path
import click
import numpy as np
import pandas as pd
from x_ray_imager_bagriff.response_interpolation import (
    CubicInterpolation,
    plot
)

logger = logging.getLogger('x_ray_imager_bagriff.identify_lines')


def set_log_level(ctx, param, value):
    """Update the log level according to a CLI value."""
    _ = ctx, param  # Not needed.
    if value is None:
        return
    logger.setLevel(value)
    handler = logging.StreamHandler()
    logger.addHandler(handler)


@click.command()
@click.argument('files', nargs=-1)
@click.option('--line', '-l', 'lines',
              multiple=True, type=(float, click.Path(dir_okay=False, exists=True)),
              help="For a gamma line, the path to a CSV of position responses "
              "and the line's energy in keV. May be provided multiple times.")
@click.option('--output', '-o',
              type=click.Path(dir_okay=False), default='./grid.npz',
              help="Output path instead of stdout.")
@click.option('--diagnostics', '-p', 'plot_diagnostics', is_flag=True,
              help="Save diagnostic plots.")
@click.option('--verbose', '-v', flag_value=logging.INFO,
              callback=set_log_level, expose_value=False,
              help="Print extra information during run.")
@click.option('--debug', '-d', flag_value=logging.DEBUG,
              callback=set_log_level, expose_value=False,
              help="Print out all debug information during run.")
def cli(files, lines, output, plot_diagnostics):
    """Collects a set of gamma line responses and outputs an interpolated grid.

    Line responses should be contained in a CSV with one source position
    in each row. There should two columns for x and y position. Response
    for the line should be stored in n_detectors columns each named
    "<Line Energy> keV d<Detector Number>", with energy in keV to one decimal.
    This is the same energy that should be provided with each file.

    Line response can either be provided individually with the --line argument
    or with CSV files. Each CSV should have one row per line and have columns
    titled in the header "csv_path" and "energy".
    """
    n_detectors = 4

    # Collect the calibration data set
    energy_list = list()
    csv_path_list = list()

    for f in files:
        df = pd.read_csv(f)
        energy_list += [float(x) for x in df['energy']]
        csv_path_list += [Path(x) for x in df['csv_path']]

    for line in lines:
        energy_list.append(float(line[0]))
        csv_path_list.append(str(line[1]))

    # Sort the lists by energy

    energy_list, csv_path_list = \
        (list(t) for t in zip(*sorted(zip(energy_list, csv_path_list))))

    # Check for no duplicate energies
    check_e_set = [int(round(x*10)) for x in energy_list]
    for x in set(check_e_set):
        check_e_set.remove(x)
    duplicates_energies = [x/10 for x in check_e_set]
    if duplicates_energies:
        raise ValueError(f'Duplicate energies provided: {duplicates_energies}')

    # Load the files
    position_list = list()
    response_list = list()

    for filename, energy_hr in zip(csv_path_list, energy_list):
        logger.info('Loading %.1f keV: %s', energy_hr, filename)
        df = pd.read_csv(filename)
        df.sort_values(by=['x', 'y'], inplace=True)
        df.reset_index(inplace=True, drop=True)
        positions = np.array([df['x'].to_numpy(),
                              df['y'].to_numpy()])
        response = df[[f'{energy_hr:.1f} keV d{i}'
                       for i in range(n_detectors)]].to_numpy()

        position_list.append(positions)
        response_list.append(response)

    # Check shape of positions
    n_total_positions = np.shape(position_list[0])[1]
    width = int(round(np.sqrt(n_total_positions)))
    if width*width != n_total_positions:
        raise ValueError(f'Calibration contains {n_total_positions} points '
                         'which cannot be made into a square grid.')

    for p in position_list:
        if position_list[0].shape != p.shape:
            raise ValueError('Set with mismatched number of points.')

        if np.any(np.abs(p - position_list[0]) > 0.1):
            raise ValueError('Set with mismatched grid value.')

    # Convert these into correctly shaped numpy arrays
    sampled_energy = np.array(energy_list)
    sampled_position = position_list[0].reshape((2, width, width))
    sampled_response = np.array([x.reshape((width, width, 4))
                                 for x in response_list])

    # Create the high resolution output grid
    mesh = dict()
    energy_hr = np.arange(10, 1000, 2, dtype=np.double)
    x_hr = np.linspace(-70, 70, 141)
    y_hr = np.linspace(-70, 70, 141)
    mesh['energy'], mesh['x'], mesh['y'] = np.meshgrid(energy_hr, x_hr, y_hr, indexing='ij')

    diagnostic_kwargs = dict()
    if plot_diagnostics:
        diagnostic_kwargs = \
            {'input_diagnostic': plot.GridWireframeDiagnostic(),
             'output_diagnostic': plot.ColorMeshDiagnostic(),
             'error_diagnostic': plot.ColorMeshDiagnostic()}

    interpolator = CubicInterpolation(sampled_energy,
                                      sampled_position,
                                      sampled_response,
                                      **diagnostic_kwargs)

    # Interpolate
    mesh['response'] = interpolator.values(mesh['energy'],
                                           mesh['x'],
                                           mesh['y'])

    # Save arrays
    print()
    np.savez(str(output), *mesh)
