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

"""Command Line Interface to identify gamma lines"""
import csv
import logging
import click
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from x_ray_imager_bagriff.identify_lines import (
    MinOPTICS,
    SourceParams,
    find_lines
)
from x_ray_imager_bagriff.identify_lines.plot import diagnostics

logger = logging.getLogger('x_ray_imager_bagriff.identify_lines')

matplotlib.use('Agg')

SOURCE_CHOICE = click.Choice(SourceParams.source_choices())
DIAGNOSTIC_CHOICE = click.Choice([None] + list(diagnostics.keys()))


def set_log_level(ctx, param, value):
    """Update the log level according to a CLI value."""
    _ = ctx, param  # Not needed.
    if value is None:
        return
    logger.setLevel(value)
    handler = logging.StreamHandler()
    logger.addHandler(handler)


def load_measurement_csv(filename):
    """Load a CSV of measurements.
    
    Should have a one line header, then one measurement per row. Each
    column should be a detector in order. Values are expected to be integers
    that fit in a 32 bit signed int.
    """
    return np.loadtxt(filename, delimiter=',', skiprows=1, dtype=np.long)


@click.group()
def cli():
    """Tools to identify gamma source lines in x-ray imager data."""


@cli.command()
@click.argument('filename', type=click.File())
@click.argument('source', type=SOURCE_CHOICE)
@click.option('--gain', '-g',
              nargs=2, type=float, default=None,
              help="Possible gain range (det units / keV).")
@click.option('--diagnostic', '-p',
              type=DIAGNOSTIC_CHOICE, default=None,
              help="Diagnostic to visualize groupings.")
@click.option('--output', '-o',
              type=click.File(mode='w'), default='-',
              help="Output CSV path instead of stdout.")
@click.option('--verbose', '-v', flag_value=logging.INFO,
              callback=set_log_level, expose_value=False,
              help="Print extra information during run.")
@click.option('--debug', '-d', flag_value=logging.DEBUG,
              callback=set_log_level, expose_value=False,
              help="Print out all debug information during run.")
def single(filename, source, gain, diagnostic, output):
    """Identify gamma source lines in FILENAME, a CSV of measurements.
    
    A CSV of OUTPUT will be output with one line per row. The first column
    is the line energy, then one column for each detector.
    """
    events = load_measurement_csv(filename)
    source = SourceParams.get_source(source)
    logger.info('Selected %s source', source.name)

    if diagnostic is not None:
        diagnostic = diagnostics[diagnostic]()  # Setup figure here.

    # Currently there is no way to select the cluster that is used other than
    #   modifying this function. Frequently adjusting these settings would
    #   indicate that I should add an option for it.
    cluster = MinOPTICS(min_clusters=len(source),
                        max_eps=10,
                        cluster_method='dbscan')

    responses = find_lines(events,
                           cluster,
                           source,
                           gain_range=gain,
                           diagnostic=diagnostic)

    # Add a column for with energy
    output_array = np.vstack((np.float64(source.energies),
                              responses.T)).T
    column_names = ['energy'] + [f'd{i}'
                                 for i in range(responses.shape[1])]

    np.savetxt(output, output_array, fmt='%.6f',
               header=','.join(column_names), delimiter=',')


@cli.command()
@click.argument('filename', type=click.File())
@click.argument('source', type=SOURCE_CHOICE)
@click.option('--gain', '-g',
              nargs=2, type=float, default=None,
              help="Possible gain range (det units / keV).")
@click.option('--output', '-o',
              type=click.File(mode='w'), default='-',
              help="Output CSV path instead of stdout.")
@click.option('--bar', '-b', is_flag=True,
              help="Print extra information during run.")
@click.option('--verbose', '-v', flag_value=logging.INFO,
              callback=set_log_level, expose_value=False,
              help="Print extra information during run.")
@click.option('--debug', '-d', flag_value=logging.DEBUG,
              callback=set_log_level, expose_value=False,
              help="Print out all debug information during run.")
def multiple(filename, source, gain, output, bar):
    """Identify gamma source lines multiple times for multiple sets.
    
    FILENAME should be a CSV with headers where each row being a single set of
    measurements. The only required column is "csv_path". That file should be
    compatable with the `single FILENAME` command. It may be an absolute path
    or relative to the working directory. Other columns will be caried to the
    output. This can be used for metadata like the position for that set.
    
    The output will append n_detectors*n_lines columns. They're titled
    "<Line Energy> keV d<Detector Number>", with energy in keV to one decimal.
    Currently n_detectors is fixed equal to four.
    """
    logger.debug('file: %s source %s output %s', filename, source, output)
    df = pd.read_csv(filename)
    source = SourceParams.get_source(source)

    cluster = MinOPTICS(min_clusters=len(source),
                        max_eps=10,
                        cluster_method='dbscan')

    n_detectors = 4  # This could be added as an option if needed.
    line_cols = [f'{line_e:.1f} keV d{n}'
                 for line_e in source.energies
                 for n in range(n_detectors)]

    # For each file, load it
    if bar:
        tqdm.pandas()
        df[line_cols] = df[['csv_path']].progress_apply(
            lambda x: find_lines(load_measurement_csv(x['csv_path']),
                                cluster,
                                source,
                                gain_range=gain).flatten(),
            axis=1,
            result_type="expand")
    else:
        df[line_cols] = df[['csv_path']].apply(
            lambda x: find_lines(load_measurement_csv(x['csv_path']),
                                cluster,
                                source,
                                gain_range=gain).flatten(),
            axis=1,
            result_type="expand")

    df.to_csv(output,
              index=False,
              float_format='%.6f',
              quoting=csv.QUOTE_NONNUMERIC)
