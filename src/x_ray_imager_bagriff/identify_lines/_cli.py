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
from x_ray_imager_bagriff.identify_lines import (
    MinOPTICS,
    SourceParams,
    find_lines
)
from x_ray_imager_bagriff.identify_lines.plot import diagnostics

logger = logging.getLogger('x_ray_imager_bagriff.identify_lines')

matplotlib.use('Agg')

# More clustering options could be added from sklearn.cluster
# TODO: Fix this to handle sources.
# ClusterChoice = click.Choice([MinDBSCAN])
SourceChoice = click.Choice(SourceParams.source_choices())
DiagnosticChoice = click.Choice([None] + list(diagnostics.keys()))


@click.group()
@click.option('--verbose', '-v', 'log_level',
              flag_value=logging.INFO, default=logging.WARNING)
@click.option('--debug', '-d', 'log_level', flag_value=logging.DEBUG)
def cli(log_level):
    """Identify gamma line for x-ray imager calibration."""
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

@cli.command('point', short_help='Identify lines for one csv of measurements.')
@click.argument('filename', type=click.File())
@click.argument('source', type=SourceChoice)
@click.option('--gain', nargs=2, type=float, default=None)
@click.option('--diagnostic', type=DiagnosticChoice,
              default=None)
@click.option('--output', '-o', type=click.File(mode='w'), default='-')
def point(filename, source, gain, diagnostic, output):
    events = np.loadtxt(filename, delimiter=',', skiprows=1, dtype=np.long)
    source = SourceParams.get_source(source)
    if diagnostic is not None:
        diagnostic = diagnostics[diagnostic]()  # Setup figure here.

    cluster = MinOPTICS(min_clusters=len(source),
                        max_eps=10,
                        cluster_method='dbscan')

    responses = find_lines(events, cluster,
                                    source,
                                    gain_range=gain,
                                    diagnostic=diagnostic)

    np.savetxt(output, responses, delimiter=',', fmt='%.6f')


@cli.command('grid', short_help='Identify lines at each point in a grid.')
@click.argument('filename', type=click.File())
@click.argument('source', type=SourceChoice)
@click.option('--gain', nargs=2, type=float, default=None)
@click.option('--output', '-o', type=click.File(mode='w'), default='-')
def grid(filename, source, gain, output):
    logger.debug('file: %s source %s output %s', filename, source, output)
    df = pd.read_csv(filename)
    source = SourceParams.get_source(source)

    cluster = MinOPTICS(min_clusters=len(source),
                        max_eps=10,
                        cluster_method='dbscan')

    line_cols = [f'{x:.1f} keV T{n}'
                 for x in source.energies
                 for n in range(4)]

    df[line_cols] = df[['csv_path']].apply(
        lambda x: find_lines(
            np.loadtxt(x['csv_path'],
                       delimiter=',',
                       skiprows=1,
                       dtype=np.long),
            cluster,
            source,
            gain_range=gain).flatten(),
        axis=1,
        result_type="expand")

    df.to_csv(output, index=False, quoting=csv.QUOTE_NONNUMERIC)
