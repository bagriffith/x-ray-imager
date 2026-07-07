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

"""Command line interface to estimate position/energy for an x-ray imager."""
import csv
import logging
import click
import numpy as np
import pandas as pd
from x_ray_imager_bagriff.position_estimation import TreeLookup

logger = logging.getLogger('x_ray_imager_bagriff.position_estimation')


def set_log_level(ctx, param, value):
    """Update the log level according to a CLI value."""
    _ = ctx, param  # Not needed.
    if value is None:
        return
    logger.setLevel(value)
    handler = logging.StreamHandler()
    logger.addHandler(handler)


@click.group()
def cli():
    """Tools for estimating position/energy for x-ray imager observations."""


@cli.command()
@click.argument('calibration', type=click.Path(file_okay=True,
                                               exists=True,
                                               dir_okay=False))
@click.argument('observations', type=click.File())
@click.option('--output', '-o',
              type=click.File(mode='w'), default='-',
              help="Output CSV path instead of stdout.")
@click.option('--threshold', type=click.IntRange(min=0), default=15,
              help="Minimum sum of detector values to output.")
@click.option('--verbose', '-v', flag_value=logging.INFO,
              callback=set_log_level, expose_value=False,
              help="Print extra information during run.")
@click.option('--debug', '-d', flag_value=logging.DEBUG,
              callback=set_log_level, expose_value=False,
              help="Print out all debug information during run.")
def series(calibration, observations, output, threshold):
    """Estimate the position and energy for a list of observations.
    
    CALIBRATION is an npz file with four arrays, x, y, energy, and response.
    Each row should be a single calibration point (interpolated or otherwise).
    OBSERVATIONS is a CSV with each row being a single observed x-ray. An
    estimated position and energy will be added as new columns in a CSV.
    """
    calibration = np.load(calibration)

    cal_e, cal_x, cal_y = np.meshgrid(calibration['energy'],
                                      calibration['x'],
                                      calibration['y'],
                                      indexing='ij')

    estimator = TreeLookup(calibration['response'],
                           cal_e, [cal_x, cal_y],
                           k_lookup=4096)

    obs_df = list(pd.read_csv(observations, chunksize=4096))

    # Create column names
    n_detectors = 4
    det_cols = [f'T{i+1}' for i in range(n_detectors)]
    est_cols = ['energy', 'x', 'y']
    columns = ['#t', *det_cols, *est_cols]

    # Write header
    output.write(','.join(columns) + '\n')

    with click.progressbar(obs_df, show_percent=True) as progress_bar:
        for df_chunk in progress_bar:
            amp = np.sum(df_chunk[det_cols].to_numpy(), axis=-1)
            df_chunk = df_chunk[amp > threshold]

            logger.info("Processing %d observations.", len(df_chunk))

            df_chunk[est_cols] = estimator.get_value(df_chunk[det_cols]).T

            df_chunk.to_csv(output,
                            columns=columns,
                            mode='a',
                            header=False,
                            index=False,
                            quoting=csv.QUOTE_NONNUMERIC)
