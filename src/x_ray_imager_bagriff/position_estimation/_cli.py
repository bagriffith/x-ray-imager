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
from pathlib import Path
import click
import numpy as np
import pandas as pd
from x_ray_imager_bagriff.position_estimation import TreeLookup, plot
from x_ray_imager_bagriff.cli import log_level_options

logger = logging.getLogger('x_ray_imager_bagriff.position_estimation')

PLOT_CHOICE = click.Choice(plot.figure_names.keys())


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
@log_level_options(logger)
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
    error_cols = ['d_energy', 'd_x', 'd_y']
    columns = ['t', *det_cols, *est_cols]

    # Write header
    output.write(','.join(columns) + '\n')

    with click.progressbar(obs_df, show_percent=True) as progress_bar:
        for df_chunk in progress_bar:
            if '#t' in df_chunk:
                df_chunk['t'] = df_chunk['#t']

            amp = np.sum(df_chunk[det_cols].to_numpy(), axis=-1)
            df_chunk = df_chunk[amp > threshold]

            logger.info("Processing %d observations.", len(df_chunk))

            # df_chunk[est_cols] = estimator.get_value(df_chunk[det_cols]).T
            df_chunk[est_cols], df_chunk[error_cols] = \
                (x.T
                 for x in estimator.get_values_with_error(df_chunk[det_cols]))

            df_chunk.to_csv(output,
                            columns=columns,
                            mode='a',
                            header=False,
                            index=False,
                            quoting=csv.QUOTE_NONNUMERIC)


@cli.command("plot")
@click.argument('observations', type=click.File())
@click.argument('figure', type=PLOT_CHOICE)
@click.option('--output', '-o', default=Path("./plot.png"),
              type=click.Path(file_okay=True, dir_okay=False),
              help="Output plot path. Any type supported by matplotlib.")
@click.option('--energy', '-e', 'energy_range', nargs=2,
              type=click.FloatRange(min=0),
              help="Only use x-rays in this energy band for position plots.")
@log_level_options(logger)
def plot_fig(observations, figure, output, energy_range):
    """Plot a set of x-ray positions and/or energies.

    OBSERVATIONS is a CSV with each row being a single observed x-ray.
    It should have columns 'energy', 'x', and 'y'. FIGURE is the selected
    plot type. All energies should be in keV and lengths in mm.
    """
    df = pd.read_csv(observations)

    figsize = (6.5, {'spectrum': 4.0, 'image': 5.0, 'both': 8}[figure])
    fig = plot.figure_names[figure](figsize=figsize, image_max=400)
    fig.plot_observations(df['energy'], df['x'], df['y'],
                          energy_range=energy_range)
    fig.savefig(output)


@cli.command()
@click.argument('observations', type=click.File())
@click.argument('figure', type=PLOT_CHOICE)
@click.option('--output', '-o', default=Path("./imager.mp4"),
              type=click.Path(file_okay=True, dir_okay=False),
              help="Output animation plot with a file type supported by "
                   "matplotlib's FFMpegWriter.")
@click.option('--step', '-s', 'step_duration',
              type=click.FloatRange(min=0), default=10.0,
              help="Observation time for each frame, in seconds.")
@click.option('--energy', '-e', 'energy_range', nargs=2,
              type=click.FloatRange(min=0),
              help="Only use x-rays in this energy band for position plots.")
@click.option('--interval', '-i', 'frame_interval',
              type=click.IntRange(min=1), default=100,
              help="Frame duration in the final video.")
@click.option('--maxspectrum', 'max_spectrum',
              type=click.FloatRange(min=0, min_open=True),
              help="Max on the energy spectrum y-axis (x-rays / keV.s).")
@click.option('--maximage', 'max_image',
              type=click.FloatRange(min=0, min_open=True),
              help="Max on the image colormap (x-rays / bin).")
@log_level_options(logger)
def animation(observations, figure, output, step_duration, frame_interval,
              energy_range, max_spectrum, max_image):
    """Animate a time series of x-ray positions and/or energies.
    
    Uses the same figures as the ``plot`` command for each frame.
    OBSERVATIONS should be a CSV with columns 'energy', 'x', 'y', and 't'.
    All times are in seconds.
    """
    df = pd.read_csv(observations)
    if '#t' in df:
        df['t'] = df['#t']

    figsize = (6.5, {'spectrum': 4.0, 'image': 5.0, 'both': 8}[figure])
    fig = plot.figure_names[figure](figsize=figsize,
                                    spectrum_max=max_spectrum,
                                    image_max=max_image)

    anim = plot.ImagerAnimation(fig=fig, df=df,
                                step_duration=step_duration,
                                energy_range=energy_range,
                                interval=frame_interval)

    # The total number of frames:
    steps = int(df['t'].max() // pd.Timedelta(step_duration, 's'))

    # Override the animation iterator with the progressbar
    # pylint: disable=protected-access
    with click.progressbar(anim._framedata, length=steps) as bar_fd:
        anim._framedata = bar_fd  # type: ignore
        anim.save(filename=output, writer="ffmpeg")
