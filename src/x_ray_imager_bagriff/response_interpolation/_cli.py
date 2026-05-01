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

import logging
import click
import numpy as np
import pandas as pd
from x_ray_imager_bagriff.response_interpolation import (
    CubicInterpolation,
    plot
)

logger = logging.getLogger('x_ray_imager_bagriff.identify_lines')


@click.command()
@click.option('--line', '-l', multiple=True, type=(click.File(), float),)
@click.option('--verbose', '-v', 'log_level',
              flag_value=logging.INFO, default=logging.WARNING)
@click.option('--debug', '-d', 'log_level', flag_value=logging.DEBUG)
def cli(line, log_level):
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

    sampled_energy = np.empty(len(line), dtype=np.double)
    sampled_position = np.empty((0, 0), dtype=np.double)
    sampled_response = np.empty((len(line), 0, 0, 4), dtype=np.double)

    for i, (filename, energy) in enumerate(line):
        sampled_energy[i] = energy
        df = pd.read_csv(filename)

        width = int(round(np.sqrt(len(df))))
        assert width*width == len(df)

        df.sort_values(['y', 'x'], inplace=True)
        positions = np.array([df['x'].to_numpy().reshape((width, width)),
                              df['y'].to_numpy().reshape((width, width))])

        if i == 0:
            # First point, fill in positions
            sampled_position = positions
            sampled_response = np.empty((sampled_response.shape[0],
                                         *positions.shape[1:],
                                         sampled_response.shape[-1]))
            logger.debug('sampled_response shape: %s', sampled_response.shape)
        else:
            # Check that this set of positions is the same
            assert positions.shape == sampled_position.shape
            assert np.all(np.abs(positions - sampled_position) < 1e-7)

        for det_n in range(4):
            sampled_response[i, :, :, det_n] = \
                df[f'{energy:.1f} keV T{det_n}']\
                    .to_numpy()\
                    .reshape(positions.shape[1:])

    # TODO: Diagnostics
    interpolator = CubicInterpolation(sampled_energy,
                                      sampled_position,
                                      sampled_response)

    energy = np.arange(10, 1000, 2, dtype=np.double)
    x = np.linspace(-70, 70, 141)
    y = np.linspace(-70, 70, 141)
    E, X, Y = np.meshgrid(energy, x, y, indexing='ij')

    grid_hr = interpolator.values(E, X, Y)
    np.savez('grid', energy=E, x=X, y=Y, response=grid_hr)
