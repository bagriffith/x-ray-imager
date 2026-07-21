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

"""Plot observations from an x-ray imager."""
from importlib import resources
import logging
from typing import Optional, override
from matplotlib import rc_params_from_file
from matplotlib.animation import TimedAnimation
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from pandas import DataFrame
from scipy.stats import truncnorm
import x_ray_imager_bagriff

STYLE_FILE = str(resources.files(x_ray_imager_bagriff)
                 .joinpath('plot_style.mplstyle'))

logger = logging.getLogger(__name__)


class ImagerAxes(Axes):
    """Axes with specific plotting functions for x-ray imagers."""
    def energy_spectrum(
            self,
            energy: ArrayLike,
            bins: Optional[ArrayLike] = None,
            duration: Optional[float] = None
            ) -> BarContainer | Polygon | list[BarContainer | Polygon]:
        """Plot an energy spectrum histogram.
        
        Args:
            energy: Observation energies to bin into the historgram.
                Expected to be in keV.
            bins: Energy edges for histogram bins Default if not provided
                spans from 10 keV to 600 keV.
            duration: Total time for collecting these measurements. If
                provided the y-axis will be time normalized.
        """
        if bins is None:
            # Default value
            bins = np.linspace(10, 600, 296)  # keV

        bins = list(np.array(bins, dtype=np.float64))

        self.set_xlim(bins[0], bins[-1])
        self.set_xlabel(r'$\textrm{keV}$')
        self.set_ylabel(r'$\textrm{x-rays} / \textrm{keV}$' if duration is None
                        else r'$/\textrm{s}\,\textrm{keV}$')
        # 1/t weight puts the plot in units of /s.
        duration_weight = np.full_like(energy,
                                       1.0 if duration is None else 1/duration)

        return self.hist(energy, bins,
                         density=True,
                         weights=duration_weight)[-1]

    def image_hist(self,
                   x: ArrayLike, y: ArrayLike,
                   bins: Optional[ArrayLike] = None,
                   image_max: Optional[float] = None,
                   dx: Optional[ArrayLike] = None,
                   dy: Optional[ArrayLike] = None,
                   duration: Optional[float] = None
                   ) -> QuadMesh:
        """Create a 2D histogram of observed x-ray positions.
        
        Args:
            x: Observation array x-coordinate. Shape matches y.
            y: Observation array y-coordinate.
            bins: Position bin edges used for both x and y coordinates.
                Also sets the axis limits. If not provided, defaults to
                -70 mm to +70 mm in 2.5 mm bins.
        """
        _ = duration  # Not used

        if bins is None:
            # Default value
            bins = np.linspace(-70, 70, 57)  # mm

        bins = list(np.array(bins, dtype=np.float64))

        self.set_xlim(bins[0], bins[-1])
        self.set_ylim(bins[0], bins[-1])
        self.set_aspect('equal')

        if dx is None and dy is None:
            return self.hist2d(x, y, bins, vmin=0, vmax=image_max)[-1]

        if dx is None or dy is None:
            raise ValueError('Must provide both dx and dy.')

        x = np.array(x)
        y = np.array(y)
        dx = np.array(dx)
        dy = np.array(dy)

        if np.any(dx < bins[1] - bins[0]) or np.any(dx < bins[1] - bins[0]):
            logger.warning('Image pixels are larger than position errors.')

        image = np.zeros([len(bins)-1]*2)

        for pos, err in zip(zip(x, y), zip(dx, dy)):
            p = [(truncnorm.cdf(bins[1:], bins[0], bins[-1], pos_j, err_j)
                  - truncnorm.cdf(bins[:-1], bins[0], bins[-1], pos_j, err_j))
                 for pos_j, err_j in zip(pos, err)]

            single_image = np.outer(*p)
            image_total = np.sum(single_image)

            if np.abs(image_total - 1) > 0.01:
                logger.warning("Total error should be one, but is instead %s.",
                               image_total)

            image += single_image / image_total

        return self.pcolormesh(bins, bins, image, vmin=0, vmax=image_max)


class ImagerFigure(Figure):
    """Generic base figure for use with ImagerAxes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rc_params = rc_params_from_file(STYLE_FILE, fail_on_error=True)
        self.rc_params.update(dict())

    def plot_observations(self,
                          energy: ArrayLike,
                          x: ArrayLike,
                          y: ArrayLike,
                          **kwargs):
        """Create all plots for the observations provided.
        
        Args:
            energy: Observation array energies. Shape matches x and y
            x: Observation array x-coordinate.
            y: Observation array y-coordinate.
        """
        _ = energy, x, y, kwargs
        raise NotImplementedError('Base class not implemented.')

    @staticmethod
    def _filter_by_energy(energy: ArrayLike,
                          x: ArrayLike,
                          y: ArrayLike,
                          energy_range: tuple[float, float]
                          ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Produce a filtered x and y with only events in an energy range.
        
        Args:
            energy: Observation array energies. Shape matches x and y
            x: Observation array x-coordinate.
            y: Observation array y-coordinate.
            energy_range: Tuple with the lower and upper limit to be selected.
        """
        if energy_range[0] > energy_range[1]:
            raise ValueError("energy_range right edge is smaller.")

        energy = np.array(energy, dtype=np.float64)
        in_range = (energy > energy_range[0]) & (energy < energy_range[1])
        x = np.array(x, dtype=np.float64)[in_range]
        y = np.array(y, dtype=np.float64)[in_range]
        return x, y


class SpectrumFigure(ImagerFigure):
    """Figure with only the energy spectrum plot."""

    def __init__(self, *args,
                 spectrum_max: Optional[float] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        gs = GridSpec(1, 1, figure=self)
        self.ax_spectrum = self.add_subplot(gs[0], axes_class=ImagerAxes)
        self.spectrum_max = spectrum_max

    def plot_observations(self,
                          energy: ArrayLike,
                          x: ArrayLike,
                          y: ArrayLike,
                          duration: Optional[float] = None,
                          **kwargs):
        """Plot the energy spectrum.
        
        Args:
            energy: Observation array energies. Shape matches x and y
            x: Observation array x-coordinate.
            y: Observation array y-coordinate.
            duration: Total time for collecting these measurements. See
                ``ImagerAxes.energy_spectrum()`` for usage.
        """
        with plt.rc_context(self.rc_params):
            _ = x, y
            self.ax_spectrum.clear()
            assert isinstance(self.ax_spectrum, ImagerAxes)
            self.ax_spectrum.energy_spectrum(energy, duration=duration)
            self.ax_spectrum.set_ylim(0, self.spectrum_max)


class ImageHistFigure(ImagerFigure):
    """Figure with only the position histogram image.
    
    Attributes:
        ax_image: The position histogram Axes.
        max_imager: The maximum value for the position colormap.
        ax_colorbar: The colorbar Axes.
    """

    def __init__(self, *args,
                 image_max: Optional[float] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        gs = GridSpec(1, 8, figure=self, wspace=1.0)
        self.ax_image = self.add_subplot(gs[:-1], axes_class=ImagerAxes)
        self.image_max = image_max
        self.ax_colorbar = self.add_subplot(gs[-1], axes_class=ImagerAxes)

    def plot_observations(self,
                          energy: ArrayLike,
                          x: ArrayLike,
                          y: ArrayLike,
                          energy_range: Optional[tuple[float, float]] = None,
                          **kwargs):
        """Plot the position historgram.
        
        Args:
            energy: Observation array energies. Shape matches x and y
            x: Observation array x-coordinate.
            y: Observation array y-coordinate.
            energy_range: Tuple with the lower and upper limit to be used for
                the position histogram. Others are ignored.
            dx:
            dy: 
        """
        with plt.rc_context(self.rc_params):
            if energy_range is not None:
                x, y = self._filter_by_energy(energy, x, y, energy_range)

            self.ax_image.clear()
            self.ax_colorbar.clear()
            assert isinstance(self.ax_image, ImagerAxes)
            col = self.ax_image.image_hist(x, y,
                                           image_max=self.image_max,
                                           **kwargs)
            self.colorbar(col, cax=self.ax_colorbar)
            self.ax_colorbar.set_ylabel('x-rays / bin')


class ImageSpectrumFigure(SpectrumFigure, ImageHistFigure):
    """Figure with an energy spectrum on top of a position historgram.
    
    Attributes:
        ax_spectrum: The energy spectrum Axes.
        spectrum_max: The y-axis for the energy spectrum. None will use auto.
        ax_image: The position histogram Axes.
        max_imager: The maximum value for the position colormap.
        ax_colorbar: The colorbar Axes.
    """

    def __init__(self, *args,
                 spectrum_max: Optional[float] = None,
                 image_max: Optional[float] = None,
                 **kwargs):
        ImagerFigure.__init__(self, *args, **kwargs)
        gs = GridSpec(8, 8, figure=self,
                      hspace=1.0, wspace=1.0)
        self.ax_spectrum = self.add_subplot(gs[:2, :], axes_class=ImagerAxes)
        self.spectrum_max = spectrum_max
        self.ax_image = self.add_subplot(gs[2:, :-1], axes_class=ImagerAxes)
        self.image_max = image_max
        self.ax_colorbar = self.add_subplot(gs[2:, -1], axes_class=ImagerAxes)

    @override
    def plot_observations(self, *args, **kwargs):
        """Plot both energy spectrum and position histogram."""
        SpectrumFigure.plot_observations(self, *args, **kwargs)
        ImageHistFigure.plot_observations(self, *args, **kwargs)


class ImagerAnimation(TimedAnimation):
    """Animation for a time series of x-ray observations."""
    def __init__(self,
                 fig: ImagerFigure,
                 df: DataFrame,
                 step_duration: float,
                 energy_range: Optional[tuple[float, float]] = None,
                 **kwargs):
        """Initialize the x-ray imager animation.
        
        Args:
            fig: The figure used to plot each time step. Should have a
                ``plot_observations()`` function.
            df: DataFrame with the observations. It should have columns
                'x', 'y', 'energy' plus the time in seconds under 't'.
            step_duration: Duration of each frame in seconds.
            energy_range: Energy limits to be used. For details,
                see ``ImageHistFigure.plot_observations()``.
        """
        self.fig = fig
        self.step_duration = step_duration
        self.energy_range = energy_range
        time_delta = step_duration * pd.Timedelta('1s')
        df['t'] = df['t'] * pd.Timedelta('1s')

        self._framedata = df.resample(time_delta, on='t')
        super().__init__(fig, **kwargs)

    def _draw_frame(self, framedata: tuple[pd.Timedelta, DataFrame]):
        """Draw the frame for each step.
        
        Args:
            framedata: Information to draw each frame, should be the output
                of the ``Dataframe.resample()`` function.
        """
        t, df = framedata
        self.fig.plot_observations(
            df['energy'].to_numpy(),
            df['x'].to_numpy(),
            df['y'].to_numpy(),
            duration=self.step_duration,
            energy_range=self.energy_range)
        self.fig.suptitle(f't = {t.isoformat()}')


figure_names = {'spectrum': SpectrumFigure,
                'image': ImageHistFigure,
                'both': ImageSpectrumFigure}
