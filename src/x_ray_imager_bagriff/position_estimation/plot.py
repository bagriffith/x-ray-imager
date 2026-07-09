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
from typing import Optional
from matplotlib.animation import TimedAnimation
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from pandas import DataFrame


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
            bins = np.linspace(10, 600, 119)  # keV

        bins = list(np.array(bins, dtype=np.float64))

        self.set_xlim(bins[0], bins[-1])
        self.set_xlabel(r'$\text{keV}$')
        self.set_ylabel(r'$\text{x-rays} / \text{keV}$' if duration is None
                        else r'$/\text{s}\,\text{keV}$')
        # 1/t weight puts the plot in units of /s.
        duration_weight = np.full_like(energy,
                                       1.0 if duration is None else 1/duration)

        return self.hist(energy, bins,
                         density=True,
                         weights=duration_weight)[-1]

    def image_hist(self,
                   x: ArrayLike, y: ArrayLike,
                   bins: Optional[ArrayLike] = None
                   ) -> QuadMesh:
        """Create a 2D histogram of observed x-ray positions.
        
        Args:
            x: Observation array x-coordinate. Shape matches y.
            y: Observation array y-coordinate.
            bins: Position bin edges used for both x and y coordinates.
                Also sets the axis limits. If not provided, defaults to
                -70 mm to +70 mm in 2.5 mm bins.
        """
        if bins is None:
            # Default value
            bins = np.linspace(-70, 70, 57)  # mm

        bins = list(np.array(bins, dtype=np.float64))

        self.set_xlim(bins[0], bins[-1])
        self.set_ylim(bins[0], bins[-1])
        self.set_aspect('equal')

        return self.hist2d(x, y, bins)[-1]


class ImagerFigure(Figure):
    """Generic base figure for use with ImagerAxes."""

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gs = GridSpec(1, 1, figure=self)
        self.ax_spectrum = self.add_subplot(gs[0], axes_class=ImagerAxes)

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
        _ = x, y
        self.ax_spectrum.clear()
        assert isinstance(self.ax_spectrum, ImagerAxes)
        self.ax_spectrum.energy_spectrum(energy, duration=duration)


class ImageHistFigure(ImagerFigure):
    """Figure with only the position histogram image."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gs = GridSpec(1, 8, figure=self, wspace=1.0)
        self.ax_image = self.add_subplot(gs[:-1], axes_class=ImagerAxes)
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
        """
        if energy_range is not None:
            x, y = self._filter_by_energy(energy, x, y, energy_range)

        self.ax_image.clear()
        self.ax_colorbar.clear()
        assert isinstance(self.ax_image, ImagerAxes)
        col = self.ax_image.image_hist(x, y)
        self.colorbar(col, cax=self.ax_colorbar)


class ImageSpectrumFigure(ImagerFigure):
    """Figure with an energy spectrum on top of a position historgram."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gs = GridSpec(8, 8, figure=self,
                      hspace=1.0, wspace=1.0)
        self.ax_spectrum = self.add_subplot(gs[:2, :], axes_class=ImagerAxes)
        self.ax_image = self.add_subplot(gs[2:, :-1], axes_class=ImagerAxes)
        self.ax_colorbar = self.add_subplot(gs[2:, -1], axes_class=ImagerAxes)

    def plot_observations(self,
                          energy: ArrayLike,
                          x: ArrayLike,
                          y: ArrayLike,
                          duration: Optional[float] = None,
                          energy_range: Optional[tuple[float, float]] = None,
                          **kwargs):
        """Plot both energy spectrum and position histogram.
        
        Args:
            energy: Observation array energies. Shape matches x and y
            x: Observation array x-coordinate.
            y: Observation array y-coordinate.
            duration: Total time for collecting these measurements. See
                ``ImagerAxes.energy_spectrum()`` for usage.
            energy_range: Tuple with the lower and upper limit to be used for
                the position histogram. Others are ignored.
        """
        self.ax_spectrum.clear()
        assert isinstance(self.ax_spectrum, ImagerAxes)
        self.ax_spectrum.energy_spectrum(energy, duration=duration)

        if energy_range is not None:
            x, y = self._filter_by_energy(energy, x, y, energy_range)
            self.ax_spectrum.axvspan(*energy_range, alpha=0.25, color='C1')

        self.ax_image.clear()
        self.ax_colorbar.clear()
        assert isinstance(self.ax_image, ImagerAxes)
        col = self.ax_image.image_hist(x, y)
        self.colorbar(col, cax=self.ax_colorbar)


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
        super().__init__(fig, **kwargs)
        self.fig = fig
        self.step_duration = step_duration
        time_delta = step_duration * pd.Timedelta('1s')
        self.energy_range = energy_range
        df['t'] = df['t'] * pd.Timedelta('1s')

        self._framedata = df.resample(time_delta, on='t')

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
