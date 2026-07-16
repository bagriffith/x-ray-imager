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

"""Diagnostic plots for an x-ray imager position response."""
from typing import Optional
from importlib import resources
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy.typing import NDArray
import numpy as np
import x_ray_imager_bagriff

STYLE_FILE = str(resources.files(x_ray_imager_bagriff)
                 .joinpath('plot_style.mplstyle'))


class GenericResponseDiagnostic(Figure):
    """Generic class to plot position response for an x-ray imager.
    
    It's assumed that the imager only has four detectors and they are
    positioned:
           +y
        (2) | (1)
        ----+---- +x
        (3) | (0)
    """
    _SUBPLOT_KWARGS = dict()

    def __init__(self, *args,
                 rc_params: Optional[dict] = None,
                 **kwargs) -> None:
        self.rc_params = mpl.rc_params_from_file(STYLE_FILE,
                                                 fail_on_error=True)
        if rc_params is not None:
            self.rc_params.update(rc_params)

        super().__init__(*args, **kwargs)

    def plot_diagnostic(self,
                        X: NDArray[np.double],  # pylint: disable=invalid-name
                        position: NDArray[np.double],
                        ) -> None:
        """Create axes to plot all four detectors.

        Specific implementation of what to plot on each axis is handled by the
        _diagnostic function.

        Args:
            X: Array containing the imager response to plot. Shape should be
                (n_x_points, n_y_points, n_detectors=4).
            position: Array of positions where X is sampled. Shape should be
                (2, n_x_points, n_y_points).
        
        Raises:
            ValueError: If X and position have mismatched shapes.
        """
        if X.shape[-1] != 4:
            raise ValueError('Response must have four detectors. '
                             f'{X.shape[0]} were provided.')

        if position.shape[0] != 2:
            raise ValueError('Position array has invalid shape '
                             f'{position.shape}. Should be'
                             '(2, n_x_points, n_y_points)')

        if position.shape[1:] != X.shape[:-1]:
            raise ValueError('Mismatch between '
                             f'position shape {position.shape} and '
                             f'response shape {X.shape}.')

        with plt.rc_context(self.rc_params):
            ax_n = [4, 2, 1, 3]

            for i, ax_i in enumerate(ax_n):
                ax = self.add_subplot(2, 2, ax_i, **self._SUBPLOT_KWARGS)
                ax.set_title(f'({i+1})', loc='left', y=0.,
                             verticalalignment='top', pad=-2,
                             horizontalalignment='right',
                             size='medium', weight='heavy')
                self._diagnostic(ax, X[:, :, i], position)

    def _diagnostic(self,
                    ax: Axes,
                    X: NDArray[np.double],  # pylint: disable=invalid-name
                    position: NDArray[np.double],
                    ) -> None:
        """Plot one detector response on the provided Axis.
        
        This should be reimplemented for the specific plot needed.

        Args:
            ax: Axis on which to plot the response.
            X: Single detector response. Shape should be
                (n_x_points, n_y_points).
            position: Array of positions where X is sampled. Shape should be
                (2, n_x_points, n_y_points).
        """
        _ = ax, X, position


class GridWireframeDiagnostic(GenericResponseDiagnostic):
    """Wireframe 3D plot of position response for an x-ray imager."""
    _SUBPLOT_KWARGS = GenericResponseDiagnostic._SUBPLOT_KWARGS.copy()
    _SUBPLOT_KWARGS.update(dict(projection='3d'))

    def plot_diagnostic(self,
                        X: NDArray[np.double],  # pylint: disable=invalid-name
                        position: NDArray[np.double],
                        ) -> None:
        """Create axes to plot all four detectors.
        
        Args:
            X: Array containing the imager response to plot. Shape should be
                (n_detectors=4, n_x_points, n_y_points).
            position: Array of positions where X is sampled. Shape should be
                (2, n_x_points, n_y_points).
        """
        # Plot each component and adjust the shape.
        super().plot_diagnostic(X=X, position=position)
        self.subplots_adjust(hspace=0.15, wspace=0.25,
                             left=0.15, right=0.85,
                             top=0.9, bottom=0.15)

    def _diagnostic(self,
                    ax: Axes3D,
                    X: NDArray[np.double],  # pylint: disable=invalid-name
                    position: NDArray[np.double],
                    ) -> None:
        """Plot a 2D wireframe for one detector response on the provided Axis.
        
        Args:
            ax: Axis on which to plot the response.
            X: Single detector response. Shape should be
                (n_x_points, n_y_points).
            position: Array of positions where X is sampled. Shape should be
                (2, n_x_points, n_y_points).
        """
        ax.plot_wireframe(position[0, :, :],
                          position[1, :, :],
                          X,
                          lw=.25, color='k')


class ColorMeshDiagnostic(GenericResponseDiagnostic):
    """Color mesh plot of position response for an x-ray imager."""
    def __init__(self, *args,
                 rc_params: Optional[dict] = None,
                 **kwargs) -> None:
        # Colorbar scale for each colormap.
        self.norm = Normalize()

        super().__init__(*args, rc_params=rc_params, **kwargs)

    def plot_diagnostic(self,
                        X: NDArray[np.double],  # pylint: disable=invalid-name
                        position: NDArray[np.double],
                        ) -> None:
        """Create axes to plot all four detectors plus a colorbar.
        
        Args:
            X: Array containing the imager response to plot. Shape should be
                (n_detectors=4, n_x_points, n_y_points).
            position: Array of positions where X is sampled. Shape should be
                (2, n_x_points, n_y_points).
        """
        # Establish a constistent colormap for each detector before plotting.
        self.norm.autoscale(X)

        # Plot each component and adjust the shape.
        super().plot_diagnostic(X=X, position=position)
        self.subplots_adjust(hspace=0.15, wspace=0.2,
                             left=0.15, right=0.75)

        # Add the colorbar.
        cbar_ax = self.add_axes((0.8, 0.25, 0.025, 0.5))
        color_mappable = ScalarMappable(norm=self.norm)
        color_mappable.set_cmap(self.rc_params['image.cmap'])
        self.colorbar(color_mappable, cax=cbar_ax)

    def _diagnostic(self,
                    ax: Axes,
                    X: NDArray[np.double],  # pylint: disable=invalid-name
                    position: NDArray[np.double],
                    ) -> None:
        """Plot a colormesh for one detector response on the provided Axis.
        
        Args:
            ax: Axis on which to plot the response.
            X: Single detector response. Shape should be
                (n_x_points, n_y_points).
            position: Array of positions where X is sampled. Shape should be
                (2, n_x_points, n_y_points).
        """
        ax.set_aspect('equal', adjustable='box', anchor='C')
        ax.pcolormesh(position[0, :, :], position[1, :, :], X,
                      norm=self.norm, shading='gouraud')


# Lookup dictionary to grab each response with a cli option.
diagnostics = {'gridwire': GridWireframeDiagnostic,
               'colormesh': ColorMeshDiagnostic}
