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
        with plt.rc_context(self.rc_params):
            self.subplots_adjust(hspace=.15)
            ax_n = [4, 2, 1, 3]

            for i, ax_i in enumerate(ax_n):
                ax = self.add_subplot(2, 2, ax_i, **self._SUBPLOT_KWARGS)
                ax.set_title(f'PMT {i+1}')
                self._diagnostic(ax, X[:, :, i], position)

    def _diagnostic(self,
                    ax: Axes,
                    X: NDArray[np.double],  # pylint: disable=invalid-name
                    position: NDArray[np.double],
                    ) -> None:
        pass


class GridWireframeDiagnostic(GenericResponseDiagnostic):
    """"""
    _SUBPLOT_KWARGS = dict(projection='3d')

    def _diagnostic(self,
                    ax: Axes3D,
                    X: NDArray[np.double],  # pylint: disable=invalid-name
                    position: NDArray[np.double],
                    ) -> None:
        ax.plot_wireframe(position[:, 0], position[:, 1], X, lw=.25, color='k')


class ColorMeshDiagnostic(GenericResponseDiagnostic):
    """"""
    def __init__(self, *args,
                 rc_params: Optional[dict] = None,
                 **kwargs) -> None:
        self.norm = Normalize()

        super().__init__(*args, rc_params=rc_params, **kwargs)

    def plot_diagnostic(self,
                        X: NDArray[np.double],  # pylint: disable=invalid-name
                        position: NDArray[np.double],
                        ) -> None:
        self.norm.autoscale(X)
        super().plot_diagnostic(X=X, position=position)
        self.subplots_adjust(right=0.85)
        cbar_ax = self.add_axes((0.88, 0.05, 0.05, 0.9))
        self.colorbar(ScalarMappable(norm=self.norm), cax=cbar_ax)

    def _diagnostic(self,
                    ax: Axes,
                    X: NDArray[np.double],  # pylint: disable=invalid-name
                    position: NDArray[np.double],
                    ) -> None:
        ax.pcolormesh(position[:, 0], position[:, 1], X.T, norm=self.norm)
