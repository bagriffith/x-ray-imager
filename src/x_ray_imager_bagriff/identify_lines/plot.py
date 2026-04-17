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
"""Creates diagnostic plots for the line identification."""
from typing import Optional
from importlib import resources
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import x_ray_imager_bagriff

STYLE_FILE = str(resources.files(x_ray_imager_bagriff)
                 .joinpath('plot_style.mplstyle'))


class GenericDiagnostic(Figure):
    def __init__(self, *args,
                 rc_params: Optional[dict] = None,
                 **kwargs) -> None:
        self.rc_params = mpl.rc_params_from_file(STYLE_FILE,
                                                 fail_on_error=True)
        if rc_params is not None:
            self.rc_params.update(rc_params)

        color_list = [plt.get_cmap('magma')(i/5) for i in range(6)]
        self.rc_params['axes.prop_cycle'] = cycler(color=color_list)

        super().__init__(*args, **kwargs)

    def plot_diagnostic(self,
                        points: np.typing.ArrayLike,
                        labels: Optional[np.typing.NDArray[np.long]] = None
                        ) -> None:
        with plt.rc_context(self.rc_params):
            self._diagnostic(points, labels)

    def _diagnostic(self,
                    points: np.typing.ArrayLike,
                    labels: Optional[np.typing.NDArray[np.long]] = None
                    ) -> None:
        pass


class AngerDiagnostic(GenericDiagnostic):
    def _diagnostic(self,
                    points: np.typing.ArrayLike,
                    labels: Optional[np.typing.NDArray[np.long]] = None
                    ) -> None:
        ax = self.subplots()
        self.anger(ax, points, labels)

    def anger(self,
              ax: Axes,
              points: np.typing.ArrayLike,
              labels: Optional[np.typing.NDArray[np.long]] = None,
              limit_points: Optional[int] = 1000
              ) -> None:
        
        points = np.array(points, dtype=np.float64)

        if labels is None:
            labels = np.ones(np.shape(points)[0], dtype=np.long)

        labels_used = sorted(set(labels))  # type: ignore

        to_plot = np.random.choice(np.shape(points)[0],
                                   limit_points,
                                   replace=False)
        points = np.array(points, dtype=np.float64)[to_plot, :]
        labels = np.array(labels, dtype=np.long)[to_plot]

        x, y = self._anger_metric(points)

        for i in labels_used:
            ax.scatter(x[labels == i], y[labels == i], marker='o', s=0.25)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('T1 + T2 - (T3 + T4)')
        ax.set_ylabel('T2 + T3 - (T1 + T4)')

    @staticmethod
    def _anger_metric(points: np.typing.ArrayLike
                      ) -> tuple[np.typing.NDArray[np.float64],
                                 np.typing.NDArray[np.float64]]:
        if np.shape(points)[1] != 4:
            raise ValueError('Each event must have four values.')

        amplitude = np.sum(points, axis=1)
        amplitude[amplitude < 1] = np.nan
        x = np.dot(points, [1, 1, -1, -1]) / amplitude
        y = np.dot(points, [-1, 1, 1, -1]) / amplitude

        return x, y


class AmplitudeDiagnostic(GenericDiagnostic):
    def _diagnostic(self,
                    points: np.typing.ArrayLike,
                    labels: Optional[np.typing.NDArray[np.long]] = None
                    ) -> None:
        ax = self.subplots()
        self.amplitude_hist(ax, points, labels)

    def amplitude_hist(self,
                       ax: Axes,
                       points: np.typing.ArrayLike,
                       labels: Optional[np.typing.NDArray[np.long]] = None
                       ) -> None:
        points = np.array(points, dtype=np.float64)

        if labels is None:
            labels = np.ones(np.shape(points)[0], dtype=np.long)

        labels_used = sorted(set(labels))  # type: ignore
        amplitude = np.sum(points, axis=1)

        bins = list(np.arange(0., 1024., 2.))
        ax.hist([amplitude[labels == i] for i in labels_used],
                bins=bins, density=True, histtype='bar', stacked=True)


class FullDiagnostic(AngerDiagnostic, AmplitudeDiagnostic):
    def _diagnostic(self,
                    points: np.typing.ArrayLike,
                    labels: Optional[np.typing.NDArray[np.long]] = None
                    ) -> None:
        ax_hist, ax_anger = self.subplots(2)
        self.amplitude_hist(ax_hist, points, labels)
        self.anger(ax_anger, points, labels)


__all__ = ['GenericDiagnostic',
           'AngerDiagnostic',
           'AmplitudeDiagnostic',
           'FullDiagnostic']
