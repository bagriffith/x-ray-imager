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

"""Creates diagnostic plots for visualizing imager event groupings.

Typical usage example:
    data = np.loadtxt("imager-event-list.txt")
    group_ids = cluster_method.fit(data)
    diagnostic = FullDiagnostic()
    diagnostic.plot_diagnostic(data, group_ids)
    diagnostic.savefig('diagnostic.png')
"""
from typing import Optional
from importlib import resources
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import ArrayLike, NDArray
import x_ray_imager_bagriff
from x_ray_imager_bagriff.position_estimation import anger_basis

STYLE_FILE = str(resources.files(x_ray_imager_bagriff)
                 .joinpath('plot_style.mplstyle'))


class GenericIdentifyDiagnostic(Figure):
    """Generic figure for visualizing imager event groupings.
    
    A list of events and optionally, their group ID should be passed to
    plot_diagnostic(). This diagnostic can then be saved or displayed like
    any other pyplot figure. The specific diagnostic should be implemented
    as a subclass.
    """
    def __init__(self, *args,
                 rc_params: Optional[dict] = None,
                 **kwargs) -> None:
        """Initialize the diagnostic figure.

        Args:
            rc_params: A dictionary of matplotlib rc_params to use for this
                figure. If not provided, there is default style in this
                package. It's linked here by the variable STYLE_FILE.
        """
        self.rc_params = mpl.rc_params_from_file(STYLE_FILE,
                                                 fail_on_error=True)
        if rc_params is not None:
            self.rc_params.update(rc_params)

        color_list = [plt.get_cmap('magma')(i/5) for i in range(6)]
        self.rc_params['axes.prop_cycle'] = cycler(color=color_list)

        super().__init__(*args, **kwargs)

    def plot_diagnostic(self,
                        X: ArrayLike,  # pylint: disable=invalid-name
                        labels: Optional[NDArray[np.long]] = None
                        ) -> None:
        """Visualize the events and classifications.

        This calls _diagnostic which should be overloaded with the specific
        diagnostic.

        Args:
            X: Array of measurements. Shape is (n events, n detectors).
                See _identify.find_lines() for specifics.
            labels: Integer index of cluster identified for each measurement.
        """
        with plt.rc_context(self.rc_params):
            self._diagnostic(X, labels)

    def _diagnostic(self,
                    X: ArrayLike,  # pylint: disable=invalid-name
                    labels: Optional[NDArray[np.long]] = None
                    ) -> None:
        """Diagnostic implementation, to be overloaded.
        
        Args:
            points:
            labels:
        """
        _ = X
        _ = labels


class AngerDiagnostic(GenericIdentifyDiagnostic):
    """Scatter plot of events by Anger imager position, colored by id.

    Uses the simple x-ray imager positioning algorithm for an Anger imager,
    x = sum(detectors_plus_x) - sum(detectors_minus_x)
    y = sum(detectors_plus_y) - sum(detectors_minus_y)
    an then normalized by the sum of all detectore
    
    It is assumed here that the detectors are numbered
            +y
        (2) | (1)
        ----+----+x
        (3) | (0)
    """
    def _diagnostic(self,
                    X: ArrayLike,  # pylint: disable=invalid-name
                    labels: Optional[NDArray[np.long]] = None
                    ) -> None:
        ax = self.subplots()
        self.anger(ax, X, labels)

    def anger(self,
              ax: Axes,
              X: ArrayLike,  # pylint: disable=invalid-name
              labels: Optional[NDArray[np.long]] = None,
              limit_points: Optional[int] = 1000
              ) -> None:
        """Plot the anger diagnostic on the Axis provided.
        
        Args:
            ax: Axis to use for the scatter plot.
            X: Array of measurements. Shape is (n events, n detectors).
            labels: Integer index of cluster identified for each measurement.
        """
        X = np.array(X, dtype=np.float64)

        if labels is None:
            labels = np.ones(np.shape(X)[0], dtype=np.long)

        labels_used = sorted(set(labels))  # type: ignore

        to_plot = np.random.choice(np.shape(X)[0],
                                   limit_points,
                                   replace=False)
        X = np.array(X, dtype=np.float64)[to_plot, :]
        labels = np.array(labels, dtype=np.long)[to_plot]

        _, x, y = anger_basis(X)

        for i in labels_used:
            ax.scatter(x[labels == i], y[labels == i], marker='o', s=0.25)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('T1 + T2 - (T3 + T4)')
        ax.set_ylabel('T2 + T3 - (T1 + T4)')


class AmplitudeDiagnostic(GenericIdentifyDiagnostic):
    """Histogram of the sum of all detectors for each event, colored by id.
    """
    def _diagnostic(self,
                    X: ArrayLike,  # pylint: disable=invalid-name
                    labels: Optional[NDArray[np.long]] = None
                    ) -> None:
        ax = self.subplots()
        self.amplitude_hist(ax, X, labels)

    def amplitude_hist(self,
                       ax: Axes,
                       X: ArrayLike,  # pylint: disable=invalid-name
                       labels: Optional[NDArray[np.long]] = None
                       ) -> None:
        """Plot the event amplitude histogram on the Axis provided.
        
        Args:
            ax: Axis to use for the scatter plot.
            X: Array of measurements. Shape is (n events, n detectors).
            labels: Integer index of cluster identified for each measurement.
        """
        X = np.array(X, dtype=np.float64)

        if labels is None:
            labels = np.ones(np.shape(X)[0], dtype=np.long)

        labels_used = sorted(set(labels))  # type: ignore
        amplitude = np.sum(X, axis=1)

        bins = list(np.arange(0., 1024., 2.))
        ax.hist([amplitude[labels == i] for i in labels_used],
                bins=bins, density=True, histtype='bar', stacked=True)


class FullDiagnostic(AngerDiagnostic, AmplitudeDiagnostic):
    """Combine the Anger and Amplitude diagnostics in one figure."""
    def _diagnostic(self,
                    X: ArrayLike,  # pylint: disable=invalid-name
                    labels: Optional[NDArray[np.long]] = None
                    ) -> None:
        ax_hist, ax_anger = self.subplots(2)
        self.amplitude_hist(ax_hist, X, labels)
        self.anger(ax_anger, X, labels)


diagnostics = {'anger': AngerDiagnostic,
               'amplitude': AmplitudeDiagnostic,
               'full': FullDiagnostic}


__all__ = ['GenericIdentifyDiagnostic',
           'AngerDiagnostic',
           'AmplitudeDiagnostic',
           'FullDiagnostic',
           'diagnostics']
