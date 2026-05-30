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

"""Tools to find the response for each gamma line from a source.

Typical usage example:

    data = np.loadtxt("imager-event-list.txt")
    cluster_method = sklearn.cluster.KMeans()  # Or some other algorithm
    source = SourceParams.get_source('Am241')

    responses = source_identify_all(data, cluster_method, source, (1.0, 4.0))
"""
import logging
from typing import Optional
from itertools import combinations
import numpy as np
from numpy.typing import NDArray
from sklearn.base import ClusterMixin
from x_ray_imager_bagriff.identify_lines import (
    SourceParams,
    check_gain_range
)
from x_ray_imager_bagriff.identify_lines.plot import GenericIdentifyDiagnostic

logger = logging.getLogger(__name__)


def find_lines(X: NDArray[np.long],  # pylint: disable=invalid-name
               cluster_method: ClusterMixin,
               source: SourceParams,
               gain_range: Optional[tuple[float, float]] = None,
               diagnostic: Optional[GenericIdentifyDiagnostic] = None
               ) -> NDArray[np.float64]:
    """Finds the mean value associated with every source gamma line.

    Args:
        X: Array of measurements. Shape is (n events, n detectors). Integer
            values are expected from the detectors. Noise up to 1 is added to
            each point when converted to a float for clustering.
        cluster_method: Clustering algorithm from scikit-learn.clustering
            or otherwise compatible with ClusterMixin. fit() is called
            to assign labels, ideally with one cluster plus lines and
            background points as -1. If more labels are assigned, 
            match_energy() selects the best fit.
        source: Parameters for the gamma source used. Should contain all
            peaks in the gamma spectrum, even if they're not from decays
            directly, like a Compton edge.
        gain_range: The max and min reasonable gain (detector units / keV).
        diagnostic: Figure type object with a plot_diagnostic(X, labels).
            Useful to check that the cluster_method is accurately assigning
            gamma lines to the measurements.

    Returns:
        Array of mean detector responses. Shape is
        (n detectors, n gamma lines)
    """
    if gain_range is not None:
        gain_range = check_gain_range(gain_range=gain_range)

    logger.debug('Data shape %s', X.shape)
    if X.shape[1] != 4:
        # Project specific warning. Other imagers may need this removed.
        logger.warning('Expected 4 detectors, but got %s', X.shape[1])

    # Apply noise to create a smooth distribution for clustering.
    # Serves to approximately undo discretization by the imager detectors.
    continuous = X + np.random.uniform(size=np.shape(X))
    in_range = source.get_filter(continuous, gain_range=gain_range)

    cluster_method.fit(continuous[in_range])

    if diagnostic is not None:
        diagnostic.plot_diagnostic(X[in_range], cluster_method.labels_)
        diagnostic.savefig(f'./{source.name}-diagnostic.png', dpi=300)

    cluster_means = line_means(X[in_range], cluster_method.labels_)
    matched_labels = match_energy(cluster_means, source.energies, gain_range)

    return cluster_means[matched_labels]


def line_means(X: NDArray[np.long],  # pylint: disable=invalid-name
               labels: NDArray[np.long]
               ) -> NDArray[np.float64]:
    """Find mean for each point group.

    Args:
        X: Array of measurements. See find_lines() for more.
        labels: Integer index of cluster identified for each measurement. 
            Negative labels may be included, but will be ignored. Numpy
            warnings raised if non-zero labels < max(labels) are skipped.

    Returns:
        Array of the mean value for each detector in each labeled group.
        Shape is (max(labels), n detectors).

    Raises:
        ValueError: len(labels) != measurements in X.
    """
    unique_label_ids = set(labels)
    n_clusters = max(max(unique_label_ids) + 1, 0)  # Number of groups

    if X.shape[1] != 4:
        # Project specific warning. Other imagers may need this removed.
        logger.warning('Expected 4 detectors, but got %s', X.shape[1])

    if len(labels) != X.shape[0]:
        raise ValueError(f"{X.shape[0]} events != {len(labels)} labels.")

    return np.array([np.mean(X[labels == i], axis=0)
                     for i in range(n_clusters)])


def match_energy(mean_response: NDArray[np.float64],
                 energies: NDArray[np.float64],
                 gain_range: Optional[tuple[float, float]] = None
                 ) -> tuple[NDArray[np.long], float]:
    """Matches a set of energies with best fit detector response.

    For each mean response, the sum of all detectors should be proportional
    to the energy absorbed. The association of responses that best fit the
    energies with a linear gain, in the provided gain_range, are selected.

    Args:
        mean_response: An array of detector values associated with the mean
            of some feature in a calibration set. len(energies) of these are
            from the gamma spectral lines listed there, and rest are other
            features.
        energies: An array listing some of the spectral lines captured. There
            should be fewer than the number of mean responses. Specific
            units are not required, but will change the meaning of the gain
            returned.
        gain_range: Tuple setting the minimum and maximum gain that will be
            accepted as for the returned association.

    Returns:
        Array of indexes that selects best matched mean_response.
    
    Raises:
        RuntimeError: If no fit between responses and energies is found.
    """
    amplitudes = np.sum(mean_response, axis=1)
    energies_sorted = np.sort(energies)  # Sort in case they aren't already

    if gain_range is None:
        # If not provided, use the largest conceivable range.
        gain_range = (min(amplitudes) / max(energies),
                      max(amplitudes) / min(energies))
    else:
        gain_range = check_gain_range(gain_range=gain_range)

    # Find the best mapping of clusters to energy lines
    min_error = np.inf
    best_mapping = np.full((len(energies)), -1, dtype=int)
    best_gain = np.nan
    logger.debug('Finding best fit between %s and %s', amplitudes, energies)

    # Try all combinations energy-response associations.
    # Progressively fits with a lower RMS deviation from a linear gain.
    for trial_index in combinations(np.argsort(amplitudes), len(energies)):
        # Make the index work for a numpy array.
        trial_index = np.array(trial_index, dtype=int)
        logger.debug('Amplitudes: %s', amplitudes[trial_index])

        trial_gain = np.mean(amplitudes[trial_index] / energies_sorted)
        # Use the closest boundary if mean_gain is outside the gain_range.
        trial_gain = min(max(trial_gain, gain_range[0]), gain_range[1])

        calculated_amplitude = trial_gain*energies_sorted
        error = np.linalg.norm(amplitudes[trial_index] - calculated_amplitude)
        if error < min_error:
            logger.debug('Better index found: %s (error %s < %s)',
                            trial_index, error, min_error)
            min_error = error
            best_mapping = trial_index
            best_gain = trial_gain

    # This should only be used if mean_response or energies are malformed.
    if np.isinf(min_error):
        raise RuntimeError('Best match energy not found.')

    logger.info('Best fit gain is %s', best_gain)
    return best_mapping
