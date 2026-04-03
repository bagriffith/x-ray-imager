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

"""Produce the mean response from each gamma line in a list of events

1. Applies Filters
2. Clusters Events
3. Attributes clusters to spectral lines
4. Calculates mean response and other statistics
"""
import logging
from typing import Optional
from itertools import combinations
import numpy as np


def find_centers(points: np.typing.NDArray[np.long],
                 labels: np.typing.NDArray[np.long]
                 ) -> tuple[np.typing.NDArray[np.float64],
                            np.typing.NDArray[np.float64],
                            np.typing.NDArray[np.long]]:
    """Find mean, spread, and count of each point group.

    Args:
      points:
        Array of detector values for a set of events.
        Values are expected to be integers (discretized into channels).
        Shape should be (Number of Events, Number of Detectors).
      labels:
        Integer indexed group identity of each event. It is assumed that all
        labels from 0 to max(labels) are options, even if not used. Negative
        labels may be included, but will be ignored.

    Returns:
      A tuple (centers, spreads, n_points), where centers is an am Array of
      the mean value for each detector in each labeled group, spreads is the
      same but for standard deviation, and n_points is the count in each group.
    """
    unique_label_ids = set(labels)
    n_groups = max(max(unique_label_ids) + 1, 0)  # Number of groups
    n_events, n_detectors = np.shape(points)      # Number of PMTs/SiPMs

    if n_detectors != 4:
        # BOOMS uses 4 PMTs. Other projects may need a different number, but
        # a very large number is likely a transposed array.
        logging.warning("%s values per event, but 4 are expected.",
                        n_detectors)

    if len(labels) != n_events:
        raise ValueError(f"{n_events}"
                         " events doesn't match "
                         f"{len(labels)} labels.")

    centers = np.ndarray((n_groups, n_detectors), dtype=np.float64)
    spreads = np.ndarray((n_groups, n_detectors), dtype=np.float64)
    n_points = np.zeros(n_groups, dtype=np.long)
    for i in unique_label_ids:
        if i >= 0:
            centers[i, :] = np.mean(points[labels == i], axis=0)
            spreads[i, :] = np.std(points[labels == i], axis=0)
            n_points[i] = np.sum(labels == i)
    return centers, spreads, n_points


def match_energy(mean_response: np.typing.NDArray[np.float64],
                 energies: np.typing.NDArray[np.float64],
                 gain_range: Optional[tuple[float, float]] = None
                 ) -> tuple[np.typing.NDArray[np.long], float]:
    """Matches a list of spectral lines with the mean detector response.

    This takes in a list of mean detector responses. The sum of all
    detector values gives a total that should be proportional to the
    total light in the event, and the x-ray energy absorbed. Not all
    mean responses given may have an associated line. The ones that
    do should all have the same energy scaling.

    Args:
      mean_response:
        An array of detector values associated with the mean of some feature
        in a calibration set. len(energies) of these are from the gamma
        spectral lines listed there, and rest are other features.
      energies:
        An array listing some of the spectral lines captured. There should be
        fewer than the number of mean responses. Specific units are not
        required, but will change the meaning of the gain returned.
      gain_range:
        Tuple setting the minimum and maximum gain that will be accepted as
        for the returned association.

    Returns:
        A tuple (matched_index, gain). matched_index is a Array of indexes
        that selects the mean_response that best matches energy in that same
        array position. gain is the associated dector response / energy, in
        the units used for those arguments, associated with best mapping.
    """
    amplitudes = np.sum(mean_response, axis=1)
    energies_sorted = np.sort(energies)  # Sort in case they aren't already

    if gain_range is None:
        # If not provided, use the largest conceivable range.
        gain_range = (min(amplitudes) / max(energies),
                      max(amplitudes) / min(energies))

    # Find the best mapping of clusters to energy lines
    min_error = np.inf
    best_mapping = np.full((len(energies)), -1, dtype=int)
    best_gain = np.nan

    for trial_index in combinations(np.argsort(amplitudes), len(energies)):
        trial_index = np.array(trial_index, dtype=int)
        trial_amplitudes = amplitudes[trial_index]
        for gain_trial in np.arange(*gain_range, 0.1):
            diff = trial_amplitudes - gain_trial*energies_sorted
            error = np.linalg.norm(diff)
            if error < min_error:
                logging.debug('Better index found: %s (error %s < %s)',
                              trial_index, error, min_error)
                min_error = error
                best_mapping = trial_index
                best_gain = gain_trial

    if np.isinf(min_error):
        raise RuntimeError('Best match energy not found.')

    return best_mapping, best_gain
