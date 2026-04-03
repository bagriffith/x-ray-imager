from itertools import combinations
import logging
# from pkg_resources import Requirement, resource_filename
import numpy as np
import yaml
from sklearn.cluster import DBSCAN, KMeans
from . import plot

logger = logging.getLogger(__name__)

# param_file = resource_filename(Requirement.parse("."),
#                                'cluster_params.yaml')
# TODO Add cluster params as a resource filename
param_file = 'cluster_params.yaml'

with open(param_file, encoding='ascii') as f:
    optimal_params = yaml.safe_load(f)

for s in optimal_params.keys():
    optimal_params[s]['ch'] = [e//energy_scale
                               for e in optimal_params[s]['energy']]


def identify_clusters(points, channel_limits=(None, None), min_clusters=0):
    # points = stream.imager_tubes()
    logger.debug('identify_clusters points.shape: %s', points.shape)

    # Convert the type of `points`. Undo the rounding by the ADC.
    points_smoothed = np.float16(points) + np.random.uniform(0, 1, points.shape)

    # Total of all tubes is roughly proportional to the absorbed energy
    amplitude = np.sum(points, axis=1)

    # Limit energy band
    low_limit = channel_limits[0] if not None else 0.,
    high_limit = channel_limits[1] if not None else np.inf
    in_e_range = (amplitude > low_limit) & (amplitude < high_limit)


    # Use -2 as the default cluster id, marking points skipped by the search
    cluster_id = np.full(points.shape[0], -2, dtype=np.int8)

    cluster_id[in_e_range] = DBSCAN(metric='canberra')\
        .fit(points_smoothed[in_e_range]).labels_

    # Only positive labels are clusters. -1 marks noisy (background) points.
    in_cluster = cluster_id >= 0
    n_clusters = len(set(cluster_id[in_cluster]))
    logger.info('DBSCAN found %s clusters.', n_clusters)

    # Split into enough clusters with KMeans
    if n_clusters < min_clusters:
        logger.info('Min of %s clusters requested. Applying K-means.',
                    min_clusters)
        
        kmeans_model = KMeans(min_clusters, n_init=32)
        cluster_id[in_cluster] = \
            kmeans_model.fit(amplitude[in_cluster].reshape(-1, 1)).labels_
        n_clusters = min_clusters

    return cluster_id


def find_centers(points, labels):
    unique_label_ids = set(labels)
    n_clusters = len(unique_label_ids)
    centers = np.ndarray((n_clusters, 4), dtype=np.float64)
    spreads = np.ndarray((n_clusters, 4), dtype=np.float64)
    n_points = np.zeros(n_clusters, dtype=np.float64)
    for i in unique_label_ids:
        if i < 0:
            continue
        centers[i, :] = np.mean(points[labels == i], axis=0)
        spreads[i, :] = np.std(points[labels == i], axis=0)
        n_points[i] = np.sum(labels == i)
    return centers, spreads, n_points


def identify_point(points, plot=False):
    cluster_id = identify_clusters(points, channel_limits, min_clusters)
    centers, spreads, n_points = find_centers(points, cluster_id)

    cluster_amplitudes = np.sum(centers, axis=1)

    if plot:
        nn_calibration.plot.diagnostic(points, cluster_id)

    logger.debug('Contains %s events', np.sum(n_points))

    # Find the best mapping of clusters to energy lines
    # TODO Identify best energy match
    # TODO Add logging for fit quality
    min_error = np.inf
    best_mapping = None
    best_gain = None
    sorted_cluster_index = np.argsort(cluster_amplitudes)
    for trial_index in combinations(sorted_cluster_index):
        trial_amplitudes = cluster_amplitudes[trial_index]
        for gain_trial in np.arange(min_gain, max_gain, 0.1):
            error = np.linalg.norm(trial_amplitudes - gain_trial*line_energies)
            if error < min_error:
                min_error = error
                best_mapping = trial_index
                best_gain = gain_trial

    return centers[best_mapping], spreads[best_mapping], best_gain
