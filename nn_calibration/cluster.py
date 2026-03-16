from itertools import permutations
import shutil
import logging
import pathlib
from pkg_resources import Requirement, resource_filename
import numpy as np
import yaml
from sklearn.cluster import DBSCAN, KMeans
import booms_packet
import xy_testing
import nn_calibration.plot

logger = logging.getLogger(__name__)

MAX_CLUSTER = 25_000
energy_scale = .6

param_file = resource_filename(Requirement.parse("BoomsTools"),
                               'nn_calibration/cluster_params.yaml')

with open(param_file, encoding='ascii') as f:
    optimal_params = yaml.safe_load(f)

for s in optimal_params.keys():
    optimal_params[s]['ch'] = [e//energy_scale for e in optimal_params[s]['energy']]


def identify_clusters(points, **kwargs):
    """Identify clusters in the tube channels.

    This will pick out all groups of points which are all close in the four
    tube channels. This first utilized the DBSCAN algorithm and then will apply
    KMeans on the sum of tube channels to get the correct number of clusters.
    Then for every cluster, the mean and standard deviation of points in the
    cluster are reported.

    Args:
        stream (DataStream):
        return_labels (bool): Should a list of event labels be returned with the
            means and standard deviations? Default value is False
        energy_peaks (int): The number of energies to identify. default 0
        channel_limits (float, float): bounds of channel for each peak
        dbscan: False to disable the DBSCAN step. Pass a dict to provides kwargs
            to sklearn.cluster.DBSCAN. frac_min_samples sets min_samples to the
            number of events // frac_min_samples.

    Returns:
        ndarray: An nx4 array of the mean of all clusters.
        ndarray: An nx4 array of the standard deviation of all clusters.
    """
    # points = stream.imager_tubes()
    logger.debug(f'identify_clusters points.shape: {points.shape}')

    # Correct for channel rounding to smooth point spread
    amplitude = np.sum(points, axis=1)
    labels = np.full(points.shape[0], -1, dtype=np.int8)

    # Run DBSCAN to find point clusters
    dbscan = kwargs.get('dbscan', dict())
    if dbscan or dbscan is dict():
        in_set = np.full(points.shape[0], True)

        if 'channel_limits' in kwargs:
            low, high = kwargs['channel_limits']
            if low is not None:
                in_set &= amplitude >= low
            if high is not None:
                in_set &= amplitude <= high

        idx = np.arange(points.shape[0])[in_set]

        if idx.shape[0] > MAX_CLUSTER:
            logger.info(f'Max clustering points exceeded. Using {MAX_CLUSTER} out of {points.shape[0]}.')

            idx = np.random.choice(idx, size=MAX_CLUSTER, replace=False)

        if 'frac_min_samples' in dbscan:
            if 'min_samples' in dbscan:
                logger.warning('frac_min_samples overrides min_samples.')

            dbscan['min_samples'] = int(idx.shape[0] / dbscan.pop('frac_min_samples'))

            if dbscan['min_samples'] < 1:
                logger.warning('DBSCAN min_samples will be too small')
                dbscan['min_samples'] = 1

        cluster_model = DBSCAN(metric='canberra',
                               **kwargs.get('dbscan', dict()))
        search_set = np.float16(points[idx])
        search_set += np.random.uniform(0, 1, search_set.shape)
        search_set += 2

        clustering = cluster_model.fit(search_set)

        # Points excluded are flagged with -2
        labels = np.full(points.shape[0], -2, dtype=np.int8)
        labels[idx] = clustering.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0) - (1 if -2 in labels else 0)
    logger.info(f'DBSCAN found %s clusters.', n_clusters)

    # Split into enough clusters with KMeans
    target_n = len(kwargs.get('ch', []))
    
    if 'energy_peaks' in kwargs:
        target_n = int(kwargs['energy_peaks'])

    if n_clusters < target_n:
        in_set = labels == -1 if n_clusters == 0 else (labels != -1)

        kmeans_model = KMeans(target_n, n_init=32)
        clustering_kmeans = kmeans_model.fit(amplitude[in_set].reshape(-1, 1))

        labels[in_set] = clustering_kmeans.labels_
        n_clusters = target_n

    # Crete output
    if n_clusters > 0:
        centers = np.ndarray((n_clusters,4), dtype=np.float64)
        spreads = np.ndarray((n_clusters,4), dtype=np.float64)
        n_points = np.zeros(n_clusters, dtype=np.float64)
        for i in set(labels):
            if i < 0:
                continue
            centers[i,:] = np.mean(points[labels == i], axis=0)
            spreads[i,:] = np.std(points[labels == i], axis=0)
            n_points[i] = np.sum(labels == i)
    else:
        centers = np.zeros((1,4), dtype=np.float64)
        spreads = np.zeros((1,4), dtype=np.float64)
        n_points = 0

    if kwargs.get('return_labels', False):
        return (centers, spreads, n_points), (points, labels)
    else:
        return centers, spreads, n_points


def identify_point(points, plot=True, label=None, source=None):
    """Find the center and spread of the source for a packet DataStream.

    Args:
        stream (booms_packet.DataStream): Packet stream to identify the source
            points.
        plot (bool): Should diagnostic plots be made for the point
        label (str): Label for this point. Used in plotting.
        source (dict): 

    Returns:
        numpy.ndarray: Central tube channels of the highest energy cluster.
        numpy.ndarray: Standard deviation of the highest energy cluster.
    """
    cluster_kwargs = optimal_params.get(source, dict())
    target_ch = cluster_kwargs.get('ch')

    results = identify_clusters(points, return_labels=plot, **cluster_kwargs)

    if plot:
        (centers, spreads, n_points), (points, labels) = results
        nn_calibration.plot.diagnostic(points, labels, label)
    else:
        centers, spreads, n_points = results

    logger.debug(f'Contains {np.sum(n_points)} events')

    # Sum of channels, should be approximately proportional to the total energy deposited
    total_energy = np.sum(centers, axis=1)

    if target_ch is None:
        target_clusters = [np.argmax(total_energy)]
    else:
        logger.debug(f'{len(total_energy)} clusters found'
                     f' for {len(target_ch)} energies')
        lowest_error = None

        cluster_indexes = list(range(len(total_energy)))
        # If three weren't enough clusters, pad with None to decide what to drop
        if len(cluster_indexes) < len(target_ch):
            cluster_indexes.extend([None]*(len(target_ch) - len(cluster_indexes)))

        for cluster_assignment in permutations(cluster_indexes, len(target_ch)):
            distance = np.square([((total_energy[i] - e)/e if i is not None else 0) for i, e in zip(cluster_assignment, target_ch)])
            small_set_penalty = -np.log([n_points[i] for i in cluster_assignment]/np.max(n_points))*len(np.nonzero(distance))/30
            error = np.sum(distance + small_set_penalty)

            if (True if lowest_error is None else error < lowest_error):
                lowest_error = error
                target_clusters = cluster_assignment

    center_comb = np.zeros((*(centers[0].shape), len(target_clusters)))
    spread_comb = np.zeros((*(spreads[0].shape), len(target_clusters)))
    for i, t in enumerate(target_clusters):
        if t is not None:
            center_comb[:,i] = centers[t]
            spread_comb[:,i] = spreads[t]
    return center_comb, spread_comb


def run_all_points(stream, grid, duration, buffer=None, plot=True, source=None):
    """Find the most probably set of channels for all points in a calibration grid.

    Args:
        dat_file
        grid (xy_testing.Grid): Grid used to generate the gcode for the
            calibration.
        duration (int): The number of seconds at one point.
        buffer (int): Optional, the time to ignore at the beginning and end of
            duration. This provides space for errors in timing. By default, it
            5% of the duration.
        plot (bool): Produce diagnostic figure? Default is true
        print_status (bool): Print out the results of each points while running.
            Default is True.

    Returns:
        numpy.ndarray: n x n x 2 array of x, y position for n x n grid
        numpy.ndarray: n x n x 4 array of the mean channel for the 4 tubes.
        numpy.ndarray: n x n x 4 array of the standard deviation channel for the 4 tubes.
    """
    if buffer is None:
        logging.info('Default 5\% buffer used')
        buffer = round(.05*duration)

    target_ch = optimal_params[source].get('ch') if source in optimal_params else None
    target_clusters = 1 if target_ch is None else len(target_ch)

    mesh = np.meshgrid(grid.x, grid.y)
    center = np.zeros((*mesh[0].shape, 4, target_clusters), dtype=np.float64)
    spread = np.zeros((*mesh[0].shape, 4, target_clusters), dtype=np.float64)

    source_path = xy_testing.path.SourcePath(grid, duration)
    for n, point in enumerate(source_path):
        if (source_path.row is None) or (source_path.column is None):
            continue # Skip until real points start

        i = source_path.column
        j = source_path.row

        logger.info('Running point %s/%s', n, mesh[0].shape[0]*mesh[0].shape[1])
        start = point.start + buffer
        end = point.end - buffer
        assert start < end
        logger.info('From time %s to %s', start, end)

        stream.time_slice(point.start+buffer, in_place=True)
        point_stream = stream.time_slice(start, end)
        plot_ij = plot and (i % 4 == 0) and (j % 4 == 0)

        # logger.warning('Skipping points for testing')
        # if j < 12:
        #     continue
        center[i, j, :, :], spread[i, j, :, :] = \
            identify_point(point_stream.imager_tubes(), plot_ij,
                           label=('' if source is None else f'{source}_') + f'{i:02d}_{j:02d}',
                           source=source)

    return mesh[0], mesh[1], center, spread


def run_file(dat_file, set_params=None, plot=True):
    """Find and plot all point tube channels

    Args:
        dat_file (str): Path to the dat file for the calirbation run
        run_params (dict): Dictionary of parameters for the calibration setup.
            Can include 'nice_spacing' and 'grid' which match the settings from
            the grid class in the xy_testing package. 'duration' is the integer
            seconds as an int spent at each calibration position. 'buffer' is
            an integer number of seconds at the beginning and end which is not
            used to identify clusters.
    
        Returns:
            numpy.ndarray: Energy of target lines for the source in keV. Length
                is n_energy.
            numpy.ndarray: x & y source positions for all n_pts calibration
                points. Shape (2, n_energy, n_pts).
            numpy.ndarray: Channels that are the median of the cluster
                identified for the selected energy and position. Shape is
                (4, n_energy, n_pts).
            numpy.ndarray: Standard deviation of channels that are in the
                cluster identified for the selected energy and position. Shape
                is (4, n_energy, n_pts).
    """
    run_params = {'nice_spacing': True,
                  'grid': '13x13',
                  'duration': 132,
                  'buffer': 6}

    if set_params is not None:
        run_params.update(set_params)

    source = run_params.get('source')

    if run_params['nice_spacing']:
        grid = xy_testing.grid.nice_spacing[run_params['grid']]
    else:
        grid = xy_testing.grid.even_spacing[run_params['grid']]

    stream = booms_packet.DatFileStream(dat_file)
    x, y, centers, spreads = \
        run_all_points(stream, grid, run_params['duration'],
                       run_params['buffer'], plot, source=source)

    target_ch = None
    if source in optimal_params:
        target_ch = optimal_params[source].get('energy')
    if target_ch is None:
        target_ch = [0.]
    if plot:
        for i, e in enumerate(target_ch):
            label = ('' if source is None else f'{source}_') + f'{e:.0f}keV'
            nn_calibration.plot.grid(x, y, centers[:,:,:,i], spreads[:,:,:,i], label)

    centers = np.reshape(centers, (-1, 4, centers.shape[-1]))
    centers = np.moveaxis(centers, 0, 1)
    spreads = np.reshape(spreads, (-1, 4, spreads.shape[-1]))
    spreads = np.moveaxis(spreads, 0, 1)
    return centers, spreads


def identify_and_save(sidecar, plot=False):
    dat_path, run_info = nn_calibration.sidecars.load(sidecar)
    clusters, _ = run_file(dat_path, run_info, plot=plot)

    # Save Results
    cluster_path = pathlib.Path('./clusters') / (sidecar.stem + '_cluster.txt')
    cluster_path.parent.mkdir(exist_ok=True)
    logger.info('Saving clusters to %s', cluster_path.resolve())
    save_centers(cluster_path, sidecar, clusters)


def save_centers(cluser_path, sidecar_path, clusters):
    cluser_path = pathlib.Path(cluser_path)
    sidecar_path = pathlib.Path(sidecar_path)
    # TODO, check numpy array shape (4 channels, pts, clusters)
    shutil.copy(str(sidecar_path.absolute()),
                str(cluser_path.with_suffix('.yaml')))
    for source_n in range(clusters.shape[-1]):
        np.savetxt(cluser_path.with_suffix(f'.{source_n}.txt'),
                   np.transpose(clusters[:,:,source_n]))


def load_center(cluser_sidecar, mask=False):
    energies, positions, energy_numbers = \
        nn_calibration.sidecars.load_info(cluser_sidecar, mask=mask, label=True)
    n_energies = len(energies)
    n_x = positions[0].shape[0]
    n_y = positions[1].shape[1]

    a = np.zeros((n_x, n_y, 4, n_energies))
    for i in range(n_energies):
        cluser_path = cluser_sidecar.with_suffix(f'.{energy_numbers[i]}.txt')
        a[:,:,:,i] = np.reshape(np.loadtxt(cluser_path), (n_x, n_y, 4))
    return a


def load_all_centers(sidecar_list, mask=False):
    logger.debug(sidecar_list)
    clusters =[load_center(f, mask) for f in sidecar_list]
    logger.debug([x.shape for x in clusters])
    a = np.concatenate(clusters, axis=3)
    return a


def grab_list(list_path):
    target_files = []
    with open(list_path) as f:
        for line in f.readlines():
            path_str = line.strip()
            if path_str[0] == '#':
                continue
            target_files.append(pathlib.Path(path_str))
            if not target_files[-1].exists():
                raise RuntimeError('File not found: '+str(target_files[-1]))
    if len(target_files) == 0:
        raise ValueError('No files found.')
    return target_files