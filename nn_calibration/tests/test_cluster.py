"""Tests the identificaion of clusters from imager data streams.

The objective is to check that clustering works correctly under ideal
conditions. The key constituents of this operation are:
 - Each point correctly identifies a cluster on a background
 - The full script correctly time slices the data stream for each point.

TODO:
 - Simply tests
 - Explain tests
"""
import pytest
from unittest import mock
import numpy as np
import booms_packet
from nn_calibration import cluster
import xy_testing


@pytest.fixture
def mock_source():
    """Add a test set of parameters to the source parameter list.
    
    This sets up an environment where parameters are configured for the
    clusters in test_single_cluster. Can also be added to a test to use the
    test source name without errors.
    """
    cluster.optimal_params['test'] = {'ch': [140., 1800.],
                                      'energy': [140.*cluster.energy_scale,
                                                 1800.*cluster.energy_scale],
                                      'dbscan': {'eps': .375,
                                                 'frac_min_samples': 16}}


def test_single_cluster(mock_source):
    """Verify that clusters of PMT values can be identified above a background.

    Creates a data set of 2000 background counts. Two clusters are then added.
    Checks that identify_point will correctly report the cluster location and
    spread.
    """
    # Create a cluster on a background. Equal BG and points
    # Use gamma since it is always positive. Similar to normal for large values
    target_centers = np.double([[70, 30, 20, 10], [300, 400, 500, 600]]).T
    rng = np.random.default_rng()
    sample = rng.choice(target_centers, 2000, axis=1)
    sd = sample*.1
    sample = rng.gamma((sample/sd)**2, sd**2/sample)
    test_set = np.hstack((sample,
                          rng.exponential(50, size=(4, int(2000))))).T
    test_set = test_set.astype(np.uint16)

    # Run the test
    centers, spread = cluster.identify_point(test_set, source='test')

    # Check center and spread are within 5% + 1 channel
    assert np.all(np.abs(centers - target_centers) < (1+.05*target_centers))
    assert np.all(np.abs(spread - .1*target_centers) < (1+.005*target_centers))


def test_all_points(mock_source):
    """Verify run_all_points slices time in stream correctly.

    Mock the single point clustering and stream class. Be sure the time slice
    is being called with only the correct times for a grid.
    """
    stream = booms_packet.DataStream()
    test_grid = xy_testing.grid.nice_spacing['8x8']

    target_centers = np.double([[70, 30, 20, 10], [300, 400, 500, 600]]).T
    tubes = np.arange(1, 41, dtype=np.uint16).reshape((10, 4))

    # Mock
    stream.time_slice = mock.MagicMock(return_value=stream)
    stream.imager_tubes = mock.MagicMock(return_value=tubes)
    cluster.identify_point = mock.MagicMock(return_value=(target_centers,
        .1*target_centers))

    # Run
    x, y, clusters, spreads = \
        cluster.run_all_points(stream, grid=test_grid, duration=47, buffer=4,
                           plot=False, source='test')

    # Check results
    source_path = xy_testing.path.SourcePath(test_grid, time=47)

    # Check pmt values were collected
    stream.imager_tubes.assert_called()

    # Check that only the 64 points in the 8x8 grid were evaluated
    assert cluster.identify_point.call_count == 64

    # Check time slice calls. Should only be points in grid or removing old
    #   times from the stream.
    point_times = [mock.call(p.start + 4, p.end - 4) for p in source_path]
    point_times = point_times[2:] # Skip movement to first position
    for call in stream.time_slice.mock_calls:
        if 'in_place' in call.kwargs:
            continue

        assert call == point_times.pop(0)

    # Check returned grid points match 8x8 grid
    x_g, y_g = np.meshgrid(test_grid.x, test_grid.y)
    assert np.all(x == x_g)
    assert np.all(y == y_g)

    # Check that the identify_point are being added to the grid correctly
    for cluster_n in range(2):
        for tube in range(4):
            assert np.all(clusters[:,:,tube, cluster_n] \
                          == target_centers[tube, cluster_n])
            assert np.all(spreads[:,:,tube, cluster_n] \
                          == .1*target_centers[tube, cluster_n])
