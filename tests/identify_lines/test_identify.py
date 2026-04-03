import pytest
import numpy as np
from x_ray_imager_bagriff.identify_lines\
    import find_centers, match_energy


def test_find_centers():
    """Tests `find_centers()` for a Poisson distribution."""
    n_points = 10_000
    mean = 256
    std = 16    # Sqrt(256)
    np.random.seed(0)  # Keep the set consistent between tests
    example_set = np.random.poisson(mean, size=(n_points, 4))
    example_labels = np.zeros(n_points, dtype=int)

    c, s, n = find_centers(example_set, example_labels)

    assert c == pytest.approx(np.full((1, 4), mean), 0.05)
    assert s == pytest.approx(np.full((1, 4), std), 0.05)
    assert n == np.array([n_points])


def test_match_energy():
    """Tests `match_energy()` for evenly spaced centers.
    """
    example_centers = np.transpose([np.arange(100., 1000., 100.)]*4)
    example_energies = np.array([30., 80.])

    idx, g = match_energy(example_centers, example_energies, (20, 80))

    assert np.all(idx == [2, 7])
    assert g == pytest.approx(40.)
