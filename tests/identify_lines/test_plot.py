import numpy as np
from x_ray_imager_bagriff.identify_lines import plot


def test_full_diagnostic(tmp_path):
    n_points = 1000
    means = (64, 128)
    np.random.seed(0)  # Keep the set consistent between tests
    example_set = np.concat(
        [np.random.randint(0, 256, size=(n_points, 4))] +
        [np.random.poisson(x, size=(n_points, 4)) for x in means]
    )
    cluster_labels = np.repeat(np.arange(len(means) + 1), n_points)

    fig = plot.FullDiagnostic()
    fig.plot_diagnostic(example_set, cluster_labels)
    plot_path = tmp_path / 'diagnostic.pdf'
    fig.savefig(plot_path)
    assert plot_path.exists()
