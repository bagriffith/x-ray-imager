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

"""Test x-ray imager plotting tools."""
import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.container import BarContainer
from matplotlib.patches import Polygon
import pandas as pd
from x_ray_imager_bagriff.position_estimation.plot import (
    ImagerAxes,
    ImagerFigure, SpectrumFigure, ImageHistFigure, ImageSpectrumFigure,
    ImagerAnimation
)

# For pytest fixtures
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


@pytest.fixture
def example_imager_data():
    """Example observation position and energy arrays."""
    n_points = 2048
    x = np.random.normal(25.0, 20.0, n_points)
    x[x < -70] = -70
    x[x > 70] = 70
    y = np.random.normal(-10.0, 20.0, n_points)
    y[y < -70] = -70
    y[y > 70] = 70
    energy = 600 * np.random.beta(2, 5, n_points)
    energy[:n_points//2] = np.random.normal(100, 10.0, n_points//2)
    return energy, x, y


def test_trunc_norm_cdf():
    """Test ``ImagerAxes.trunc_norm_params`` finds corrects parameters."""
    a = -1
    b = 1
    mu_param = -0.6
    sigma_param = 0.4
    # The calculation is simplified by saying the CDF and PDF at beta is zero.
    k = 0.242 / 0.841  # normal_pdf(-1) / (1 - normal_cdf(-1))
    mu_real = mu_param + sigma_param * k
    sigma_real = sigma_param * np.sqrt(1 - k - k**2 )

    mu_calc, sigma_calc = \
        ImagerAxes.trunc_norm_params(a, b, mu_real, sigma_real)

    assert mu_calc == pytest.approx(mu_param, 0.001)
    assert sigma_calc == pytest.approx(sigma_param, 0.001)


def test_energy_spectrum(tmp_path, example_imager_data):
    """Test ``ImagerAxes.energy_spectrum()``."""
    energy, x, y = example_imager_data
    _ = x, y

    plt.rcParams['text.usetex'] = True  # Needed for axis labels
    fig = plt.figure()
    ax = fig.add_subplot(axes_class=ImagerAxes)
    result = ax.energy_spectrum(energy)  # type: ignore

    assert isinstance(result, (BarContainer, Polygon, list))

    save_path = tmp_path / 'energy_spectrum.png'
    fig.savefig(save_path)
    assert save_path.exists()

    plt.close(fig)


def test_energy_spectrum_error(tmp_path, example_imager_data):
    """Test ``ImagerAxes.energy_spectrum()`` with errors."""
    energy, x, y = example_imager_data
    _ = x, y
    error = np.sqrt(energy)

    plt.rcParams['text.usetex'] = True  # Needed for axis labels
    fig = plt.figure()
    ax = fig.add_subplot(axes_class=ImagerAxes)
    bins = np.linspace(0, 600, 121)
    result = ax.energy_spectrum(energy,  # type: ignore
                                d_energy=error,
                                bins=bins)

    assert isinstance(result, (BarContainer, Polygon, list))

    save_path = tmp_path / 'energy_spectrum.png'
    fig.savefig(save_path)
    assert save_path.exists()

    plt.close(fig)


def test_image_hist(tmp_path, example_imager_data):
    """Test ``ImagerAxes.image_hist()``."""
    energy, x, y = example_imager_data
    _ = energy

    fig = plt.figure()
    ax = fig.add_subplot(axes_class=ImagerAxes)
    result = ax.image_hist(x, y)  # type: ignore

    assert isinstance(result, QuadMesh)

    save_path = tmp_path / 'image_hist.png'
    fig.savefig(save_path)
    assert save_path.exists()

    plt.close(fig)


def test_image_hist_error(tmp_path, example_imager_data):
    """Test ``ImagerAxes.image_hist()`` with errors."""
    _, x, y = example_imager_data
    dx = np.full_like(x, 20.0)
    dy = np.full_like(x, 20.0)

    fig = plt.figure()
    ax = fig.add_subplot(axes_class=ImagerAxes)
    result = ax.image_hist(x, y, d_x=dx, d_y=dy,  # type: ignore
                           bins=np.linspace(-70, 70, 141))

    assert isinstance(result, QuadMesh)

    save_path = tmp_path / 'image_hist.png'
    fig.savefig(save_path)
    assert save_path.exists()

    plt.close(fig)


def test_filter_by_energy(example_imager_data):
    """Test ``ImagerFigure._filter_by_energy()``."""
    energy, x, y = example_imager_data

    # Define an energy range
    energy_range = (100.0, 300.0)

    # Filter the data
    filtered_x, filtered_y = \
        ImagerFigure._filter_by_energy(energy, x, y, energy_range)

    in_range = (energy > energy_range[0]) & (energy < energy_range[1])

    # Verify that the filtered data is within the energy range
    assert np.all(x[in_range] == filtered_x)
    assert np.all(y[in_range] == filtered_y)


def test_imager_figures(tmp_path, example_imager_data):
    """Test ``SpectrumFigure.plot_observations()``."""
    energy, x, y = example_imager_data

    for fig_class in [SpectrumFigure, ImageHistFigure, ImageSpectrumFigure]:
        fig = fig_class(figsize=(8, 8))
        fig.plot_observations(energy, x, y, duration=1.0)

        save_path = tmp_path / f'{type(fig).__name__}.png'
        fig.savefig(save_path)
        assert save_path.exists()
        plt.close(fig)


def test_animation(tmp_path, example_imager_data):
    """Test ``ImagerAnimation``."""
    energy, x, y = example_imager_data
    df = pd.DataFrame({'energy': np.repeat(energy, 3),
                       'x': np.repeat(x, 3),
                       'y': np.repeat(y, 3),
                       't': np.repeat(np.arange(3), len(x))})

    fig = ImageSpectrumFigure()
    anim = ImagerAnimation(fig=fig, df=df, step_duration=1.0)

    save_path = tmp_path / 'animation.mp4'
    anim.save(filename=save_path, writer="ffmpeg")

    assert save_path.exists()
