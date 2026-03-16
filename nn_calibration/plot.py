import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import pathlib
from importlib import resources as impresources
import nn_calibration

OUTPUT_DIR = pathlib.Path('./figs')
STYLE_FILE = str(impresources.files(nn_calibration) / 'plot_style.mplstyle')

def histogram(ax, tube_ch, labels=None):
    """Make an energy histogram of each cluster.

    Args:
        ax (matplotlib.axes.Axes): pyplot axis to plot on.
        tube_ch (array_like): nx4 array of tube channels with a row for each
            detection.
        labels (array): n length array with cluster identifications. If not
            provided, all events will be plotted.
    """
    if labels is None:
        labels = np.zeros(len(tube_ch), dtype=np.int8)

    amplitude = np.sum(tube_ch, axis=1)
    clusters = [amplitude[labels == i] for i in sorted(set(labels))]
    bins = np.arange(0, 1024, 2)
    ax.hist(clusters, bins=bins, density=True, histtype='bar', stacked=True)
    ax.set_yscale('log')
    ax.set_xlabel('T1 + T2 + T3 + T4')
    ax.set_ylabel('Fraction of events / channel')


def scatter(ax, tube_ch, labels=None):
    """Make a labeled scatter plot of the relative tube channels.

    Args:
        See histogram.
    """
    if labels is None:
        labels = np.zeros(len(tube_ch), dtype=np.int8)

    to_plot = np.random.choice(len(tube_ch), 1000, replace=False)
    tube_ch = tube_ch[to_plot, :].astype(np.int16)
    labels = labels[to_plot].astype(np.int16)

    amplitude = np.sum(tube_ch, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        x = (tube_ch[:,0] + tube_ch[:,1]
             - tube_ch[:,2] - tube_ch[:,3]) / amplitude
        y = (tube_ch[:,1] + tube_ch[:,2]
             - tube_ch[:,0] - tube_ch[:,3]) / amplitude

    for i in sorted(set(labels)):
        ax.scatter(x[labels == i], y[labels == i], marker='o', s=0.25)

    ax.set_aspect('equal')
    ax.set_xlabel('T1 + T2 - (T3 + T4)')
    ax.set_ylabel('T2 + T3 - (T1 + T4)')

@mpl.rc_context(fname=STYLE_FILE)
def diagnostic(tube_ch, labels=None, name=None):
    """Sake a PDF with both diagnostic plots.

    Args:
        tube_ch (array_like): nx4 array of tube channels with a row for each
            detection.
        labels (array): n length array with cluster identifications. If not
            provided, all events will be plotted.
        name (name): identifier for this point set.
    """
    mpl.rcParams['axes.prop_cycle'] = cycler(color=[plt.get_cmap('magma')(i/5) for i in range(6)])

    fig, (ax_e, ax_xy) = plt.subplots(2, figsize=(5.5, 7))
    to_plot = labels != -2 # Exclude skipped points labeled with -2
    histogram(ax_e, tube_ch[to_plot], labels[to_plot])
    scatter(ax_xy, tube_ch[to_plot], labels[to_plot])

    base = '' if name is None else str(name)
    output = OUTPUT_DIR / 'cluster_diagnostic'\
        / f'BOOMS_calibration-{base}.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)
    plt.clf()


@mpl.rc_context(fname=STYLE_FILE)
def grid_wireframe(x, y, channels, spreads=None, label=None):
    """Plot of the tube channels vs source xy
    """
    label = '' if label is None else str(label)

    fig = plt.figure(figsize=(3.5, 4.5))
    fig.subplots_adjust(hspace=.15)

    ax_n = [4, 2, 1, 3]

    for i, ax_i in enumerate(ax_n):
        ax = fig.add_subplot(2, 2, ax_i, projection='3d')
        ax.set_title(f'PMT {i+1}')
        ax.plot_wireframe(x, y, channels[:,:,i], lw=.25, color='k')

    output = OUTPUT_DIR / f'grid_diagnostic/{label}_wireframe.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight')


@mpl.rc_context(fname=STYLE_FILE)
def grid_colormesh(x, y, channels, spreads=None, label=None):
    """Plot of the tube channels vs source xy
    """
    label = '' if label is None else str(label)

    fig = plt.figure(figsize=(4.0, 4.5))
    fig.subplots_adjust(hspace=.15)
    ax_n = [4, 2, 1, 3]

    lims = {'vmax': np.nanmax(channels),
            'vmin': np.nanmin(channels)}

    for i, ax_i in enumerate(ax_n):
        ax = fig.add_subplot(2, 2, ax_i)
        ax.set_title(f'PMT {i+1}')
        ax.set_aspect('equal')
        p = ax.pcolormesh(x, y, channels[:,:,i].T, cmap='magma', **lims)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.05, 0.05, 0.9])
    fig.colorbar(p, cax=cbar_ax)

    output = OUTPUT_DIR / f'grid_diagnostic/{label}_colormesh.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight')


@mpl.rc_context(fname=STYLE_FILE)
def grid(x, y, channels, spreads=None, label=None):
    grid_wireframe(x, y, channels, spreads, label)
    grid_colormesh(x, y, channels, spreads, label)


@mpl.rc_context(fname=STYLE_FILE)
def basis(components, shape, label=None, mode='wire'):
    """Plot of the tube channels vs source xy
    """
    label = '' if label is None else '_' + str(label)

    n = components.shape[1]
    columns = math.floor(np.sqrt(n))
    rows = math.ceil(n/columns)
    subplot_kw = dict()
    if mode == 'wire':
        subplot_kw['projection'] = '3d'
    fig, axs = plt.subplots(rows, columns, figsize=(3*columns, 2*rows),
                            subplot_kw=subplot_kw)

    n_x, n_y = tuple(shape)
    x, y = np.meshgrid(np.arange(n_x), np.arange(n_y))
    components = np.reshape(components, (n_x, n_y, -1))
    max_value = np.max(np.max(components))
    lims = {'vmax': max_value,
            'vmin': -max_value}

    for i, ax in enumerate(axs.flatten()[:n]):
        ax.set_aspect('equal')
        if mode == 'wire':
            ax.plot_wireframe(x, y, components[:,:,i], lw=.25, color='k')
        elif mode == 'mesh':
            ax.pcolormesh(x, y, components[:,:,i].T, cmap='RdBu', **lims)
        else:
            raise ValueError('Not valid plot mode: ' + str(mode))

    output = OUTPUT_DIR / f'components_{mode}{label}.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight')


@mpl.rc_context(fname=STYLE_FILE)
def pca_interp(x, y_set, fit, label=''):
    intercept, slope = fit
    terms = y_set.shape[0]
    columns = math.floor(np.sqrt(terms))
    rows = math.ceil(terms/columns)
    fig, axs = plt.subplots(rows, columns, figsize=(3*columns, 2*rows))
    fig.subplots_adjust(hspace=.35)

    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    x_hr = np.linspace(x_min - .05*(x_max - x_min),
                       x_max + .05*(x_max - x_min))

    for i, ax in enumerate(axs.flatten()[:terms]):
        ax.set_xlim(x_hr[0], x_hr[-1])
        ax.scatter(x, y_set[i, :], s=2)
        ax.set_title(f'Comp. {i}')

        ax.plot(x_hr, slope[i]*x_hr + intercept[i], ls='--', lw=.5)
    
    for ax in axs.flatten()[terms:]:
        ax.remove()

    output = OUTPUT_DIR / f'slope_{label}.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight')
