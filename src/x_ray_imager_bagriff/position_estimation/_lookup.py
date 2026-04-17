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

import numpy as np
from sklearn.neighbors import KDTree
from x_ray_imager_bagriff.position_estimation import (
    PointEstimator,
    anger_basis
)


class PointLookup(PointEstimator):
    short_name = 'base_lookup'

    def __init__(self, channels, energies, positions):
        self.channels = np.array(channels)
        self.energies = np.array(energies)
        self.positions = np.array(positions)

        assert self.channels.shape[:-1] == self.energies.shape
        assert self.channels.shape[:-1] == positions.shape[:-1]
        assert self.channels.shape[-1] == 4
        assert self.positions.shape[-1] == 2

        self.channels = self.channels.reshape((-1, 4))
        self.energies = self.energies.reshape((-1))
        self.positions = self.positions.reshape((-1, 2))

        self.build()
        self.validate()

    def build(self):
        pass

    def validate(self):
        pass

    def lookup_index(self, channels):
        raise NotImplementedError('Default class')
        return np.zeros_like(channels, dtype='uint'), None

    def estimate_error(self, ind):
        return (np.full_like(ind, 10.),
                np.full_like(ind, 5.),
                np.full_like(ind, 5.))

    def get_value(self, channels, return_error=False):
        ind, weights = self.lookup_index(channels)

        if weights is None:
            e = self.energies[ind]
            x = self.positions[ind, 0]
            y = self.positions[ind, 1]
        else:
            total = np.sum(weights, axis=-1)
            # clean up the worst fits
            bad_fit = total == 0

            weights[bad_fit] = 1.
            total[bad_fit] = weights.shape[-1]
            weights = (weights.T/total).T

            e = np.sum(self.energies[ind]*weights, axis=-1)
            x = np.sum(self.positions[ind, 0]*weights, axis=-1)
            y = np.sum(self.positions[ind, 1]*weights, axis=-1)

        if return_error:
            if weights is None:
                # Lookup an estimate
                err_e, err_x, err_y = self.estimate_error(ind)
            else:
                err_e = .01 + np.sqrt(
                    np.sum((self.energies[ind].T - e)**2 * weights.T,
                           axis=0))
                err_x = .1 + np.sqrt(
                    np.sum((self.positions[ind, 0].T - x)**2 * weights.T,
                           axis=0))
                err_y = .1 + np.sqrt(
                    np.sum((self.positions[ind, 1].T - y)**2 * weights.T,
                           axis=0))

            return (e, x, y), (err_e, err_x, err_y)
        else:
            return e, x, y


class LookupGradError(PointLookup):
    short_name = 'base_lookup_grad'

    def __init__(self, channels, energies, positions):
        # energy_points, x_points, y_points = energies.shape
        # TODO, check shape
        # v = np.reshape(channels, (energy_points, x_points, y_points, 4))

        g = np.gradient(channels, energies[:, 0, 0],
                        positions[0, :, 0, 0],
                        positions[0, 0, :, 1],
                        axis=[0, 1, 2])
        g = np.divide(np.sqrt(channels/10), g)
        self.errors = np.zeros((g.shape[1]*g.shape[2]*g.shape[3], 3))
        for i in range(3):
            self.errors[:, i] = np.sqrt(np.sum(g[i]**2, axis=-1)).flatten()

        super().__init__(channels, energies, positions)


class TreeLookup(LookupGradError):
    short_name = 'tree'

    def build(self):
        self.kdtree = KDTree(self.channels, leaf_size=32, metric='euclidean')

    def lookup_index(self, channels):
        return self.kdtree.query(channels)[1][:, 0], None


class ProbLookup(LookupGradError):
    short_name = 'prob'

    def build(self):
        self.kdtree = KDTree(self.channels, leaf_size=32, metric='euclidean')

    def lookup_index(self, channels):
        gain2 = .04  # g squared
        ind = self.kdtree.query(channels, k=32)[1]

        diff = np.empty((ind.shape[0], 4, ind.shape[1]))
        for i in range(ind.shape[-1]):
            diff[:, :, i] = \
                (self.channels[ind[:, i]] - channels)**2 \
                / (gain2*self.channels[ind[:, i]])

        error = np.sum(diff, axis=1)

        return ind, np.exp(-error/2)


class AngerTreeLookup(LookupGradError):
    short_name = 'tree_anger'

    def build(self):
        self.kdtree = KDTree(anger_basis(self.channels),
                             leaf_size=32, metric='euclidean')

    def lookup_index(self, channels):
        return self.kdtree.query(anger_basis(channels))[1][:, 0], None


class AngerProbLookup(LookupGradError):
    short_name = 'prob_anger'

    def build(self):
        channels = self.channels
        self.kdtree = KDTree(anger_basis(channels),
                             leaf_size=32, metric='euclidean')

    def lookup_index(self, channels):
        gain2 = .04  # g squared
        ind = self.kdtree.query(anger_basis(channels),
                                k=64, sort_results=False,
                                return_distance=False)[1]

        diff = np.empty((ind.shape[0], 4, ind.shape[1]))
        for i in range(ind.shape[-1]):
            diff[:, :, i] = (self.channels[ind[:, i]] - channels)**2 \
                / (gain2*self.channels[ind[:, i]])

        error = np.sum(diff, axis=1)
        return ind, np.exp(-error/2)
