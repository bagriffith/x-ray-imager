import numpy as np
import pickle
import builtins
from sklearn.neighbors import KDTree, BallTree
import sklearn
from tqdm import tqdm
import matplotlib.pyplot as plt
import nn_calibration

safe_builtins = {
    'range',
    'complex',
    'set',
    'frozenset',
    'slice',
}


def from_interpolation(lookup_class, interp, points):
    assert issubclass(lookup_class, PointEstimator)
    energy_points, x_points, y_points = points
    e = np.linspace(0, 600, energy_points+1)[1:]
    x = np.linspace(-70, 70, x_points)
    y = np.linspace(-70, 70, y_points)

    energies, x_g, y_g = np.meshgrid(e, x, y, indexing='ij')
    channels = interp(energies, x_g, y_g)

    return lookup_class(channels, energies, np.moveaxis([x_g, y_g], 0, -1))


def anger_basis(channels):
    amp = np.sum(channels, axis=-1)

    x = (channels[..., 0] + channels[..., 1] - \
        (channels[..., 2] + channels[..., 3])) / amp

    y = (channels[..., 1] + channels[..., 2] - \
        (channels[..., 0] + channels[..., 3])) / amp

    return amp, x, y


class PointEstimator:
    short_name = 'base'
    def __init__(self, channels, energies, positions):
        pass

    def __call__(self, *args, **kwargs):
        return self.get_value(*args, **kwargs)

    def get_value(self, channels, return_error=False):
        return

    def save_to(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)


class AngerSimple(PointEstimator):
    short_name = 'anger'
    def __init__(self, channels, energies, positions):
        amp, x, y = anger_basis(channels)

        self.a_x = 70 / np.max(np.abs(x))
        self.a_y = 70 / np.max(np.abs(y))

        self.a_e = np.mean(amp/energies)

    def get_value(self, channels, return_error=False):
        amp, x, y = anger_basis(channels)
        gain = 1/2

        d_amp = gain*np.sqrt(amp)
        d_s = d_amp/amp

        e = amp / self.a_e
        de = d_amp / self.a_e

        x = self.a_x * x
        dx = self.a_x*d_s
        y = self.a_y * y
        dy = self.a_y*d_s

        if return_error:
            return (e, x, y), (de, dx, dy)
        else:
            return (e, x, y)


class PointLookup(PointEstimator):
    short_name = 'base_lookup'
    def __init__(self, channels, energies, positions):
        self.channels = np.array(channels)
        self.energies = np.array(energies)
        self.positions = np.array(positions)

        assert self.channels.shape[:-1] == self.energies.shape == positions.shape[:-1]
        assert self.channels.shape[-1] == 4
        assert self.positions.shape[-1] == 2

        self.channels = self.channels.reshape((-1, 4))
        self.energies = self.energies.reshape((-1))
        self.positions = self.positions.reshape((-1,2))

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
        return np.full_like(ind, 10.), np.full_like(ind, 5.), np.full_like(ind, 5.)

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
                err_e = .01+np.sqrt(np.sum((self.energies[ind].T-e)**2*weights.T, axis=0))
                err_x = .1+np.sqrt(np.sum((self.positions[ind, 0].T-x)**2*weights.T, axis=0))
                err_y = .1+np.sqrt(np.sum((self.positions[ind, 1].T-y)**2*weights.T, axis=0))

            return (e, x, y), (err_e, err_x, err_y)
        else:
            return e, x, y


class LookupGradError(PointLookup):
    short_name = 'base_lookup_grad'
    def __init__(self, channels, energies, positions):
        # energy_points, x_points, y_points = energies.shape
        # TODO, check shape
        # v = np.reshape(channels, (energy_points, x_points, y_points, 4))

        g = np.gradient(channels, energies[:,0,0], positions[0,:,0,0], positions[0,0,:,1], axis=[0,1,2])
        g = np.divide(np.sqrt(channels/10), g)
        self.errors = np.zeros((g.shape[1]*g.shape[2]*g.shape[3], 3))
        for i in range(3):
            self.errors[:, i] = np.sqrt(np.sum(g[i]**2, axis=-1)).flatten() # energy

        super().__init__(channels, energies, positions)


class TreeLookup(LookupGradError):
    short_name = 'tree'
    def build(self):
        self.kdtree = KDTree(self.channels, leaf_size=32, metric='euclidean')

    def lookup_index(self, channels):
        return self.kdtree.query(channels, return_distance=False)[:, 0], None


class ProbLookup(LookupGradError):
    short_name = 'prob'
    def build(self):
        self.kdtree = KDTree(self.channels, leaf_size=32, metric='euclidean')

    def lookup_index(self, channels):
        gain2 = .04 # g squared
        ind = self.kdtree.query(channels, k=32, sort_results=False, return_distance=False)
        diff = np.empty((ind.shape[0], 4, ind.shape[1]))
        for i in range(ind.shape[-1]):
            diff[:,:,i] = (self.channels[ind[:, i]] - channels)**2 \
                / (gain2*self.channels[ind[:, i]])

        error = np.sum(diff, axis=1)

        return ind, np.exp(-error/2)


class AngerTreeLookup(LookupGradError):
    short_name = 'tree_anger'
    def build(self):
        self.kdtree = KDTree(anger_basis(self.channels),
                             leaf_size=32, metric='euclidean')

    def lookup_index(self, channels):
        return self.kdtree.query(anger_basis(channels),
                                 return_distance=False)[:, 0], None


class AngerProbLookup(LookupGradError):
    short_name = 'prob_anger'
    def build(self):
        channels = self.channels
        self.kdtree = KDTree(anger_basis(channels),
                             leaf_size=32, metric='euclidean')

    def lookup_index(self, channels):
        gain2 = .04 # g squared
        ind = self.kdtree.query(anger_basis(channels),
                                k=64, sort_results=False,
                                return_distance=False)
        diff = np.empty((ind.shape[0], 4, ind.shape[1]))
        for i in range(ind.shape[-1]):
            diff[:,:,i] = (self.channels[ind[:, i]] - channels)**2 \
                / (gain2*self.channels[ind[:, i]])

        error = np.sum(diff, axis=1)
        return ind, np.exp(-error/2)


def all_subclasses(c):
    a = []
    for subclass in c.__subclasses__():
        a.append(subclass)
        a.extend(all_subclasses(subclass))
    return a


methods = {x.short_name: x for x in all_subclasses(PointEstimator)}


class LookupUnpicler(pickle.Unpickler):
    def find_class(self, module, name):
        lookup_names = [x.__name__ for x in methods.values()]
        if module == __name__ and name in lookup_names:
            return getattr(nn_calibration.lookup, name)
        if module == 'builtins' and name in safe_builtins:
            return getattr(builtins, name)
        if module == 'numpy.core.multiarray' and name in ['_reconstruct', 'scalar']:
            return getattr(np.core.multiarray, name)
        if module == 'numpy' and name in ['ndarray', 'dtype']:
            return getattr(np, name)
        if module == 'sklearn.neighbors._kd_tree' and name in ['newObj', 'KDTree']:
            return getattr(sklearn.neighbors._kd_tree, name)
        if module == 'sklearn.neighbors._ball_tree' and name in ['newObj', 'BallTree']:
            return getattr(sklearn.neighbors._ball_tree, name)
        if module == 'sklearn.metrics._dist_metrics' and name in ['newObj', 'EuclideanDistance', 'CanberraDistance']:
            return getattr(sklearn.metrics._dist_metrics, name)
        raise pickle.UnpicklingError(f"global '{module}.{name}' is forbidden")


def load(path):
    with open(path, 'rb') as f:
        lookup = LookupUnpicler(f).load()
    return lookup
