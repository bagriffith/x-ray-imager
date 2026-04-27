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

"""Clustering algorithms used to collect events with the same energy/position.

Every algorithms here should be a child of the scikit-learn's `ClusterMixin`.
"""
from typing import Any, Optional
import logging
from sklearn.base import ClusterMixin
from sklearn.cluster import DBSCAN, OPTICS, KMeans
import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


class KMeansMin(ClusterMixin):
    """Mixin for clustering to ensure a min number of clusters with K-Means."""
    def __init__(self, **kwargs) -> None:
        self.labels_ = None

        if not hasattr(self, 'kmeans_kwargs'):
            self.kmeans_kwargs = None

        if not hasattr(self, 'min_clusters'):
            self.min_clusters = 0

        # Default K-Means args
        kmeans_kwargs = {'n_init': 32}

        if self.kmeans_kwargs is not None:
            kmeans_kwargs.update(self.kmeans_kwargs)

        self.kmeans_kwargs = kmeans_kwargs

        self.kmeans_cluster = KMeans(self.min_clusters, **self.kmeans_kwargs)
        super().__init__()

    def fit_predict(self,
                    X: ArrayLike,  # pylint: disable=invalid-name
                    y: Any = None):
        """Apply K-Means as backup."""
        super().fit_predict(X, y)

        if self.labels_ is None:
            raise RuntimeError('No labels created.')

        in_cluster = self.labels_ >= 0
        n_clusters = len(set(self.labels_[in_cluster]))
        logger.info('First clustering found %s clusters.', n_clusters)
        if n_clusters == 0:
            logger.warning('No clusters found.')
            # Open all points for kmeans
            self.labels_ = np.full_like(self.labels_, 0)
            in_cluster = np.full_like(in_cluster, True)
            n_clusters = 1

        # Split into enough clusters with KMeans
        if n_clusters < self.min_clusters:
            logger.info('Min of %s clusters requested. Applying K-means.',
                        self.min_clusters)
            norm = np.sum(X[in_cluster], axis=1).reshape(-1, 1)
            kmeans_fit = self.kmeans_cluster.fit(
                norm,
                y if y is None else y[in_cluster])
            self.labels_[in_cluster] = kmeans_fit.labels_
            n_clusters = self.min_clusters

        return self.labels_


class MinDBSCAN(KMeansMin, DBSCAN):
    def __init__(self,
                 min_clusters: int,
                 kmeans_kwargs: Optional[dict] = None,
                 **kwargs
                 ) -> None:
        """ """
        if kmeans_kwargs is None:
            kmeans_kwargs = dict()

        self.kmeans_kwargs = kmeans_kwargs
        self.min_clusters = min_clusters

        super().__init__(**kwargs)


class MinOPTICS(KMeansMin):
    def __init__(self,
                 min_clusters: int,
                 kmeans_kwargs: Optional[dict] = None,
                 **kwargs
                 ) -> None:
        """ """
        if kmeans_kwargs is None:
            kmeans_kwargs = dict()

        self.kmeans_kwargs = kmeans_kwargs
        self.min_clusters = min_clusters
        super().__init__()
        self.optics = OPTICS(**kwargs)

    def fit(self, X, y=None):
        self.optics.fit(X, y)
        self.labels_ = self.optics.labels_
