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
from sklearn.cluster import DBSCAN, KMeans
import numpy as np
from numpy.typing import ArrayLike


class DBSCANFallbackKMeans(ClusterMixin):
    """Clustering with DBSCAN, and if too few K-Means on the L1 norm."""
    def __init__(self,
                 min_clusters: int,
                 dbscan_kwargs: Optional[dict] = None,
                 kmeans_kwargs: Optional[dict] = None
                 ) -> None:
        """Initialize the clustering instance.

        Args:
            min_clusters: Minimum clusters DBSCAN must find, otherwise K-Means
                is invoked.
            dbscan_kwargs: Passed as kwargs to the scikit-learn's `DBSCAN`.
            kmeans_kwargs: Passed as kwargs to the scikit-learn's `KMeans`.
        """
        self.min_clusters = min_clusters

        if dbscan_kwargs is None:
            dbscan_kwargs = {'eps': 5.0,
                             'min_samples': 1000}

        if kmeans_kwargs is None:
            kmeans_kwargs = {'n_init': 32}

        self.dbscan_cluster = DBSCAN(**dbscan_kwargs)
        self.kmeans_cluster = KMeans(self.min_clusters, **kmeans_kwargs)

        self.labels_ = None  # To be filled in by `fit()`.

        super().__init__()

    def fit(self,
            X: ArrayLike,  # pylint: disable=invalid-name
            y: Any = None,
            sample_weight: ArrayLike | None = None):
        """Preform DBSCAN then if too few clusters, K-Means on L1 norm.

        Args:
            X: Array of points to cluster.
            y: Not used.
        """
        X = np.array(X, dtype=np.float64)
        self.labels_ = self.dbscan_cluster.fit(X, y, sample_weight).labels_

        # Only positive labels are clusters.
        # -1 marks noisy (background) points.
        in_cluster = self.labels_ >= 0
        n_clusters = len(set(self.labels_[in_cluster]))
        logging.info('DBSCAN found %s clusters.', n_clusters)
        if n_clusters == 0:
            logging.warning('No clusters found.')
            # Open all points for kmeans
            self.labels_ = np.full_like(self.labels_, 0)
            in_cluster = np.full_like(in_cluster, True)
            n_clusters = 1

        # Split into enough clusters with KMeans
        if n_clusters < self.min_clusters:
            logging.info('Min of %s clusters requested. Applying K-means.',
                         self.min_clusters)
            norm = np.sum(X[in_cluster], axis=1).reshape(-1, 1)
            kmeans_fit = self.kmeans_cluster.fit(
                norm,
                y if y is None else y[in_cluster],
                sample_weight
                if sample_weight is None else sample_weight[in_cluster])
            self.labels_[in_cluster] = kmeans_fit.labels_
            n_clusters = self.min_clusters

        return self
