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

"""Clustering algorithms to find events with the same energy/position.

DBSCAN and OPTICS return a variable number of clusters, meaning lines with
similar energies merge into the same group. To accommodate this, a minimum
number of clusters can be specified. K-Means is then applied to separate
the lines.

Algorithms here should be a child of the scikit-learn's `ClusterMixin`.
"""
from typing import Any, Optional
import logging
from sklearn.cluster import DBSCAN, OPTICS, KMeans
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class KMeansMinMixin:
    """Mixin to guarantee a min number of clusters.

    If too few clusters are found, K-means splits all events, other than noise,
    into the min number of groups.
    """
    def __init__(self, min_clusters: int, **kwargs) -> None:
        """Initialize the k-means clustering object.
        
        Args:
            min_clusters: Minimum number of clusters returned. Sets the
                threshold to apply k-means.
            **kwargs: Passed to the KMeans constructor.
                See sklearn.cluster.KMeans for more details.
        """
        self.min_clusters = min_clusters
        self.kmeans_cluster = KMeans(min_clusters, **kwargs)

    def fit_min(self,
                X: NDArray[Any],  # pylint: disable=invalid-name
                y: Any = None):
        """Check if the min number of clusters is met, and apply K-means if not.
        
        This should be called in the subclass after that clustering is
        completed. For example:
            super().fit(X, y, **kwargs)
            self.fit_min(X, y, **kwargs)
        
        Args:
            X: The data to cluster. Shape should be (n_samples, n_features).
            y: Passed to the clustering fit(), but not used. Kept for
                consistency with Scikit-learn's clustering algorithms.
        """
        in_cluster = self.labels_ >= 0
        n_clusters = len(set(self.labels_[in_cluster]))
        logger.info('First clustering found %s clusters.', n_clusters)
        if n_clusters == 0:
            logger.warning('No clusters found.')
            # Open all points for kmeans
            self.labels_ = np.full_like(self.labels_, 0)
            in_cluster = np.full_like(in_cluster, True)
            n_clusters = 1

        if self.kmeans_cluster is None:
            logger.warning('No kmeans cluster initialized.')
            return

        # Split into enough clusters with KMeans
        if n_clusters < self.min_clusters:
            logger.info('Min of %s clusters requested. Applying K-means.',
                        self.min_clusters)
            norm = np.sum(X[in_cluster], axis=1).reshape(-1, 1)
            kmeans_fit = self.kmeans_cluster.fit(
                norm,
                y if y is None else y[in_cluster])
            self.labels_[in_cluster] = kmeans_fit.labels_


class MinDBSCAN(KMeansMinMixin, DBSCAN):
    """DBSCAN with a minimum number of clusters."""
    def __init__(self,
                 min_clusters: int,
                 kmeans_kwargs: Optional[dict] = None,
                 **kwargs
                 ) -> None:
        """Initialize DBSCAN and set min clusters returned.
        
        Args:
            min_clusters: Minimum number of clusters returned, passed to
                the KMeansMinMixin.
            kmeans_kwargs: Keyword arguments passed to KMeansMinMixin to
                the sklearn.cluster.KMeans constructor.
            **kwargs: Passed to the DBSCAN constructor.
                See sklearn.cluster.DBSCAN for more details.
        """
        if kmeans_kwargs is None:
            kmeans_kwargs = dict()

        self.kmeans_kwargs = kmeans_kwargs  # Needed for numpy
        super().__init__(min_clusters, **kmeans_kwargs)
        super(KMeansMinMixin, self).__init__(**kwargs)

    def fit(self,
            X: NDArray[Any],  # pylint: disable=invalid-name
            y: Any = None,
            sample_weight: Optional[NDArray[Any]] = None
            ) -> 'MinDBSCAN':
        """Fit DBSCAN and check if the min number of clusters is met.
        
        Args:
            X: The data to cluster. Shape should be (n_samples, n_features).
            y: Passed to the clustering fit(), but not used. Kept for
                consistency with Scikit-learn's clustering algorithms.
            sample_weight: Passed to the clustering fit(), but not used. Kept
                for consistency with Scikit-learn's clustering algorithms.
        """
        super().fit(X, y, sample_weight)
        self.fit_min(X, y)
        return self


class MinOPTICS(KMeansMinMixin, OPTICS):
    """OPTICS with a minimum number of clusters."""
    def __init__(self,
                 min_clusters: int,
                 kmeans_kwargs: Optional[dict] = None,
                 **kwargs
                 ) -> None:
        """Initialize OPTICS and set min clusters returned.
        
        Args:
            min_clusters: Minimum number of clusters returned, passed to
                the KMeansMinMixin.
            kmeans_kwargs: Keyword arguments passed to KMeansMinMixin to
                the sklearn.cluster.KMeans constructor.
            **kwargs: Passed to the OPTICS constructor.
                See sklearn.cluster.OPTICS for more details.
        """
        if kmeans_kwargs is None:
            kmeans_kwargs = dict()

        self.kmeans_kwargs = kmeans_kwargs  # Needed for numpy
        super().__init__(min_clusters, **kmeans_kwargs)
        super(KMeansMinMixin, self).__init__(**kwargs)

    def fit(self,
            X: NDArray[Any],  # pylint: disable=invalid-name
            y: Any = None,
            **kwargs):
        """Fit OPTICS and check if the min number of clusters is met.
        
        Args:
            X: The data to cluster. Shape should be (n_samples, n_features).
            y: Passed to the clustering fit(), but not used. Kept for
                consistency with Scikit-learn's clustering algorithms.
            sample_weight: Passed to the clustering fit(), but not used. Kept
                for consistency with Scikit-learn's clustering algorithms.
        """
        super().fit(X, y, **kwargs)
        self.fit_min(X, y)
        return self
