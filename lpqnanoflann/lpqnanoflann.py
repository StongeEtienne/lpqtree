"Sklearn interface to the native nanoflann module"
import copyreg
import warnings
from typing import Optional

import nanoflann_ext
import numpy as np
from sklearn.neighbors._base import KNeighborsMixin, NeighborsBase, RadiusNeighborsMixin
from sklearn.utils.validation import check_is_fitted
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix

SUPPORTED_TYPES = [np.float32, np.float64]


def pickler(c):
    X = c._fit_X if hasattr(c, "_fit_X") else None
    return unpickler, (c.n_neighbors, c.radius, c.leaf_size, c.metric, X)


def unpickler(n_neighbors, radius, leaf_size, metric, X):
    # Recreate an kd-tree instance
    tree = KDTree(n_neighbors, radius, leaf_size, metric)
    # Unpickling of the fitted instance
    if X is not None:
        tree.fit(X)
    return tree


def _check_arg(points):
    if points.dtype not in SUPPORTED_TYPES:
        raise ValueError("Supported types: [{}]".format(SUPPORTED_TYPES))
    if len(points.shape) != 2:
        raise ValueError(f"Incorrect shape {len(points.shape)} != 2")


class KDTree(NeighborsBase, KNeighborsMixin, RadiusNeighborsMixin):
    def __init__(self, n_neighbors=5, radius=1.0, leaf_size=10, metric="l2"):

        metric = metric.lower()
        if metric not in ["l1", "l2", "l21"]:
            raise ValueError('Supported metrics: ["l1", "l2", "l21"]')

        if metric == "l2":  # nanoflann uses squared distances
            radius = radius ** 2

        super().__init__(
            n_neighbors=n_neighbors, radius=radius, leaf_size=leaf_size, metric=metric
        )

    def fit(self, X: np.ndarray, index_path: Optional[str] = None):
        """
        Args:
            X: np.ndarray data to use
            index_path: str Path to a previously built index. Allows you to not rebuild index.
                NOTE: Must use the same data on which the index was built.
        """
        _check_arg(X)
        if X.dtype == np.float32:
            self.index = nanoflann_ext.KDTree32(
                self.n_neighbors, self.leaf_size, self.metric, self.radius
            )
        else:
            self.index = nanoflann_ext.KDTree64(
                self.n_neighbors, self.leaf_size, self.metric, self.radius
            )

        if X.shape[1] > 64:
            warnings.warn(
                "KD Tree structure is not a good choice for high dimensional spaces."
                "Consider a more suitable search structure."
            )

        self.index.fit(X, index_path if index_path is not None else "")
        self._fit_X = X

    def get_data(self, copy: bool = True) -> np.ndarray:
        """Returns underlying data points. If copy is `False` then no modifications should be applied to the returned data.

        Args:
            copy: whether to make a copy.
        """
        check_is_fitted(self, ["_fit_X"], all_or_any=any)

        if copy:
            return self._fit_X.copy()
        else:
            return self._fit_X

    def save_index(self, path: str) -> int:
        "Save index to the binary file. NOTE: Data points are NOT stored."
        return self.index.save_index(path)

    def radius_neighbors(self, X, radius=None, return_distance=True, n_jobs=1, return_array=True):
        check_is_fitted(self, ["_fit_X"], all_or_any=any)
        _check_arg(X)

        if radius is None:
            radius = self.radius
        elif self.metric == "l2":
            radius = radius ** 2  # L2 nanoflann internally uses squared distances

        if n_jobs == 1:
            if return_distance:
                self.index.radius_neighbors_idx_dists(X, radius)
            else:
                self.index.radius_neighbors_idx(X, radius)
        else:
            if return_distance:
                self.index.radius_neighbors_idx_dists_multithreaded(X, radius, n_jobs)
            else:
                self.index.radius_neighbors_idx_multithreaded(X, radius, n_jobs)

        if not return_array:
            return

        if return_distance:
            if self.metric == "l2":
                return self.index.getResultIndicesRow(), self.index.getResultIndicesCol(), np.sqrt(self.index.getResultDists())
            else:
                return self.index.getResultIndicesRow(), self.index.getResultIndicesCol(), self.index.getResultDists()

        return self.index.getResultIndicesRow(), self.index.getResultIndicesCol()

    # Results getter with sparse matrices
    def get_dists(self):
        if self.metric == "l2":
            return np.sqrt(self.index.getResultDists())
        else:
            return self.index.getResultDists()

    def get_rows(self):
        return self.index.getResultIndicesRow()

    def get_cols(self):
        return self.index.getResultIndicesCol()

    def get_csr_matrix(self):
        return csr_matrix((self.get_dists(), self.get_cols(), self.index.getResultIndicesPtr()))

    def get_coo_matrix(self):
        return coo_matrix((self.get_dists(), (self.get_rows(), self.get_cols())))

    def get_csc_matrix(self):
        return self.get_coo_matrix().to_csc()

    # Advanced operation
    def radius_neighbors_full(self, X_mpts, Data_full, X_full, radius_full, n_jobs=1):
        nb_mpts = X_mpts.shape[1]
        nb_dim = X_full.shape[1]

        assert(X_mpts.shape[1] <= X_full.shape[1])

        assert(X_full.shape[1] == Data_full.shape[1])
        assert(X_mpts.shape[0] == X_full.shape[0])
        assert(self.get_data(copy=False).shape[0] == Data_full.shape[0])
        assert(nb_dim % nb_mpts == 0)

        if self.metric == "l2":
            radius_full = radius_full ** 2  # L2 nanoflann internally uses squared distances

        mpts_dist = radius_full * nb_mpts / nb_dim

        if n_jobs == 1:
            self.index.radius_neighbors_idx_dists_full(X_mpts, Data_full, X_full, mpts_dist, radius_full)
        else:
            self.index.radius_neighbors_idx_dists_full_multithreaded(X_mpts, Data_full, X_full, mpts_dist, radius_full, n_jobs)


# Register pickling of non-trivial types
copyreg.pickle(KDTree, pickler, unpickler)
