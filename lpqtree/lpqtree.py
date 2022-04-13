"""Sklearn interface to the native nanoflann module"""
import copyreg
import warnings
from typing import Optional

import nanoflann_ext
import numpy as np
from sklearn.neighbors._base import KNeighborsMixin, NeighborsBase, RadiusNeighborsMixin
from sklearn.utils.validation import check_is_fitted
from scipy.sparse import csr_matrix, coo_matrix

SUPPORTED_TYPES = [np.float32, np.float64]
SUPPORTED_DIM = [2, 3]
# SUPPORTED_METRIC = ["lpq"] where p and q are any digit


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
        raise ValueError(f"Supported types: {points.dtype} not in {SUPPORTED_TYPES}")
    if len(points.shape) not in SUPPORTED_DIM:
        raise ValueError(f"Incorrect shape {len(points.shape)} not in {SUPPORTED_DIM}")


class KDTree(NeighborsBase, KNeighborsMixin, RadiusNeighborsMixin):
    def __init__(self, n_neighbors=5, radius=1.0, leaf_size=10, metric="l2"):
        """
        Lpq KDTree initialisation, where the "metric"  should to be set,
        the "n_neighbors" and "radius" can be changed afterward.

        Parameters
        ----------
        n_neighbors : int
            Default number of nearest neighbors for KDTree.query()
        radius : float
            Default radius for the KDTree.radius_neighbors()
        leaf_size : int
            Tree leaf size, values between 5 and 100 should work fine
        metric : str
            Lp or Lpq metric, where "p" and "q" are integer > 0,
            Lpq only work with list of matrices (list of 2D array).
        """

        metric = metric.lower()
        if len(metric) < 2 or not metric[1:].isnumeric():
            raise ValueError(f"Metric should start with 'l' followed with 1 or 2 numerical value")

        super().__init__(
            n_neighbors=n_neighbors, radius=radius, leaf_size=leaf_size, metric=metric
        )

        self.index = None
        self._fit_X = None
        self._nb_vts_in_tree = None
        self._nb_vts_in_search = None

    def fit(self, X: np.ndarray, index_path: Optional[str] = None):
        """
        Create the Lpq KDTree with the given list of vertices / matrices (X[i]).

        Parameters
        ----------
        X : np.ndarray
            List of points (for Lp) or List of matrices (for Lpq).
        index_path : str
            str Path to a previously built index.
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

        if self.metric == "l2" or self.metric == "l1":
            last_dim = 1
        else:
            if X.ndim == 3:
                last_dim = X.shape[2]
            else:
                raise ValueError(f"{self.metric} metric should be used with 3dim array")

        self._fit_X = X.reshape((X.shape[0], -1))
        self._nb_vts_in_tree = self._fit_X.shape[0]
        self.index.fit(self._fit_X, index_path if index_path is not None else "", last_dim)

    def get_data(self, copy: bool = True) -> np.ndarray:
        """
        Return the data inside the tree

        Parameters
        ----------
        copy : bool
            Return a copy of the data, if set to True

        Returns
        -------
        X : np.ndarray
            List of points (for Lp) or List of matrices (for Lpq).
        """
        check_is_fitted(self, ["_fit_X"], all_or_any=any)

        if copy:
            return self._fit_X.copy()
        else:
            return self._fit_X

    def save_index(self, path: str) -> int:
        """Save index to the binary file. NOTE: Data points are NOT stored."""
        return self.index.save_index(path)

    def query(self, X, k=1, return_distance=True, n_jobs=1):
        """
        Compute k-nearest neighbors (knn) in the KDTree, for each X[i],
        and return a numpy of reference indices and distances.

        Parameters
        ----------
        X : np.ndarray
            List of points (for Lp) or List of matrices (for Lpq).
        k : int
            Number of nearest neighbors wanted per slines
        return_distance : bool
            Compute and return the distance
        n_jobs : integer
            Number of processor cores (multithreading)

        Returns
        -------
        knn_res : numpy array (len(X) x k)
            Reference indices of the k-nearest neighbors of each X[i]
        dists : numpy array (len(X) x k)
            Distances for all k-nearest neighbors
        """
        check_is_fitted(self, ["_fit_X"], all_or_any=any)
        _check_arg(X)

        if X.ndim == 3:
            X = X.reshape((X.shape[0], -1))

        if k is None:
            k = self.n_neighbors
        else:
            self.n_neighbors = k

        if k > len(self._fit_X):
            raise ValueError(f"KD Tree query bigger for {k}-NN however the "
                             f"KD Tree only contain {len(self._fit_X)} points")

        if n_jobs == 1:
            self.index.kneighbors(X, k)
        else:
            self.index.kneighbors_multithreaded(X, k, n_jobs)

        knn_res = self.index.getResultIndicesCol().reshape((-1, k))
        if return_distance:
            dists = self.index.getResultDists().reshape((-1, k))
            return knn_res, dists

        return knn_res

    def radius_neighbors(self, X, radius=None, return_distance=True, n_jobs=1, no_return=False):
        """
        Compute radius search for each streamlines in X searching into the KDTree,
        and return a list of indices containing the neighborhood information.

        Parameters
        ----------
        X : np.ndarray
            List of points (for Lp) or List of matrices (for Lpq).
        radius : float
            Searching Radius
        return_distance : bool
            Compute and return the distance
        n_jobs : integer
            Number of processor cores (multithreading)
        no_return : bool
            Avoid directly returning the result, to avoid memory overhead,
            KDTree get_rows(), get_cols() and get_dists() can be use afterward


        Returns
        -------
        ids_x : numpy array
            Indices of the given X
        ids_tree : numpy array
            Indices of the KDTree data
        dists : numpy array (len(X) x k)
            Distances for each match
        """
        check_is_fitted(self, ["_fit_X"], all_or_any=any)
        _check_arg(X)

        if X.ndim == 3:
            X = X.reshape((X.shape[0], -1))

        if radius is None:
            radius = self.radius
        else:
            self.radius = radius

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

        self._nb_vts_in_search = X.shape[0]

        if no_return:
            return

        ids_x = self.index.getResultIndicesRow()
        ids_tree = self.index.getResultIndicesCol()
        if return_distance:
            dists = self.index.getResultDists()
            return ids_x, ids_tree, dists

        return ids_x, ids_tree

    def radius_neighbors_full(self, X_mpts, Data_full, X_full, radius, n_jobs=1):
        """
        Advanced radius_neighbors search using 2 sampling for each data point.
        Compute radius search for each streamlines in X searching into the KDTree,
        and return a list of indices containing the neighborhood information.

        KDTree should have fitted the "Data_mpts"


        Parameters
        ----------
        X_mpts : np.ndarray
            List of mean-points, same dimension as the KDTree.fit() data
        Data_full : np.ndarray
            List of non averaged points, inside KDTree.fit()
        X_full : np.ndarray
            List of non averaged points, same dimension as "Data_full"
        radius : float
            Searching Radius
        n_jobs : integer
            Number of processor cores (multithreading)
        """
        if X_mpts.ndim == 3:
            X_mpts = X_mpts.reshape((X_mpts.shape[0], -1))
        if Data_full.ndim == 3:
            Data_full = Data_full.reshape((Data_full.shape[0], -1))
        if X_full.ndim == 3:
            X_full = X_full.reshape((X_full.shape[0], -1))

        nb_mpts = X_mpts.shape[1]
        nb_dim = X_full.shape[1]

        assert(X_mpts.shape[1] <= X_full.shape[1])
        assert(X_full.shape[1] == Data_full.shape[1])
        assert(X_mpts.shape[0] == X_full.shape[0])
        assert(self.get_data(copy=False).shape[0] == Data_full.shape[0])
        assert(nb_dim % nb_mpts == 0)

        pnorm = float(self.metric[-1])
        mpts_radius = radius * (nb_mpts / nb_dim)**(1.0/pnorm)

        self._nb_vts_in_search = X_mpts.shape[0]
        if n_jobs == 1:
            self.index.radius_neighbors_idx_dists_full(X_mpts, Data_full, X_full, mpts_radius, radius)
        else:
            self.index.radius_neighbors_idx_dists_full_multithreaded(X_mpts, Data_full, X_full, mpts_radius, radius, n_jobs)

    def fit_and_radius_search(self, tree_vts, search_vts, radius, n_jobs=1, nb_mpts=None):
        """
        Advanced radius_neighbors search using 2 sampling for each data point.
        Compute radius search for each streamlines in X searching into the KDTree,
        and return a list of indices containing the neighborhood information.

        Parameters
        ----------
        tree_vts : np.ndarray
            List of points / matrices, to fit the KDTree
        search_vts : np.ndarray
            List of points / matrices, for the search
        radius : float
            Searching Radius
        n_jobs : integer
            Number of processor cores (multithreading)
        nb_mpts : integer
            Number of mean-points used for the advanced search
        """
        assert(np.alltrue(tree_vts.shape[1:] == search_vts.shape[1:]))

        if nb_mpts and nb_mpts < tree_vts.shape[1]:
            if not(self.metric in ["l1", "l2", "l11", "l21"]):
                raise ValueError(f"Only  l1, l2, l11, or l21  can be used with nb_mpts")

            if tree_vts.shape[1] % nb_mpts != 0:
                raise ValueError(f"nb_mpts must be a divisor of tree_vts.shape[2]")

            nb_averaged = tree_vts.shape[1] // nb_mpts
            tree_mpts = np.mean(tree_vts.reshape((tree_vts.shape[0], nb_mpts, nb_averaged, -1)), axis=2)
            search_mpts = np.mean(search_vts.reshape((search_vts.shape[0], nb_mpts, nb_averaged, -1)), axis=2)

            self.fit(tree_mpts)
            self.radius_neighbors_full(search_mpts, tree_vts, search_vts, radius, n_jobs=n_jobs)

        else:
            self.fit(tree_vts)
            self.radius_neighbors(search_vts, radius=radius, n_jobs=n_jobs,
                                  return_distance=True, no_return=True)

    def get_dists(self):
        """Return the stored distances after a search"""
        return self.index.getResultDists()

    def get_rows(self):
        """Return the stored given query points indices"""
        return self.index.getResultIndicesRow()

    def get_cols(self):
        """Return the stored fitted KDTree points (search) indices"""
        return self.index.getResultIndicesCol()

    def get_csr_matrix(self):
        """Return the stored search results indices as a sparse csr_matrix"""
        mtx_shape = None
        if self._nb_vts_in_search and self._nb_vts_in_tree:
            mtx_shape = (self._nb_vts_in_search, self._nb_vts_in_tree)
        return csr_matrix((self.get_dists(), self.get_cols(), self.index.getResultIndicesPtr()), shape=mtx_shape)

    def get_coo_matrix(self):
        """Return the stored search results indices as a sparse coo_matrix"""
        mtx_shape = None
        if self._nb_vts_in_search and self._nb_vts_in_tree:
            mtx_shape = (self._nb_vts_in_search, self._nb_vts_in_tree)
        return coo_matrix((self.get_dists(), (self.get_rows(), self.get_cols())), shape=mtx_shape)

    def get_csc_matrix(self):
        """Return the stored search results indices as a sparse csc_matrix"""
        return self.get_coo_matrix().to_csc()


# Register pickling of non-trivial types
copyreg.pickle(KDTree, pickler, unpickler)
