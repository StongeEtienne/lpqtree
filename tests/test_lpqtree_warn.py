import string

import pytest

import numpy as np
import lpqtree
import warnings

# testing values
NB_MTX = 15  # Number of element for the tree
DIM1 = 4  # Number of point per element
DIM2 = 3  # Number of dimension per point
DIM3 = 11  # Number of dimension per point

LARGE_DIM = 77

MAXDIST = 28.2
LEAF_SIZE = 1  # force to test kd-tree split distance


def test_get_data():
    vts = np.random.rand(NB_MTX, DIM1, DIM2).astype(np.float32)
    lpq_tree = lpqtree.KDTree(metric="l22", radius=MAXDIST, leaf_size=LEAF_SIZE)
    lpq_tree.fit(vts)

    # Test get_data reference
    data = lpq_tree.get_data(copy=False)
    assert data.base is vts

    # Test get_data copy
    data_cp = lpq_tree.get_data(copy=True)
    assert not (data_cp.base is vts)
    assert np.allclose(data_cp.flatten(), vts.flatten())


def test_lpqtree_default():
    vts1 = np.random.rand(NB_MTX, DIM1, DIM2).astype(np.float32)
    tree1 = lpqtree.KDTree(leaf_size=LEAF_SIZE)
    tree2 = lpqtree.KDTree(n_neighbors=1, radius=1.0, metric="l2", leaf_size=LEAF_SIZE)
    tree1.fit(vts1)
    tree2.fit(vts1)

    vts2 = np.random.rand(NB_MTX, DIM1, DIM2).astype(np.float32)
    # Test knn (default k=1)
    assert np.allclose(tree1.query(vts2, k=None), tree2.query(vts2, k=None))
    assert np.allclose(tree1.query(vts2, k=None), tree2.query(vts2, k=1))

    # Test rknn (default k=1, radius=1)
    assert np.allclose(tree1.radius_knn(vts2, k=None), tree2.radius_knn(vts2, k=None))
    assert np.allclose(tree1.radius_knn(vts2, k=None, radius=1.0, return_distance=False),
                       tree2.radius_knn(vts2, k=1, radius=None, return_distance=False))

    # Test radius search (default k=1, radius=1)
    assert np.allclose(tree1.radius_neighbors(vts2, radius=None), tree2.radius_neighbors(vts2))


def test_data_type_exceptions():
    vts = np.random.rand(NB_MTX, DIM1, DIM2).astype(np.float32)

    for p in range(1, 10):
        for q in range(1, 10):
            tree_metric = "l" + str(p) + str(q)
            lpq_tree = lpqtree.KDTree(metric=tree_metric, radius=MAXDIST, leaf_size=LEAF_SIZE)

            # Test fit data_type
            for i_dtype in [int, bool, str]:
                with pytest.raises(ValueError):
                    lpq_tree.fit(vts.astype(i_dtype))

            # Test queries data_type
            lpq_tree.fit(vts)
            for i_dtype in [int, bool, str]:
                with pytest.raises(ValueError):
                    lpq_tree.query(vts.astype(i_dtype))
                with pytest.raises(ValueError):
                    lpq_tree.radius_knn(vts.astype(i_dtype), 1)
                with pytest.raises(ValueError):
                    lpq_tree.radius_neighbors(vts.astype(i_dtype))


def test_metric_exceptions():
    # Test with empty metric
    with pytest.raises(ValueError):
        lpqtree.KDTree(metric='', radius=MAXDIST, leaf_size=LEAF_SIZE)

    # Test 4 characters
    with pytest.raises(ValueError):
        lpqtree.KDTree(metric='l222', radius=MAXDIST, leaf_size=LEAF_SIZE)

    # Test with a single character
    for s in string.printable:
        with pytest.raises(ValueError):
            lpqtree.KDTree(metric=s, radius=MAXDIST, leaf_size=LEAF_SIZE)

        if s == 'l' or s == 'L':
            continue

        # Test with 2 invalid characters
        for p in string.printable:
            with pytest.raises(ValueError):
                lpqtree.KDTree(metric=s+p, radius=MAXDIST, leaf_size=LEAF_SIZE)

            # Test with 3 invalid characters
            for q in string.printable:
                with pytest.raises(ValueError):
                    lpqtree.KDTree(metric=s + p + q, radius=MAXDIST, leaf_size=LEAF_SIZE)

    # Test with a non-numerical character (and a valid one = 2)
    for p in string.ascii_letters:
        with pytest.raises(ValueError):
            lpqtree.KDTree(metric='l2' + p, radius=MAXDIST, leaf_size=LEAF_SIZE)
        with pytest.raises(ValueError):
            lpqtree.KDTree(metric='l' + p + '2', radius=MAXDIST, leaf_size=LEAF_SIZE)


def test_metric_no_exceptions():
    for s in ['L', 'l']:
        for p in range(1, 10):
            tree_metric = s + str(p)
            lpqtree.KDTree(metric=tree_metric, radius=MAXDIST, leaf_size=LEAF_SIZE)


def test_knn_exceptions():
    # Test if k in knn larger than the number of elements to search for
    vts = np.random.rand(NB_MTX, DIM1, DIM2).astype(np.float32)
    for p in range(1, 10):
        for q in range(1, 10):
            tree_metric = "l" + str(p) + str(q)
            lpq_tree = lpqtree.KDTree(metric=tree_metric, radius=MAXDIST, leaf_size=LEAF_SIZE)
            lpq_tree.fit(vts)
            with pytest.raises(ValueError):
                lpq_tree.query(vts, k=NB_MTX + 1)
            with pytest.raises(ValueError):
                lpq_tree.radius_knn(vts, k=NB_MTX + 1)


def test_data_large_dim_warning():
    vts2 = np.random.rand(NB_MTX, LARGE_DIM).astype(np.float32)
    for p in [1, 2]:
        tree_metric = "l" + str(p)
        lpq_tree = lpqtree.KDTree(metric=tree_metric, radius=MAXDIST, leaf_size=LEAF_SIZE)

        with pytest.warns(UserWarning):
            lpq_tree.fit(vts2)
        lpq_tree.radius_neighbors(vts2)

    vts3 = np.random.rand(NB_MTX, LARGE_DIM, DIM2).astype(np.float32)
    for p in range(1, 10):
        tree_metric = "l" + str(p)
        lpq_tree = lpqtree.KDTree(metric=tree_metric, radius=MAXDIST, leaf_size=LEAF_SIZE)

        with pytest.warns(UserWarning):
            lpq_tree.fit(vts3)
        lpq_tree.radius_neighbors(vts3)


def test_data_shape_no_exceptions():
    vts2 = np.random.rand(NB_MTX, DIM1).astype(np.float32)
    for p in [1, 2]:
        tree_metric = "l" + str(p)
        lpq_tree = lpqtree.KDTree(metric=tree_metric, radius=MAXDIST, leaf_size=LEAF_SIZE)
        lpq_tree.fit(vts2)

    vts3 = np.random.rand(NB_MTX, DIM1, DIM2).astype(np.float32)
    for p in range(1, 10):
        tree_metric = "l" + str(p)
        lpq_tree = lpqtree.KDTree(metric=tree_metric, radius=MAXDIST, leaf_size=LEAF_SIZE)
        lpq_tree.fit(vts3)


def test_lp_data_shape_exceptions():
    vts_2dim = np.random.rand(NB_MTX, DIM1).astype(np.float32)
    for p in range(3, 10):
        tree_metric = "l" + str(p)
        lpq_tree = lpqtree.KDTree(metric=tree_metric, radius=MAXDIST, leaf_size=LEAF_SIZE)
        with pytest.raises(ValueError):
            lpq_tree.fit(vts_2dim)

    vts_3dim = np.random.rand(NB_MTX, DIM1, DIM2, DIM3).astype(np.float32)
    for p in range(1, 10):
        tree_metric = "l" + str(p)
        lpq_tree = lpqtree.KDTree(metric=tree_metric, radius=MAXDIST, leaf_size=LEAF_SIZE)
        with pytest.raises(ValueError):
            lpq_tree.fit(vts_3dim)


def test_lpq_data_shape_exceptions():
    vts_2dim = np.random.rand(NB_MTX, DIM1).astype(np.float32)
    vts_4dim = np.random.rand(NB_MTX, DIM1, DIM2, DIM3).astype(np.float32)
    for p in range(1, 10):
        for q in range(1, 10):
            tree_metric = "l" + str(p) + str(q)
            lpq_tree = lpqtree.KDTree(metric=tree_metric, radius=MAXDIST, leaf_size=LEAF_SIZE)

            with pytest.raises(ValueError):
                lpq_tree.fit(vts_2dim)

            with pytest.raises(ValueError):
                lpq_tree.fit(vts_4dim)


def test_fit_and_radius_search_no_exceptions():

    # Test valid number of mean points
    vts1 = np.random.rand(NB_MTX, 12, DIM2).astype(np.float32)
    vts2 = np.random.rand(NB_MTX, 12, DIM2).astype(np.float32)
    for m in ["l1", "l2", "l11", "l21"]:
        lpq_tree = lpqtree.KDTree(metric=m, radius=MAXDIST, leaf_size=LEAF_SIZE)
        for nb_mpt in [None, 1, 2, 4, 6]:
            lpq_tree.fit_and_radius_search(vts1, vts2, radius=MAXDIST, nb_mpts=nb_mpt)

    # Test valid number of mean points for a prime number
    vts1 = np.random.rand(NB_MTX, 7, DIM2).astype(np.float32)
    vts2 = np.random.rand(NB_MTX, 7, DIM2).astype(np.float32)
    for m in ["l1", "l2", "l11", "l21"]:
        lpq_tree = lpqtree.KDTree(metric=m, radius=MAXDIST, leaf_size=LEAF_SIZE)
        for nb_mpt in [None, 1]:
            lpq_tree.fit_and_radius_search(vts1, vts2, radius=MAXDIST, nb_mpts=nb_mpt)


def test_fit_and_radius_search_exceptions():
    vts1 = np.random.rand(NB_MTX, 12, DIM2).astype(np.float32)
    vts2 = np.random.rand(NB_MTX, 12, DIM2).astype(np.float32)

    # Test invalid metric (not in ["l1", "l2", "l11", "l21"])
    for p in range(2, 10):
        for q in range(2, 10):
            tree_metric = "l" + str(p) + str(q)
            lpq_tree = lpqtree.KDTree(metric=tree_metric, radius=MAXDIST, leaf_size=LEAF_SIZE)
            for nb_mpt in [1, 2, 4, 6]:
                with pytest.raises(ValueError):
                    lpq_tree.fit_and_radius_search(vts1, vts2, radius=MAXDIST, nb_mpts=nb_mpt)

    # Test invalid number of mean points for a prime number
    vts1 = np.random.rand(NB_MTX, 7, DIM2).astype(np.float32)
    vts2 = np.random.rand(NB_MTX, 7, DIM2).astype(np.float32)
    for m in ["l1", "l2", "l11", "l21"]:
        lpq_tree = lpqtree.KDTree(metric=m, radius=MAXDIST, leaf_size=LEAF_SIZE)
        for nb_mpt in range(2, 15):
            with pytest.raises(ValueError):
                lpq_tree.fit_and_radius_search(vts1, vts2, radius=MAXDIST, nb_mpts=nb_mpt)
