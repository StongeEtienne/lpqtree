import pytest

import numpy as np
import lpqtree
import lpqtree.lpqpydist as lpqdist

EPS = 1.0e-5
MAXDIST = 9.9e32

# testing values
NB_MTX = 100
NB_RADIUS = 10
np.random.seed(4)


def kdtree_test(v1, v2, p, q, tree_m):
    lpq_res = lpqdist.lpq_allpairs(v1, v2, p=p, q=q)
    min_v = lpq_res.min() + EPS
    max_v = lpq_res.max() + EPS
    step = (max_v - min_v) / NB_RADIUS

    # Test for all pairs with Max float distance
    lpq_tree = lpqtree.KDTree(metric=tree_m, radius=MAXDIST)
    lpq_tree.fit(v2)
    lpq_tree.radius_neighbors(v1, MAXDIST, return_distance=True, no_return=True)
    lpq_tree_res = lpq_tree.get_coo_matrix()
    assert np.allclose(lpq_res, lpq_tree_res.A), "test dist mtx"

    # Test at various radius
    for r in np.arange(min_v, max_v, step):
        val_mask = lpq_res < r
        values = lpq_res[val_mask]
        lpq_tree = lpqtree.KDTree(metric=tree_m, radius=r)
        lpq_tree.fit(v2)
        lpq_tree.radius_neighbors(v1, r, return_distance=True, no_return=True)
        lpq_tree_mtx = lpq_tree.get_coo_matrix()
        assert np.allclose(values, lpq_tree_mtx.A[val_mask]), "test dist search coo mtx"
        lpq_tree_mtx = lpq_tree.get_csr_matrix()
        assert np.allclose(values, lpq_tree_mtx.A[val_mask]), "test dist search csr mtx"

        # Test with mean_points
        if tree_m[-1] == "1":
            for mpts in [1, 2, 3, 4]:
                if v1.shape[1] % mpts == 0:
                    lpq_tree.fit_and_radius_search(v2, v1, r, nb_mpts=mpts)
                    lpq_tree_mtx = lpq_tree.get_coo_matrix()
                    assert np.allclose(values, lpq_tree_mtx.A[val_mask]), "test nb_mpts mtx"
        else:
            lpq_tree.fit_and_radius_search(v2, v1, r, nb_mpts=None)
            lpq_tree_mtx = lpq_tree.get_coo_matrix()
            assert np.allclose(values, lpq_tree_mtx.A[val_mask]), "test nb_mpts mtx"


def test_kdtree_l1_nd_f32():
    for m in range(1, 10):
        for n in range(1, 10):
            vts1 = np.random.rand(NB_MTX, m, n).astype(np.float32)
            vts2 = np.random.rand(NB_MTX, m, n).astype(np.float32)
            kdtree_test(vts1, vts2, p=1, q=1, tree_m="l1")
            kdtree_test(vts1, vts2, p=1, q=1, tree_m="l11")


def test_kdtree_l2_nd_f32():
    for m in range(1, 10):
        for n in range(1, 10):
            vts1 = np.random.rand(NB_MTX, m, n).astype(np.float32)
            vts2 = np.random.rand(NB_MTX, m, n).astype(np.float32)
            kdtree_test(vts1, vts2, p=2, q=2, tree_m="l2")
            kdtree_test(vts1, vts2, p=2, q=2, tree_m="l22")


def test_kdtree_l21_nd_f32():
    for m in range(1, 10):
        for n in [2, 3, 4]:
            vts1 = np.random.rand(NB_MTX, m, n).astype(np.float32)
            vts2 = np.random.rand(NB_MTX, m, n).astype(np.float32)
            kdtree_test(vts1, vts2, p=2, q=1, tree_m="l21")


def test_kdtree_l1_nd_f64():
    for m in range(1, 10):
        for n in range(1, 10):
            vts1 = np.random.rand(NB_MTX, m, n).astype(np.float64)
            vts2 = np.random.rand(NB_MTX, m, n).astype(np.float64)
            kdtree_test(vts1, vts2, p=1, q=1, tree_m="l1")
            kdtree_test(vts1, vts2, p=1, q=1, tree_m="l11")


def test_kdtree_l2_nd_f64():
    for m in range(1, 10):
        for n in range(1, 10):
            vts1 = np.random.rand(NB_MTX, m, n).astype(np.float64)
            vts2 = np.random.rand(NB_MTX, m, n).astype(np.float64)
            kdtree_test(vts1, vts2, p=2, q=2, tree_m="l2")
            kdtree_test(vts1, vts2, p=2, q=2, tree_m="l22")


def test_kdtree_l21_nd_f64():
    for m in range(1, 10):
        for n in [2, 3, 4]:
            vts1 = np.random.rand(NB_MTX, m, n).astype(np.float64)
            vts2 = np.random.rand(NB_MTX, m, n).astype(np.float64)
            kdtree_test(vts1, vts2, p=2, q=1, tree_m="l21")
