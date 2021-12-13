import pytest

import numpy as np
import lpqnanoflann
import lpqnanoflann.lpqpydist as lpqdist


MAXDIST = 9.9e32

# testing values
nb_mts = 100
nb_vts = 12
nb_dim = 2

np.random.seed(4)
vts1 = np.random.rand(nb_mts, nb_vts, nb_dim)
vts2 = np.random.rand(nb_mts, nb_vts, nb_dim)


def kdtree_test(v1, v2, p, q, tree_m, dtype):
    v1 = v1.astype(dtype)
    v2 = v2.astype(dtype)
    l21_res = lpqdist.lpq_allpairs(v1, v2, p=p, q=q)

    l21_tree = lpqnanoflann.KDTree(metric=tree_m, radius=MAXDIST)
    l21_tree.fit(v2)
    l21_tree.radius_neighbors(v1, MAXDIST, return_distance=True, return_array=False)
    l21_tree_res = l21_tree.get_coo_matrix()
    assert np.allclose(l21_res, l21_tree_res.A), "test dist mtx"


def test_kdtree_l1_nd_f32():
    for m in range(1, 10):
        for n in range(1, 10):
            vts1 = np.random.rand(nb_mts, m, n)
            vts2 = np.random.rand(nb_mts, m, n)
            kdtree_test(vts1, vts2, p=1, q=1, tree_m="l1", dtype=np.float32)
            kdtree_test(vts1, vts2, p=1, q=1, tree_m="l11", dtype=np.float32)


def test_kdtree_l2_nd_f32():
    for m in range(1, 10):
        for n in range(1, 10):
            vts1 = np.random.rand(nb_mts, m, n)
            vts2 = np.random.rand(nb_mts, m, n)
            kdtree_test(vts1, vts2, p=2, q=2, tree_m="l2", dtype=np.float32)
            kdtree_test(vts1, vts2, p=2, q=2, tree_m="l22", dtype=np.float32)


def test_kdtree_l21_nd_f32():
    for m in range(2, 10):
        for n in [2, 3, 4]:
            print(("l21", m, n))
            vts1 = np.random.rand(nb_mts, m, n)
            vts2 = np.random.rand(nb_mts, m, n)
            kdtree_test(vts1, vts2, p=2, q=1, tree_m="l21", dtype=np.float32)

#
# def test_kdtree_l22_f32():
#     kdtree_test(vts1, vts2, p=2, q=2, tree_m="l2", dtype=np.float32)
#
#
# def test_kdtree_l11_f64():
#     kdtree_test(vts1, vts2, p=1, q=1, tree_m="l1", dtype=np.float64)
#
#
# def test_kdtree_l21_f64():
#     kdtree_test(vts1, vts2, p=2, q=1, tree_m="l21", dtype=np.float64)
#
#
# def test_kdtree_l22_f64():
#     kdtree_test(vts1, vts2, p=2, q=2, tree_m="l2", dtype=np.float64)
#
#
# def test_kdtree_other():
#     v1 = np.random.rand(nb_mts, 4, 3)
#     v2 = np.random.rand(nb_mts, 4, 3)
#     l21_res = lpqdist.lpq_allpairs(v1, v2, p=2, q=1)
#
#     l21_tree = lpqnanoflann.KDTree(metric="l21", radius=MAXDIST)
#     l21_tree.fit(v2.reshape((nb_mts, -1)))
#     l21_tree.radius_neighbors(v1.reshape((nb_mts, -1)), MAXDIST, return_distance=True, return_array=False)
#     l21_tree_res = l21_tree.get_coo_matrix()
#     assert np.allclose(l21_res, l21_tree_res.A), "test dist mtx"
