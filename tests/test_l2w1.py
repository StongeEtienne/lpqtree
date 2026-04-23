import pytest

import numpy as np

import lpqtree
from lpqtree.lpqpydist import l21, l2, l1


LEAF_SIZE = 1  # force to test kd-tree split distance
NB_MTX = 99
NB_VTS = 8
NB_DIM = 3
EPS = 1e-8

# Testing the single-threaded, and a multithreaded
NB_THREADS = [1, 4]


def l2w1(mts, w):
    """Compute the L21 norm, along the two last axis"""
    return np.sum(w * l2(mts, axis=-1, keepdims=False), axis=-1, keepdims=False)


def test_l2w1():
    s1 = np.random.rand(NB_MTX, NB_VTS, NB_DIM)
    s2 = np.random.rand(NB_MTX, NB_VTS, NB_DIM)
    w = np.abs(np.random.rand(NB_VTS)) + EPS
    wr = w.reshape((1, NB_VTS, 1))

    d = s2 - s1
    s = np.pi

    assert np.allclose(l21(d), l2w1(d, np.ones(8))), "L2w1 with w=1 equals l21, test failed"
    assert np.allclose(l21(d), l2w1(d, 1.0)), "L2w1 with w=1 equals l21, test failed"

    assert np.allclose(l21(d*wr), l2w1(d, w)), "L2w1 with w=1 equals l21, test failed"
    assert np.allclose(l21(d*wr), l21(s2*wr - s1*wr)), "L2w1 with w=1 equals l2(L1), test failed"
    assert np.allclose(l1(l2(d*wr)), l2w1(d, w)), "L2w1 with w=1 equals l1(l2), test failed"

    assert np.allclose(l2w1(d, w*0.0), l2w1(d, w)*0.0), "L2w1 zero failed"
    assert np.allclose(l2w1(d*s, w), l2w1(d, w)*s), "L2w1 d scaling failed"
    assert np.allclose(l2w1(d, w*s), l2w1(d, w)*s), "L2w1 w scaling failed"
    assert np.allclose(l2w1(-d, w), l2w1(d, w)), "L2w1 negative failed"

    assert np.all(l2w1(s1 + s2, w) <= l2w1(s1, w) + l2w1(s2, w)), "L2w1 Triangle inequality failed"

    # Compare all dist, create a full dense matrix
    res1 = np.zeros((len(s1), len(s2)), dtype=s1.dtype)
    res2 = np.zeros((len(s1), len(s2)), dtype=s1.dtype)
    for j in range(len(s2)):
        res1[:, j] = l2w1(s1 - s2[j], w=w)
        res2[:, j] = l2w1(s2[j] - s1, w=w)

    assert np.allclose(res1, res2), "L2w1 Symmetric"

    for cpu in NB_THREADS:
        # KNN (max k)
        lpq_tree = lpqtree.KDTree(metric="l21", leaf_size=LEAF_SIZE)
        lpq_tree.fit(s2*wr)
        lpq_tree.query(s1*wr, NB_MTX, return_distance=True, n_jobs=cpu)
        lpq_res = lpq_tree.get_coo_matrix()
        assert np.allclose(res1, lpq_res.toarray()), f"L2w1 tree knn; {cpu} threads"

        # KNN_radius (max k, max radius)
        radius = res1.max() + EPS
        lpq_tree.radius_knn(s1*wr, k=NB_MTX, radius=radius, return_distance=True, n_jobs=cpu)
        lpq_res = lpq_tree.get_coo_matrix()
        assert np.allclose(res1, lpq_res.toarray()), f"L2w1 tree radius; {cpu} threads"

        # Radius (max radius)
        radius = res1.max() + EPS
        lpq_tree.radius_neighbors(s1*wr, radius=radius, return_distance=True, n_jobs=cpu)
        lpq_res = lpq_tree.get_coo_matrix()
        assert np.allclose(res1, lpq_res.toarray()), f"L2w1 tree radius; {cpu} threads"

        # Radius (1/2 radius)
        res3 = np.copy(res1)
        res3[res3 >= radius/2.0] = 0
        lpq_tree.radius_neighbors(s1*wr, radius=radius/2.0, return_distance=True, n_jobs=cpu)
        lpq_res = lpq_tree.get_coo_matrix()
        assert np.allclose(res3, lpq_res.toarray()), f"L2w1 tree radius; {cpu} threads"

        # KNN_radius (max k, 1/2 radius)
        radius = res1.max() + EPS
        lpq_tree.radius_knn(s1*wr, k=NB_MTX, radius=radius/2.0, return_distance=True, n_jobs=cpu)
        lpq_res = lpq_tree.get_coo_matrix()
        assert np.allclose(res3, lpq_res.toarray()), f"L2w1 tree radius; {cpu} threads"

