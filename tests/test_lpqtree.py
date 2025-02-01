import pytest

import numpy as np
import lpqtree
import lpqtree.lpqpydist as lpqdist

EPS = 1.0e-4
MAXDIST = 9.9e32

# testing values
NB_MTX = 50  # number of element in the search
NB_RADIUS = 10
NB_K_QUERY = 5
np.random.seed(6)
LEAF_SIZE = 1  # force to test kd-tree split distance

# Testing the single-threaded, and a multithreaded
NB_THREADS = [1, 4]


def kdtree_test(v1, v2, p, q, nb_cpu):
    assert 1 <= p < 10
    assert 1 <= q < 10
    tree_m = "l" + str(p) + str(q)

    lpq_res = lpqdist.lpq_allpairs(v1, v2, p=p, q=q)
    min_v = lpq_res.min() + EPS
    max_v = lpq_res.max() + EPS
    r_step = (max_v - min_v) / NB_RADIUS

    # Test k-NN for all pairs
    lpq_tree = lpqtree.KDTree(metric=tree_m, radius=MAXDIST, leaf_size=LEAF_SIZE)
    lpq_tree.fit(v2)
    lpq_tree.query(v1, NB_MTX, return_distance=False, n_jobs=nb_cpu)
    lpq_tree_res = lpq_tree.get_coo_matrix()
    assert np.allclose(lpq_res, lpq_tree_res.toarray()), f"test dist mtx; {nb_cpu} threads"

    # Test at various k-NN
    argsort_lpq_res = lpq_res.argsort(axis=-1)
    for k in np.arange(1, NB_MTX, NB_MTX//NB_K_QUERY, dtype=int):
        res, dists = lpq_tree.query(v1, k, return_distance=True, n_jobs=nb_cpu)
        for i in range(NB_MTX):
            # Can fail when 2 distances are the same or very similar
            # assert np.all(np.in1d(argsort_lpq_res[i][:k], res[i]))
            assert np.allclose(lpq_res[i][argsort_lpq_res[i][:k]], np.sort(dists[i]))

    # Test Radius search for all pairs with Max float distance
    lpq_tree = lpqtree.KDTree(metric=tree_m, radius=MAXDIST, leaf_size=LEAF_SIZE)
    lpq_tree.fit(v2)
    lpq_tree.radius_neighbors(v1, MAXDIST, return_distance=True, no_return=True, n_jobs=nb_cpu)
    lpq_tree_res = lpq_tree.get_coo_matrix()
    assert np.allclose(lpq_res, lpq_tree_res.toarray()), "test radius_neighbors mtx; {cpu} threads"
    # Test run, with no distances and return
    ids_x, ids_tree = lpq_tree.radius_neighbors(v1, radius=MAXDIST, return_distance=False, no_return=False, n_jobs=nb_cpu)
    assert np.all(ids_x == lpq_tree_res.row), f"test radius_neighbors no_dist, row; {nb_cpu} threads"
    assert np.all(ids_tree == lpq_tree_res.col), f"test radius_neighbors no_dist, col; {nb_cpu} threads"
    assert np.all(lpq_tree.get_count() == NB_MTX), f"radius_neighbors count; {nb_cpu} threads"
    # Test Radius search Count
    counts = lpq_tree.radius_neighbors(v1, radius=MAXDIST, return_distance=False, count_only=True, n_jobs=nb_cpu)
    assert np.all(counts == NB_MTX), f"test radius_neighbors count_only; {nb_cpu} threads"

    # Test at various radius
    for r in np.arange(min_v, max_v, r_step):
        val_mask = lpq_res <= r
        values = lpq_res[val_mask]
        lpq_tree = lpqtree.KDTree(metric=tree_m, radius=r + EPS, leaf_size=LEAF_SIZE)
        lpq_tree.fit(v2)
        lpq_tree.radius_neighbors(v1, radius=r + EPS, return_distance=True, no_return=True, n_jobs=nb_cpu)
        lpq_tree_mtx = lpq_tree.get_coo_matrix()
        assert np.allclose(values, lpq_tree_mtx.toarray()[val_mask]), f"test dist search coo mtx; {nb_cpu} threads"
        lpq_tree_mtx = lpq_tree.get_csr_matrix()
        assert np.allclose(values, lpq_tree_mtx.toarray()[val_mask]), f"test dist search csr mtx; {nb_cpu} threads"
        lpq_tree_mtx.data[:] = 1.0
        counts1 = np.squeeze(np.asarray(lpq_tree_mtx.sum(1), dtype=np.uint64))  # Count nb elements per line in mtx
        assert np.all(counts1 == lpq_tree.get_count()), f"test radius_neighbors count_test {nb_cpu} threads"
        counts2 = lpq_tree.radius_neighbors(v1, radius=r + EPS, return_distance=False, count_only=True, n_jobs=nb_cpu)
        assert np.all(counts1 == counts2), f"test radius_neighbors count_only; {nb_cpu} threads"

        # Test with mean_points
        if tree_m in ["l1", "l2", "l11", "l21"]:
            for mpts in [1, 2, 3, 4]:
                if v1.shape[1] % mpts == 0:
                    lpq_tree.fit_and_radius_search(v2, v1, radius=r + EPS, nb_mpts=mpts, n_jobs=nb_cpu)
                    lpq_tree_mtx = lpq_tree.get_coo_matrix()
                    assert np.allclose(values, lpq_tree_mtx.toarray()[val_mask]), f"test nb_mpts mtx; {nb_cpu} threads"
                    lpq_tree_mtx.data[:] = 1.0
                    counts1 = np.squeeze(np.asarray(lpq_tree_mtx.sum(1), dtype=np.uint64))  # Count nb elements per line in mtx
                    assert np.all(counts1 == lpq_tree.get_count()), f"test radius_neighbors count_test {nb_cpu} threads"
                    lpq_tree.fit_and_radius_search(v2, v1, radius=r + EPS, nb_mpts=mpts, n_jobs=nb_cpu, count_only=True)
                    counts2 = lpq_tree.get_count()
                    assert np.all(counts1 == counts2), f"test nb_mpts count_only; {nb_cpu} threads"
        else:
            lpq_tree.fit_and_radius_search(v2, v1, radius=r + EPS, nb_mpts=None, n_jobs=nb_cpu)
            lpq_tree_mtx = lpq_tree.get_coo_matrix()
            assert np.allclose(values, lpq_tree_mtx.toarray()[val_mask]), f"test nb_mpts mtx; {nb_cpu} threads"
            lpq_tree_mtx.data[:] = 1.0
            counts1 = np.squeeze(np.asarray(lpq_tree_mtx.sum(1), dtype=np.uint64))  # Count nb elements per line in mtx
            assert np.all(counts1 == lpq_tree.get_count()), f"test radius_neighbors count_test {nb_cpu} threads"
            lpq_tree.fit_and_radius_search(v2, v1, radius=r + EPS, nb_mpts=None, n_jobs=nb_cpu, count_only=True)
            counts2 = lpq_tree.get_count()
            assert np.all(counts1 == counts2), f"test nb_mpts count_only; {nb_cpu} threads"


def test_kdtree_l1_nd_f32():
    for m in range(1, 10):
        for n in range(1, 10):
            vts1 = np.random.rand(NB_MTX, m, n).astype(np.float32)
            vts2 = np.random.rand(NB_MTX, m, n).astype(np.float32)
            for cpu in NB_THREADS:
                kdtree_test(vts1, vts2, p=1, q=1, nb_cpu=cpu)


def test_kdtree_l2_nd_f32():
    for m in range(1, 10):
        for n in range(1, 10):
            vts1 = np.random.rand(NB_MTX, m, n).astype(np.float32)
            vts2 = np.random.rand(NB_MTX, m, n).astype(np.float32)
            for cpu in NB_THREADS:
                kdtree_test(vts1, vts2, p=2, q=2, nb_cpu=cpu)


def test_kdtree_l12_nd_f32():
    for m in range(1, 10):
        for n in [2, 3, 4]:
            vts1 = np.random.rand(NB_MTX, m, n).astype(np.float32)
            vts2 = np.random.rand(NB_MTX, m, n).astype(np.float32)
            for cpu in NB_THREADS:
                kdtree_test(vts1, vts2, p=1, q=2, nb_cpu=cpu)


def test_kdtree_l21_nd_f32():
    for m in range(1, 10):
        for n in [2, 3, 4]:
            vts1 = np.random.rand(NB_MTX, m, n).astype(np.float32)
            vts2 = np.random.rand(NB_MTX, m, n).astype(np.float32)
            for cpu in NB_THREADS:
                kdtree_test(vts1, vts2, p=2, q=1, nb_cpu=cpu)


def test_kdtree_l1_nd_f64():
    for m in range(1, 10):
        for n in range(1, 10):
            vts1 = np.random.rand(NB_MTX, m, n).astype(np.float64)
            vts2 = np.random.rand(NB_MTX, m, n).astype(np.float64)
            for cpu in NB_THREADS:
                kdtree_test(vts1, vts2, p=1, q=1, nb_cpu=cpu)


def test_kdtree_l2_nd_f64():
    for m in range(1, 10):
        for n in range(1, 10):
            vts1 = np.random.rand(NB_MTX, m, n).astype(np.float64)
            vts2 = np.random.rand(NB_MTX, m, n).astype(np.float64)
            for cpu in NB_THREADS:
                kdtree_test(vts1, vts2, p=2, q=2, nb_cpu=cpu)


def test_kdtree_l12_nd_f64():
    for m in range(1, 10):
        for n in [2, 3, 4]:
            vts1 = np.random.rand(NB_MTX, m, n).astype(np.float64)
            vts2 = np.random.rand(NB_MTX, m, n).astype(np.float64)
            for cpu in NB_THREADS:
                kdtree_test(vts1, vts2, p=1, q=2, nb_cpu=cpu)


def test_kdtree_l21_nd_f64():
    for m in range(1, 10):
        for n in [2, 3, 4]:
            vts1 = np.random.rand(NB_MTX, m, n).astype(np.float64)
            vts2 = np.random.rand(NB_MTX, m, n).astype(np.float64)
            for cpu in NB_THREADS:
                kdtree_test(vts1, vts2, p=2, q=1, nb_cpu=cpu)


def test_kdtree_lp_nd_f32():
    for m in range(1, 10):
        vts1 = np.random.rand(NB_MTX, m, 1).astype(np.float32)
        vts2 = np.random.rand(NB_MTX, m, 1).astype(np.float32)
        for i in [3, 4, 5, 6, 7, 8, 9]:
            for cpu in NB_THREADS:
                kdtree_test(vts1, vts2, p=i, q=i, nb_cpu=cpu)


def test_kdtree_lp_nd_f64():
    for m in range(1, 10):
        vts1 = np.random.rand(NB_MTX, m, 1).astype(np.float64)
        vts2 = np.random.rand(NB_MTX, m, 1).astype(np.float64)
        for i in [3, 4, 5, 6, 7, 8, 9]:
            for cpu in NB_THREADS:
                kdtree_test(vts1, vts2, p=i, q=i, nb_cpu=cpu)


def test_kdtree_lpq_nd_f32():
    for m in range(2, 6):
        for n in range(2, 6):
            vts1 = np.random.rand(NB_MTX, m, n).astype(np.float32)
            vts2 = np.random.rand(NB_MTX, m, n).astype(np.float32)
            for p in range(1, 10):
                for q in range(1, 10):
                    for cpu in NB_THREADS:
                        kdtree_test(vts1, vts2, p=p, q=q, nb_cpu=cpu)


def test_kdtree_lpq_nd_f64():
    for m in range(2, 6):
        for n in range(2, 6):
            vts1 = np.random.rand(NB_MTX, m, n).astype(np.float64)
            vts2 = np.random.rand(NB_MTX, m, n).astype(np.float64)
            for p in range(1, 10):
                for q in range(1, 10):
                    for cpu in NB_THREADS:
                        kdtree_test(vts1, vts2, p=p, q=q, nb_cpu=cpu)
