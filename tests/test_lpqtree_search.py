import pytest

import numpy as np
import lpqtree
import lpqtree.lpqpydist as lpqdist

EPS = 1.0e-4

# testing values
NB_MTX = 9  # number of element in the search
NB_VTS = 7  # number of element in the tree
NB_RADIUS = 10
NB_K_QUERY = 5
np.random.seed(6)
LEAF_SIZE = 1  # force to test kd-tree split distance

# Testing the single-threaded, and a multithreaded
NB_THREADS = [1, 4]
FLOAT_TYPES = [np.float32, np.float64]


def compute_full_mtx(v1, v2, p, q):
    lpq_full = lpqdist.lpq_allpairs(v1, v2, p=p, q=q)
    assert(lpq_full.shape == (len(v1), len(v2)))
    min_v = lpq_full.min() + EPS
    max_v = lpq_full.max() + EPS
    return lpq_full, min_v, max_v

@pytest.mark.parametrize("p", range(1, 5))
@pytest.mark.parametrize("q", range(1, 5))
@pytest.mark.parametrize("m", range(1, 5))
@pytest.mark.parametrize("n", range(1, 5))
@pytest.mark.parametrize("nb_cpu", [1, 8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kdtree_basic(p: int, q: int, m: int, n: int, nb_cpu, dtype):
    lpq_str = f"l{q}" if n == 1 else f"l{p}{q}"
    v1 = np.random.rand(NB_MTX, m, n).astype(dtype)
    v2 = np.random.rand(NB_VTS, m, n).astype(dtype)

    kdtree_tests(v1, v2, p=p, q=q, lpq_str=lpq_str, nb_cpu=nb_cpu)
    kdtree_self_generic(v1, p, q, lpq_str, nb_cpu, mode="self")
    kdtree_self_generic(v1, p, q, lpq_str, nb_cpu, mode="sym")

def kdtree_tests(v1, v2, p, q, lpq_str, nb_cpu):
    lpq_full, min_v, max_v = compute_full_mtx(v1, v2, p, q)

    # Test k-NN for all pairs
    lpq_tree = lpqtree.KDTree(metric=lpq_str, radius=max_v, leaf_size=LEAF_SIZE)
    lpq_tree.fit(v2)
    lpq_tree.query(v1, NB_VTS, return_distance=False, n_jobs=nb_cpu)
    lpq_tree_res = lpq_tree.get_coo_matrix()
    assert np.allclose(lpq_full, lpq_tree_res.toarray()), "test dist mtx lpq_full"

    # Test at various k-NN
    argsort_lpq_full = lpq_full.argsort(axis=-1)
    for k in np.arange(1, NB_VTS, NB_VTS//NB_K_QUERY, dtype=int):
        res, dists = lpq_tree.query(v1, k, return_distance=True, n_jobs=nb_cpu)
        mtx_knn = lpq_tree.get_coo_matrix()
        assert(mtx_knn.shape[0] == NB_MTX)
        assert(mtx_knn.shape[1] == NB_VTS)
        for i in range(NB_MTX):
            # Can fail when 2 distances are the same or very similar
            # assert np.all(np.in1d(argsort_lpq_full[i][:k], res[i]))
            assert np.allclose(lpq_full[i][argsort_lpq_full[i][:k]], np.sort(dists[i]))

        # Test radius knn (at radius=MAX_RADIUS)
        lpq_tree.radius_knn(v1, k, radius=max_v, return_distance=True, n_jobs=nb_cpu)
        mtx_knn2 = lpq_tree.get_coo_matrix()
        assert(mtx_knn2.shape[0] == NB_MTX)
        assert(mtx_knn2.shape[1] == NB_VTS)
        assert np.all(lpq_tree.get_count() == k)
        assert np.allclose(mtx_knn.toarray(), mtx_knn2.toarray())

    # Test Radius search for all pairs with Max float distance
    lpq_tree = lpqtree.KDTree(metric=lpq_str, radius=max_v, leaf_size=LEAF_SIZE)
    lpq_tree.fit(v2)
    lpq_tree.radius_neighbors(v1, max_v, return_distance=True, no_return=True, n_jobs=nb_cpu)
    lpq_tree_res = lpq_tree.get_coo_matrix()
    assert np.allclose(lpq_full, lpq_tree_res.toarray()), "test radius_neighbors mtx"
    # Test run, with no distances and return
    ids_x, ids_tree = lpq_tree.radius_neighbors(v1, radius=max_v, return_distance=False, no_return=False, n_jobs=nb_cpu)
    assert np.all(ids_x == lpq_tree_res.row), "test radius_neighbors no_dist, row"
    assert np.all(ids_tree == lpq_tree_res.col), "test radius_neighbors no_dist, col;"
    assert np.all(lpq_tree.get_count() == NB_VTS), "radius_neighbors count"
    # Test Radius search Count
    counts = lpq_tree.radius_neighbors(v1, radius=max_v, return_distance=False, count_only=True, n_jobs=nb_cpu)
    assert np.all(counts == NB_VTS), "test radius_neighbors count_only"

    # Test at various radius
    for r in np.arange(min_v, max_v, (max_v - min_v) / NB_RADIUS):
        val_mask = lpq_full <= r
        values = lpq_full[val_mask]
        lpq_tree = lpqtree.KDTree(metric=lpq_str, radius=r + EPS, leaf_size=LEAF_SIZE)
        lpq_tree.fit(v2)
        lpq_tree.radius_neighbors(v1, radius=r + EPS, return_distance=True, no_return=True, n_jobs=nb_cpu)
        lpq_tree_mtx = lpq_tree.get_coo_matrix()
        assert np.allclose(values, lpq_tree_mtx.toarray()[val_mask]), "test dist search coo mtx"
        lpq_tree_mtx = lpq_tree.get_csr_matrix()
        assert np.allclose(values, lpq_tree_mtx.toarray()[val_mask]), "test dist search csr mtx"

        # Test radius knn (at k=NB_VTS)
        lpq_tree.radius_knn(v1, k=NB_VTS, radius=r + EPS, return_distance=True, no_return=True, n_jobs=nb_cpu)
        assert np.allclose(lpq_tree_mtx.toarray(), lpq_tree.get_csc_matrix().toarray())

        # Test radius search count_only
        lpq_tree_mtx.data[:] = 1.0
        counts1 = np.squeeze(np.asarray(lpq_tree_mtx.sum(1), dtype=np.uint64))  # Count nb elements per line in mtx
        assert np.all(counts1 == lpq_tree.get_count()), "test radius_neighbors count_test"
        counts2 = lpq_tree.radius_neighbors(v1, radius=r + EPS, return_distance=False, count_only=True, n_jobs=nb_cpu)
        assert np.all(counts1 == counts2), "test radius_neighbors count_onlylpq_full"

        # Test with mean_points
        if lpq_str in ["l1", "l2", "l11", "l21"]:
            for mpts in [1, 2, 3, 4]:
                if v1.shape[1] % mpts == 0 and mpts < v1.shape[1]:
                    lpq_tree.fit_and_radius_search(v2, v1, radius=r + EPS, nb_mpts=mpts, n_jobs=nb_cpu)
                    lpq_tree_mtx = lpq_tree.get_coo_matrix()
                    assert np.allclose(values, lpq_tree_mtx.toarray()[val_mask]), "test nb_mpts mtxlpq_full"
                    lpq_tree_mtx.data[:] = 1.0
                    counts1 = np.squeeze(np.asarray(lpq_tree_mtx.sum(1), dtype=np.uint64))  # Count nb elements per line in mtx
                    assert np.all(counts1 == lpq_tree.get_count()), "test radius_neighbors count_test {nb_cpu} threads"
                    lpq_tree.fit_and_radius_search(v2, v1, radius=r + EPS, nb_mpts=mpts, n_jobs=nb_cpu, count_only=True)
                    counts2 = lpq_tree.get_count()
                    assert np.all(counts1 == counts2), "test nb_mpts count_onlylpq_full"
        else:
            lpq_tree.fit_and_radius_search(v2, v1, radius=r + EPS, nb_mpts=None, n_jobs=nb_cpu)
            lpq_tree_mtx = lpq_tree.get_coo_matrix()
            assert np.allclose(values, lpq_tree_mtx.toarray()[val_mask]), "test nb_mpts mtxlpq_full"
            lpq_tree_mtx.data[:] = 1.0
            counts1 = np.squeeze(np.asarray(lpq_tree_mtx.sum(1), dtype=np.uint64))  # Count nb elements per line in mtx
            assert np.all(counts1 == lpq_tree.get_count()), "test radius_neighbors count_test {nb_cpu} threads"
            lpq_tree.fit_and_radius_search(v2, v1, radius=r + EPS, nb_mpts=None, n_jobs=nb_cpu, count_only=True)
            counts2 = lpq_tree.get_count()
            assert np.all(counts1 == counts2), "test nb_mpts count_onlylpq_full"


def kdtree_self_generic(v1, p, q, lpq_str, nb_cpu, mode="self"):
    assert mode in {"self", "sym"}

    # Setup v2
    if mode == "self":
        v2 = v1
    elif mode == "sym":
        v2 = np.concatenate([v1, np.flip(v1, axis=1)])

    # Compute full lpq matrix
    lpq_full, min_v, max_v = compute_full_mtx(v1, v2, p, q)
    assert np.allclose(np.diag(lpq_full), 0.0)

    nb_pts_to_search = 0
    both_direction = False
    if mode == "self":
        assert np.allclose(lpq_full, lpq_full.T)

        # Test - KNN, all pairs
        nb_results = len(v2) - 1
        lpq_tree = lpqtree.KDTree(metric=lpq_str, radius=max_v, leaf_size=LEAF_SIZE)
        lpq_tree.fit(v2)
        lpq_tree.self_query(k=nb_results, return_distance=False, n_jobs=nb_cpu)
        coo = lpq_tree.get_coo_matrix()

        assert np.allclose(lpq_full, coo.toarray()), "query, dist mtxlpq_full"
        assert not (coo.row == coo.col % NB_MTX).any(), "query, no diagonal (no self)lpq_full"
        assert np.all(np.bincount(coo.row, minlength=NB_MTX) == nb_results), "query, same number of entries per row"

        # Test - KNN, multiple k
        argsort_lpq_res = lpq_full.argsort(axis=-1)[:, 1:]  # remove "first" / self
        for k in np.arange(2, nb_results, nb_results // NB_K_QUERY, dtype=int):
            res, dists = lpq_tree.self_query(k=k, return_distance=True, n_jobs=nb_cpu)
            mtx_knn = lpq_tree.get_coo_matrix()

            assert (mtx_knn.shape[0] == NB_MTX)
            assert (mtx_knn.shape[1] == NB_MTX)
            assert not (mtx_knn.row == mtx_knn.col).any(), "test no diagonal (no self)lpq_full"
            for i in range(NB_MTX):
                # Can fail when 2 distances are the same or very similar
                # assert np.all(np.isin(argsort_lpq_res[i][:k], res[i]))
                assert np.allclose(lpq_full[i][argsort_lpq_res[i][:k]], np.sort(dists[i]))

    elif mode == "sym":
        assert np.allclose(lpq_full[:, :NB_MTX], lpq_full[:, :NB_MTX].T)
        assert np.allclose(lpq_full[:, NB_MTX:], lpq_full[:, NB_MTX:].T)

        # remove "self" matches
        self_idx = np.concatenate((np.arange(NB_MTX), np.arange(NB_MTX)))
        self_idy = np.arange(2 * NB_MTX)
        lpq_full[self_idx, self_idy] = 0.0
        nb_pts_to_search = len(v1)
        both_direction = True

    # Test - Radius
    tree = lpqtree.KDTree(metric=lpq_str, leaf_size=LEAF_SIZE)
    tree.fit(v2)
    tree.self_radius_neighbors(max_v, n_jobs=nb_cpu, nb_pts_to_search=nb_pts_to_search)

    coo = tree.get_coo_matrix()
    assert np.allclose(lpq_full, coo.toarray())
    assert not (coo.row == coo.col % NB_MTX).any()

    for r in np.arange(min_v, max_v, (max_v - min_v) / NB_RADIUS):
        mask = lpq_full <= r
        values = lpq_full[mask]

        tree = lpqtree.KDTree(metric=lpq_str, radius=r + EPS, leaf_size=LEAF_SIZE)
        tree.fit(v2)
        tree.self_radius_neighbors(r + EPS, n_jobs=nb_cpu, nb_pts_to_search=nb_pts_to_search)

        coo = tree.get_coo_matrix()
        assert np.allclose(values, coo.toarray()[mask])
        assert np.allclose(values, tree.get_csr_matrix().toarray()[mask])
        assert not (coo.row == coo.col % NB_MTX).any()

        # Test - Mean points
        if lpq_str in ["l1", "l2", "l11", "l21"]:
            for mpts in [1, 2, 3, 4]:
                if v1.shape[1] % mpts == 0 and mpts < v1.shape[1]:
                    tree.fit_and_self_radius_search(v1, radius=r + EPS, nb_mpts=mpts, n_jobs=nb_cpu, both_direction=both_direction)
                    coo = tree.get_coo_matrix()
                    assert np.allclose(values, coo.toarray()[mask])
                    assert not (coo.row == coo.col % NB_MTX).any()

                    coo.data[:] = 1.0
                    counts = np.asarray(coo.sum(1)).squeeze().astype(np.uint64)
                    assert np.all(counts == tree.get_count())
        else:
            tree.fit_and_self_radius_search(v1, radius=r + EPS, nb_mpts=None, n_jobs=nb_cpu, both_direction=both_direction)
            coo = tree.get_coo_matrix()
            assert np.allclose(values, coo.toarray()[mask])
            assert not (coo.row == coo.col % NB_MTX).any()

            coo.data[:] = 1.0
            counts = np.asarray(coo.sum(1)).squeeze().astype(np.uint64)
            assert np.all(counts == tree.get_count())

