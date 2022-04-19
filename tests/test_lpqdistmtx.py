import pytest

import numpy as np
import lpqtree.lpqpydist as lpqdist

# testing values
NBMTS = 10
MAXLPQ = 5
MAXVEC = 12
MAXDIM = 8


def pairwise_test(v1, v2, p, q):
    dists = lpqdist.lpq_pairwise(v1, v2, p=p, q=q)
    for i in range(NBMTS):
        assert np.allclose(dists[i], lpqdist.lpq(v1[i] - v2[i], p=p, q=q)), "test lpq_pairwise"


def test_lpq_pairwise():
    np.random.seed(4)
    for nb_i in range(1, MAXVEC):
        for nb_j in range(1, MAXDIM):
            vts1 = np.random.rand(NBMTS, nb_i, nb_j)
            vts2 = np.random.rand(NBMTS, nb_i, nb_j)
            for p in range(1, MAXLPQ):
                for q in range(1, MAXLPQ):
                    pairwise_test(vts1, vts2, p, q)


def allpairs_test(v1, v2, p, q):
    dists = lpqdist.lpq_allpairs(v1, v2, p=p, q=q)
    d_iter = lpqdist.lpq_allpairs_iter(v1, v2, p=p, q=q)
    for i in range(NBMTS):
        for j in range(NBMTS):
            assert np.allclose(dists[i, j], lpqdist.lpq(v1[i] - v2[j], p=p, q=q)), "test lpq_allpairs"
            assert np.allclose(d_iter[i, j], lpqdist.lpq(v1[i] - v2[j], p=p, q=q)), "test lpq_allpairs_iter"


def test_lpq_allpairs():
    np.random.seed(4)
    for nb_i in range(1, MAXVEC):
        for nb_j in range(1, MAXDIM):
            vts1 = np.random.rand(NBMTS, nb_i, nb_j)
            vts2 = np.random.rand(NBMTS, nb_i, nb_j)
            for p in range(1, MAXLPQ):
                for q in range(1, MAXLPQ):
                    allpairs_test(vts1, vts2, p, q)
