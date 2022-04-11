# Etienne St-Onge

import numpy as np


# Vector norm operator
def l1(vts, axis=-1, keepdims=False):
    return np.sum(np.abs(vts), axis=axis, keepdims=keepdims)


def l2(vts, axis=-1, keepdims=False):
    return np.sqrt(np.sum(np.square(vts), axis=axis, keepdims=keepdims))


def lp(vts, p, axis=-1, keepdims=False):
    inv_p = 1.0/p
    return np.sum(np.abs(vts**p), axis=axis, keepdims=keepdims)**inv_p


# Matrix norm operator, along the two last axis
def l11(mts):
    # Equivalent to L1 matrix norm
    return np.sum(l1(mts, axis=-1, keepdims=False), axis=-1, keepdims=False)


def l12(mts):
    return l2(l1(mts, axis=-1, keepdims=False), axis=-1, keepdims=False)


def l21(mts):
    return np.sum(l2(mts, axis=-1, keepdims=False), axis=-1, keepdims=False)


def l22(mts):
    # Equivalent to L2 matrix norm
    return np.sqrt(np.sum(np.sum(np.square(mts), axis=-1, keepdims=False), axis=-1, keepdims=False))


def lp1(mts, p):
    return np.sum(lp(mts, p=p, axis=-1, keepdims=False), axis=-1, keepdims=False)


def lp2(mts, p):
    return l2(lp(mts, p=p, axis=-1, keepdims=False), axis=-1, keepdims=False)


def l1q(mts, q):
    inv_q = 1.0/q
    return np.sum(l1(mts, axis=-1, keepdims=False)**q, axis=-1, keepdims=False)**inv_q


def l2q(mts, q):
    inv_q = 1.0/q
    return np.sum(l2(mts, axis=-1, keepdims=False)**q, axis=-1, keepdims=False)**inv_q


def lpq(mts, p, q):
    inv_q = 1.0/q
    return np.sum(lp(mts, p=p, axis=-1, keepdims=False)**q, axis=-1, keepdims=False)**inv_q


# Matrix norm operator with mean instead of sum (homogeneous to lp1)
def l1m(mts):
    return np.mean(l1(mts, axis=-1, keepdims=False), axis=-1, keepdims=False)


def l2m(mts):
    return np.mean(l2(mts, axis=-1, keepdims=False), axis=-1, keepdims=False)


def lpm(mts, p):
    return np.mean(lp(mts, p=p, axis=-1, keepdims=False), axis=-1, keepdims=False)


# Matrix norm switch
def lpq_switch(mts, p, q):
    assert (p >= 1)

    if q == "m":
        if p == 1:
            return l1m(mts)
        elif p == 2:
            return l2m(mts)
        else:
            return lpm(mts, p=p)
    # else:
    assert (q >= 1)
    if q == 1:
        if p == 1:
            return l11(mts)
        elif p == 2:
            return l21(mts)
        else:
            return lp1(mts, p=p)

    if q == 2:
        if p == 1:
            return l12(mts)
        elif p == 2:
            return l22(mts)
        else:
            return lp2(mts, p=p)

    # else: # q > 2
    if p == 1:
        return l1q(mts, q=q)
    elif p == 2:
        return l2q(mts, q=q)

    # else: # p > 2 and q > 2
    return lpq(mts, p=p, q=q)


def lpq_str_switch(mts, norm="l22"):
    assert len(norm) == 3
    assert norm[0].lower() == "l"
    p = int(norm[1])
    if norm[2] == "m":
        q = "m"
    else:
        q = int(norm[2])
    return lpq_switch(mts, p=p, q=q)


# Compute the distance between each pair
def lpq_pairwise(mts1, mts2, p, q):
    assert mts1.ndim == 3
    assert mts2.ndim == 3
    assert np.alltrue(mts1.shape[1:] == mts2.shape[1:])

    return lpq_switch(mts1 - mts2, p=p, q=q)


# Compute the distance between all pairs (Warning about memory)
def lpq_allpairs(mts1, mts2, p, q):
    assert mts1.ndim == 3
    assert mts2.ndim == 3
    assert np.alltrue(mts1.shape[1:] == mts2.shape[1:])
    idx_i, idx_j = np.mgrid[0:mts1.shape[0], 0:mts2.shape[0]]
    return lpq_switch(mts1[idx_i] - mts2[idx_j], p=p, q=q)
