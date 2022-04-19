# Etienne St-Onge

import numpy as np


# Vector norm operator
def l1(vts, axis=-1, keepdims=False):
    """Compute the L1 / manhattan norm, along the given axis"""
    return np.sum(np.abs(vts), axis=axis, keepdims=keepdims)


def l2(vts, axis=-1, keepdims=False):
    """Compute the L2 / euclidean norm, along the given axis"""
    return np.sqrt(np.sum(np.square(vts), axis=axis, keepdims=keepdims))


def lp(vts, p, axis=-1, keepdims=False):
    """Compute the general Lp norm, along the given axis"""
    inv_p = 1.0/p
    return np.sum(np.abs(vts**p), axis=axis, keepdims=keepdims)**inv_p


# Matrix norm operator, along the two last axis
def l11(mts):
    """Compute the L11 norm, along the two last axis (equivalent to L1)"""
    return np.sum(l1(mts, axis=-1, keepdims=False), axis=-1, keepdims=False)


def l12(mts):
    """Compute the L12 norm, along the two last axis"""
    return l2(l1(mts, axis=-1, keepdims=False), axis=-1, keepdims=False)


def l21(mts):
    """Compute the L21 norm, along the two last axis"""
    return np.sum(l2(mts, axis=-1, keepdims=False), axis=-1, keepdims=False)


def l22(mts):
    """Compute the L22 norm, along the two last axis (equivalent to L2)"""
    return np.sqrt(np.sum(np.sum(np.square(mts), axis=-1, keepdims=False), axis=-1, keepdims=False))


def lp1(mts, p):
    """Compute the Lp1 norm, along the two last axis"""
    return np.sum(lp(mts, p=p, axis=-1, keepdims=False), axis=-1, keepdims=False)


def lp2(mts, p):
    """Compute the Lp2 norm, along the two last axis"""
    return l2(lp(mts, p=p, axis=-1, keepdims=False), axis=-1, keepdims=False)


def l1q(mts, q):
    """Compute the L1q norm, along the two last axis"""
    inv_q = 1.0/q
    return np.sum(l1(mts, axis=-1, keepdims=False)**q, axis=-1, keepdims=False)**inv_q


def l2q(mts, q):
    """Compute the L2q norm, along the two last axis"""
    inv_q = 1.0/q
    return np.sum(l2(mts, axis=-1, keepdims=False)**q, axis=-1, keepdims=False)**inv_q


def lpq(mts, p, q):
    """Compute the general Lpq norm, along the two last axis"""
    inv_q = 1.0/q
    return np.sum(lp(mts, p=p, axis=-1, keepdims=False)**q, axis=-1, keepdims=False)**inv_q


# Matrix norm operator with mean instead of sum (homogeneous to lp1)
def l1m(mts):
    """Compute the average L1 norm, along the two last axis"""
    return np.mean(l1(mts, axis=-1, keepdims=False), axis=-1, keepdims=False)


def l2m(mts):
    """Compute the average L2 norm, along the two last axis"""
    return np.mean(l2(mts, axis=-1, keepdims=False), axis=-1, keepdims=False)


def lpm(mts, p):
    """Compute the average Lp norm, along the two last axis"""
    return np.mean(lp(mts, p=p, axis=-1, keepdims=False), axis=-1, keepdims=False)


# Matrix norm switch
def lpq_switch(mts, p, q):
    """
    Compute the general Lpq norm

    Parameters
    ----------
    mts : numpy array
        list of matrices or a single matrix
    p: int
        first norm applied along the last axis
    q : int
        second norm applied along the second last axis
        "m" can be given instead of q=1 to compute the mean instead of the L1 sum.

    Returns
    -------
    res : numpy array (float)
        Resulting Lpq norm
    """
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
    """
    Compute the general Lpq norm

    Parameters
    ----------
    mts : numpy array
        list of matrices or a single matrix
    norm: string
        String of the "Lpq" name, where p and q are integer > 0,
        "m" can be given instead of q=1 to compute the mean instead of the L1 sum.

    Returns
    -------
    res : numpy array (float)
        Resulting Lpq norm
    """
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
    """
    For each element (mts1[i], mts2[i]) compute the Lpq distance

    Parameters
    ----------
    mts1 : numpy array (a x m x n)
        list of matrices
    mts2 : numpy array (a x m x n)
        list of matrices
    p: int
        first norm applied along the last axis
    q : int
        second norm applied along the second last axis
        "m" can be given instead of q=1 to compute the mean instead of the L1 sum.

    Returns
    -------
    res : numpy array (a x 1)
        Resulting Lpq distance each element (mts1[i], mts2[j])
    """
    assert mts1.ndim == 3
    assert mts2.ndim == 3
    assert np.alltrue(mts1.shape[1:] == mts2.shape[1:])

    return lpq_switch(mts1 - mts2, p=p, q=q)


# Compute the distance between all pairs (Warning about memory)
def lpq_allpairs(mts1, mts2, p, q):
    """
    For every pair (mts1[i], mts2[j]) compute the Lpq distance
    (Faster but requires more memory compared to the iterative)

    Parameters
    ----------
    mts1 : numpy array (a x m x n)
        list of matrices
    mts2 : numpy array (b x m x n)
        list of matrices
    p: int
        first norm applied along the last axis
    q : int
        second norm applied along the second last axis
        "m" can be given instead of q=1 to compute the mean instead of the L1 sum.

    Returns
    -------
    res : numpy array (a x b)
        Resulting Lpq distance for  every pair (mts1[i], mts2[j])
    """
    assert mts1.ndim == 3
    assert mts2.ndim == 3
    assert np.alltrue(mts1.shape[1:] == mts2.shape[1:])

    diff = mts1[:, None, ...] - mts2
    return lpq_switch(diff, p=p, q=q)


def lpq_allpairs_iter(mts1, mts2, p, q):
    """
    For every pair (mts1[i], mts2[j]) compute the Lpq distance
    (Iterative version, slower but does not need as much memory)

    Parameters
    ----------
    mts1 : numpy array (a x m x n)
        list of matrices
    mts2 : numpy array (b x m x n)
        list of matrices
    p: int
        first norm applied along the last axis
    q : int
        second norm applied along the second last axis
        "m" can be given instead of q=1 to compute the mean instead of the L1 sum.

    Returns
    -------
    res : numpy array (a x b)
        Resulting Lpq distance for  every pair (mts1[i], mts2[j])
    """
    assert mts1.ndim == 3
    assert mts2.ndim == 3
    assert np.alltrue(mts1.shape[1:] == mts2.shape[1:])

    result = np.zeros((len(mts1), len(mts2)), dtype=mts1.dtype)
    for j in range(len(mts2)):
        result[:, j] = lpq_switch(mts1 - mts2[j], p=p, q=q)
    return result
