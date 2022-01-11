import pytest

import numpy as np
import lpqtree.lpqpydist as lpqdist


# testing values
a = np.array([[[2.0, 2.0, -2.0, 2.0],
               [0.0, 0.0, 0.0, 0.0],
               [1.0, 0.0, 0.0, 0.0]],
              [[-4.0, 0.0, 0.0, 0.0],
               [0.0, -1.0, 1.0, 0.0],
               [1.0, -2.0, 2.0, 4.0]]])
l1_a = np.array([[8.0, 0.0, 1.0], [4.0, 2.0, 9.0]])
l2_a = np.array([[4.0, 0.0, 1.0], [4.0, np.sqrt(2.0), 5.0]])

# scaling factor
sc = np.pi


def test_l1():
    assert np.allclose(lpqdist.l1(a), l1_a), "L1 test failed"
    assert np.allclose(lpqdist.l1(a*0.0), l1_a*0.0), "L1 zero failed"
    assert np.allclose(lpqdist.l1(a*sc), l1_a*sc), "L1 scaling failed"
    assert np.allclose(lpqdist.l1(-a), l1_a), "L1 negative failed"
    assert np.allclose(lpqdist.l1(a), np.linalg.norm(a, 1, axis=-1)), "L1 linalg failed"


def test_l2():
    assert np.allclose(lpqdist.l2(a), l2_a), "L2 test failed"
    assert np.allclose(lpqdist.l2(a*0.0), l2_a*0.0), "L2 zero failed"
    assert np.allclose(lpqdist.l2(a*sc), l2_a*sc), "L2 scaling failed"
    assert np.allclose(lpqdist.l2(-a), l2_a), "L1 negative failed"
    assert np.allclose(lpqdist.l2(a), np.linalg.norm(a, 2, axis=-1)), "L1 linalg failed"


def test_lp():
    assert np.allclose(lpqdist.lp(a, p=1), lpqdist.l1(a)), "Lp, p=1 test failed"
    assert np.allclose(lpqdist.lp(a*0.0, p=1), lpqdist.l1(a*0.0)), "Lp, p=1 zero failed"
    assert np.allclose(lpqdist.lp(a*sc, p=1), lpqdist.l1(a*sc)), "Lp, p=1 scaling failed"
    assert np.allclose(lpqdist.lp(-a, p=1), lpqdist.l1(a)), "Lp, p=1 negative failed"

    assert np.allclose(lpqdist.lp(a, p=2), lpqdist.l2(a)), "Lp, p=2 test failed"
    assert np.allclose(lpqdist.lp(a*0.0, p=2), lpqdist.l2(a*0.0)), "Lp, p=2 zero failed"
    assert np.allclose(lpqdist.lp(a*sc, p=2), lpqdist.l2(a*sc)), "Lp, p=2 scaling failed"
    assert np.allclose(lpqdist.lp(-a, p=2), lpqdist.l2(a)), "Lp, p=2 negative failed"

    for p in range(1, 100):
        linalg_res = np.linalg.norm(a, p, axis=-1)
        assert np.allclose(lpqdist.lp(a, p=p), linalg_res), "Lp linalg test failed"
        assert np.allclose(lpqdist.lp(a*0.0, p=p), linalg_res*0.0), "Lp linalg zero failed"
        assert np.allclose(lpqdist.lp(a*sc, p=p), linalg_res*sc), "Lp linalg scaling failed"
        assert np.allclose(lpqdist.lp(-a, p=p), linalg_res), "Lp linalg negative failed"


def test_l11():
    l11_res = lpqdist.l1(lpqdist.l1(a))
    assert np.allclose(lpqdist.l11(a), l11_res), "L11 test failed"
    assert np.allclose(lpqdist.l11(a*0.0), l11_res*0.0), "L11 zero failed"
    assert np.allclose(lpqdist.l11(a*sc), l11_res*sc), "L11 scaling failed"
    assert np.allclose(lpqdist.l11(-a), l11_res), "L11 negative failed"


def test_l12():
    l12_res = lpqdist.l2(lpqdist.l1(a))
    assert np.allclose(lpqdist.l12(a), l12_res), "L12 test failed"
    assert np.allclose(lpqdist.l12(a*0.0), l12_res*0.0), "L12 zero failed"
    assert np.allclose(lpqdist.l12(a*sc), l12_res*sc), "L12 scaling failed"
    assert np.allclose(lpqdist.l12(-a), l12_res), "L12 negative failed"


def test_l21():
    l21_res = lpqdist.l1(lpqdist.l2(a))
    assert np.allclose(lpqdist.l21(a), l21_res), "L21 test failed"
    assert np.allclose(lpqdist.l21(a*0.0), l21_res*0.0), "L21 zero failed"
    assert np.allclose(lpqdist.l21(a*sc), l21_res*sc), "L21 scaling failed"
    assert np.allclose(lpqdist.l21(-a), l21_res), "L21 negative failed"


def test_l22():
    l22_res = lpqdist.l2(lpqdist.l2(a))
    assert np.allclose(lpqdist.l22(a), l22_res), "L22 test failed"
    assert np.allclose(lpqdist.l22(a*0.0), l22_res*0.0), "L22 zero failed"
    assert np.allclose(lpqdist.l22(a*sc), l22_res*sc), "L22 scaling failed"
    assert np.allclose(lpqdist.l22(-a), l22_res), "L22 negative failed"


def test_lp1():
    assert np.allclose(lpqdist.lp1(a, p=1), lpqdist.l11(a)), "Lp1, p=1 test failed"
    assert np.allclose(lpqdist.lp1(a, p=2), lpqdist.l21(a)), "Lp1, p=2 test failed"

    for p in range(1, 100):
        linalg_res = np.linalg.norm(np.linalg.norm(a, p, axis=-1), 1, axis=-1)
        assert np.allclose(lpqdist.lp1(a, p=p), linalg_res), "Lp1 linalg test failed"
        assert np.allclose(lpqdist.lp1(a*0.0, p=p), linalg_res*0.0), "Lp1 linalg zero failed"
        assert np.allclose(lpqdist.lp1(a*sc, p=p), linalg_res*sc), "Lp1 linalg scaling failed"
        assert np.allclose(lpqdist.lp1(-a, p=p), linalg_res), "Lp1 linalg negative failed"


def test_lp2():
    assert np.allclose(lpqdist.lp2(a, p=1), lpqdist.l12(a)), "Lp2, p=1 test failed"
    assert np.allclose(lpqdist.lp2(a, p=2), lpqdist.l22(a)), "Lp2, p=2 test failed"

    for p in range(1, 100):
        linalg_res = np.linalg.norm(np.linalg.norm(a, p, axis=-1), 2, axis=-1)
        assert np.allclose(lpqdist.lp2(a, p=p), linalg_res), "Lp2 linalg test failed"
        assert np.allclose(lpqdist.lp2(a * 0.0, p=p), linalg_res * 0.0), "Lp2 linalg zero failed"
        assert np.allclose(lpqdist.lp2(a * sc, p=p), linalg_res * sc), "Lp2 linalg scaling failed"
        assert np.allclose(lpqdist.lp2(-a, p=p), linalg_res), "Lp2 linalg negative failed"


def test_l1q():
    assert np.allclose(lpqdist.l1q(a, q=1), lpqdist.l1(lpqdist.l1(a))), "L1q, q=1 test failed"
    assert np.allclose(lpqdist.l1q(a, q=2), lpqdist.l12(a)), "L1q, q=2 test failed"

    for q in range(1, 100):
        linalg_res = np.linalg.norm(np.linalg.norm(a, 1, axis=-1), q, axis=-1)
        assert np.allclose(lpqdist.l1q(a, q=q), linalg_res), "L1q linalg test failed"
        assert np.allclose(lpqdist.l1q(a*0.0, q=q), linalg_res*0.0), "L1q linalg zero failed"
        assert np.allclose(lpqdist.l1q(a*sc, q=q), linalg_res*sc), "L1q linalg scaling failed"
        assert np.allclose(lpqdist.l1q(-a, q=q), linalg_res), "L1q linalg negative failed"


def test_l2q():
    assert np.allclose(lpqdist.l2q(a, q=1), lpqdist.l1(lpqdist.l2(a))), "L2q, q=2 test failed"
    assert np.allclose(lpqdist.l2q(a, q=2), lpqdist.l22(a)), "L2q, q=2 test failed"

    for q in range(1, 100):
        linalg_res = np.linalg.norm(np.linalg.norm(a, 2, axis=-1), q, axis=-1)
        assert np.allclose(lpqdist.l2q(a, q=q), linalg_res), "L2q linalg test failed"
        assert np.allclose(lpqdist.l2q(a*0.0, q=q), linalg_res*0.0), "L2q linalg zero failed"
        assert np.allclose(lpqdist.l2q(a*sc, q=q), linalg_res*sc), "L2q linalg scaling failed"
        assert np.allclose(lpqdist.l2q(-a, q=q), linalg_res), "L2q linalg negative failed"


def test_lpq():
    assert np.allclose(lpqdist.lpq(a, p=2, q=1), lpqdist.l21(a)), "Lpq, p=2 q=1 test failed"
    assert np.allclose(lpqdist.lpq(a, p=1, q=1), lpqdist.l1(lpqdist.l1(a))), "Lpq, p=1 q=1 test failed"
    assert np.allclose(lpqdist.lpq(a, p=2, q=2), lpqdist.l2(lpqdist.l2(a))), "Lpq, p=2 q=2 test failed"

    for p in range(1, 100):
        for q in range(1, 100):
            linalg_res = np.linalg.norm(np.linalg.norm(a, p, axis=-1), q, axis=-1)
            assert np.allclose(lpqdist.lpq(a, p=p, q=q), linalg_res), "Lp linalg test failed"
            assert np.allclose(lpqdist.lpq(a*0.0, p=p, q=q), linalg_res*0.0), "Lp linalg zero failed"
            assert np.allclose(lpqdist.lpq(a*sc, p=p, q=q), linalg_res*sc), "Lp linalg scaling failed"
            assert np.allclose(lpqdist.lpq(-a, p=p, q=q), linalg_res), "Lp linalg negative failed"


def test_l1m():
    assert np.allclose(lpqdist.l1m(a), lpqdist.l11(a)/a.shape[-2]), "L2m test failed"
    assert np.allclose(lpqdist.l1m(a*0.0), lpqdist.l1m(a)*0.0), "L2m zero failed"
    assert np.allclose(lpqdist.l1m(a*sc), lpqdist.l1m(a)*sc), "L2m scaling failed"
    assert np.allclose(lpqdist.l1m(-a), lpqdist.l1m(a)), "L2m zero failed"


def test_l2m():
    assert np.allclose(lpqdist.l2m(a), lpqdist.l21(a)/a.shape[-2]), "L2m test failed"
    assert np.allclose(lpqdist.l2m(a*0.0), lpqdist.l2m(a)*0.0), "L2m zero failed"
    assert np.allclose(lpqdist.l2m(a*sc), lpqdist.l2m(a)*sc), "L2m scaling failed"
    assert np.allclose(lpqdist.l2m(-a), lpqdist.l2m(a)), "L2m zero failed"


def test_lpm():
    assert np.allclose(lpqdist.lpm(a, p=2), lpqdist.l21(a)/a.shape[-2]), "Lpm, p=2 test failed"
    assert np.allclose(lpqdist.lpm(a, p=1), lpqdist.l1(lpqdist.l1(a))/a.shape[-2]), "Lpm, p=1 test failed"

    for p in range(1, 100):
        linalg_res = np.mean(np.linalg.norm(a, p, axis=-1), axis=-1)
        assert np.allclose(lpqdist.lpm(a, p=p), linalg_res), "Lpm linalg test failed"
        assert np.allclose(lpqdist.lpm(a*0.0, p=p), linalg_res*0.0), "Lpm linalg zero failed"
        assert np.allclose(lpqdist.lpm(a*sc, p=p), linalg_res*sc), "Lpm linalg scaling failed"
        assert np.allclose(lpqdist.lpm(-a, p=p), linalg_res), "Lpm linalg negative failed"


def test_lpq_switch():
    assert np.allclose(lpqdist.lpq_switch(a, p=1, q=1), lpqdist.l11(a)), "Lpq_switch, p=1 q=1 test failed"
    assert np.allclose(lpqdist.lpq_switch(a, p=1, q=2), lpqdist.l12(a)), "Lpq_switch, p=1 q=2 test failed"
    assert np.allclose(lpqdist.lpq_switch(a, p=2, q=1), lpqdist.l21(a)), "Lpq_switch, p=2 q=1 test failed"
    assert np.allclose(lpqdist.lpq_switch(a, p=2, q=2), lpqdist.l22(a)), "Lpq_switch, p=2 q=2 test failed"

    assert np.allclose(lpqdist.lpq_switch(a, p=1, q="m"), lpqdist.l1m(a)), "Lpq_switch, p=1 mean test failed"
    assert np.allclose(lpqdist.lpq_switch(a, p=2, q="m"), lpqdist.l2m(a)), "Lpq_switch, p=2 mean test failed"

    for p in range(1, 100):
        assert np.allclose(lpqdist.lpq_switch(a, p=p, q="m"), lpqdist.lpm(a, p=p)), "Lpq_switch, mean test failed"
        assert np.allclose(lpqdist.lpq_switch(a, p=p, q=1), lpqdist.lp1(a, p=p)), "Lpq_switch, p=1 test failed"
        assert np.allclose(lpqdist.lpq_switch(a, p=p, q=2), lpqdist.lp2(a, p=p)), "Lpq_switch, p=2 test failed"

    for q in range(1, 100):
        assert np.allclose(lpqdist.lpq_switch(a, p=1, q=q), lpqdist.l1q(a, q=q)), "Lpq_switch, q=1 test failed"
        assert np.allclose(lpqdist.lpq_switch(a, p=2, q=q), lpqdist.l2q(a, q=q)), "Lpq_switch, q=2 test failed"

    for p in range(1, 100):
        for q in range(1, 100):
            lpq_res = lpqdist.lpq(a, p=p, q=q)
            assert np.allclose(lpqdist.lpq_switch(a, p=p, q=q), lpq_res), "Lpq_switch test failed"
            assert np.allclose(lpqdist.lpq_switch(a*0.0, p=p, q=q), lpq_res*0.0), "Lpq_switch zero failed"
            assert np.allclose(lpqdist.lpq_switch(a*sc, p=p, q=q), lpq_res*sc), "Lpq_switch scaling failed"
            assert np.allclose(lpqdist.lpq_switch(-a, p=p, q=q), lpq_res), "Lpq_switch negative failed"


def test_lpq_str_switch():
    for p in range(1, 10):
        norm_str = "l" + str(p) + "m"
        assert np.allclose(lpqdist.lpq_str_switch(a, norm=norm_str), lpqdist.lpm(a, p=p)), "Lpm_switch test failed"
        for q in range(1, 10):
            norm_str = "l" + str(p) + str(q)
            lpq_res = lpqdist.lpq(a, p=p, q=q)
            assert np.allclose(lpqdist.lpq_str_switch(a, norm=norm_str), lpq_res), "Lpq_switch test failed"
            assert np.allclose(lpqdist.lpq_str_switch(a*0.0, norm=norm_str), lpq_res*0.0), "Lpq_switch zero failed"
            assert np.allclose(lpqdist.lpq_str_switch(a*sc, norm=norm_str), lpq_res*sc), "Lpq_switch scaling failed"
            assert np.allclose(lpqdist.lpq_str_switch(-a, norm=norm_str), lpq_res), "Lpq_switch negative failed"
