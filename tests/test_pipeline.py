import numpy as np

import inject
import search
from metrics import wilson

DT = 29.4 / 60 / 24


def test_trapezoid_depth_and_width():
    pl = inject.Planet(P=3.0, t0=1.0, depth=1e-3, dur=0.12, b=0.0)
    t = np.linspace(0, 3, 30001)
    m = inject.trapezoid(t, pl)
    assert np.isclose(m.min(), 1 - 1e-3)
    assert abs((m < 1).sum() * (t[1] - t[0]) - 0.12) < 2e-3


def test_duration_scaling():
    assert abs(inject.duration(365.25, 0) * 24 - 13) < 1e-9
    assert inject.duration(3, 0.5) < inject.duration(3, 0)


def test_scramble_keeps_samples_and_inverts():
    t = np.arange(0, 60, DT)
    f = 1 + 1e-3 * np.random.default_rng(1).normal(size=len(t))
    ts, fs = inject.scramble(t, f, seed=3)
    assert len(ts) == len(t) and np.all(np.diff(ts) > 0)
    assert np.allclose(np.sort(2 - fs), np.sort(f))


def test_bls_finds_injected_planet_in_white_noise():
    rng = np.random.default_rng(0)
    t = np.arange(131.5, 442, DT)
    pl = inject.Planet(P=4.3217, t0=133.0, depth=1e-3, dur=0.12, b=0)
    f = inject.inject(t, 1 + 2e-4 * rng.normal(size=len(t)), pl)
    h = search.bls(t, f)
    assert search.matches(h.P, pl.P) == 1 and h.sde > 10


def test_matches_harmonics():
    assert search.matches(2.0, 2.0) == 1
    assert search.matches(1.0, 2.0) == 0.5
    assert search.matches(4.0, 2.0) == 2
    assert search.matches(3.0, 2.0) == 0


def test_wilson_contains_p():
    lo, hi = wilson(30, 100)
    assert lo < 0.3 < hi and 0 < lo and hi < 1


def test_mcnemar():
    from metrics import mcnemar
    a = np.array([1] * 12 + [0] * 8, bool)
    b = np.array([0] * 12 + [0] * 8, bool)
    n10, n01, p = mcnemar(a, b)
    assert (n10, n01) == (12, 0) and p < 0.001
    assert mcnemar(a, a)[2] == 1.0


def test_box_retention_matches_rc_theory():
    # causal rc, box of length D, mid-transit retention ~ exp(-pi*fc*D) (class project result)
    from autocut import box_retention
    dt = 29.4 / 60 / 24
    for fc in (0.3, 1.0):
        r = box_retention({"method": "rc"}, fc, 0.1, dt)
        assert abs(r - np.exp(-np.pi * fc * 0.1)) < 0.05


def test_choose_fc_prefers_aggressive_on_red_noise():
    from autocut import choose_fc
    rng = np.random.default_rng(0)
    t = np.arange(0, 30, 29.4 / 60 / 24)
    red = np.cumsum(rng.normal(size=len(t))) * 2e-4
    fc_red, _ = choose_fc(t, 1 + red + 1e-4 * rng.normal(size=len(t)),
                          {"method": "butter", "order": 2, "two_way": True})
    fc_white, _ = choose_fc(t, 1 + 1e-4 * rng.normal(size=len(t)),
                            {"method": "butter", "order": 2, "two_way": True})
    assert fc_red > fc_white


def test_fast_mes_agrees_with_explicit_fold():
    rng = np.random.default_rng(0)
    t = np.arange(131.5, 300, DT)
    pl = inject.Planet(P=4.3217, t0=133.0, depth=6e-4, dur=0.12, b=0)
    f = inject.inject(t, 1 + 1e-4 * rng.normal(size=len(t)), pl)
    per = search.period_grid(t)[::10]
    a = search.mes_loop(t, f, per, durs=(0.12,))
    b = search.mes(t, f, per, durs=(0.12,))
    assert search.matches(a.P, pl.P) == 1 and search.matches(b.P, pl.P) == 1
    # bls fold tests finer phase offsets than 1-cadence bins, so it can only be >= the loop
    assert 0.95 * a.depth <= b.depth <= 1.25 * a.depth


def test_whitening_helps_in_red_noise():
    rng = np.random.default_rng(1)
    t = np.arange(131.5, 300, DT)
    r = np.zeros(len(t))
    e = rng.normal(size=len(t))
    for i in range(1, len(t)):
        r[i] = 0.9 * r[i - 1] + e[i]
    f = 1 + 1.5e-4 * r / r.std() + 0.5e-4 * rng.normal(size=len(t))
    pl = inject.Planet(P=4.3217, t0=133.0, depth=2.5e-4, dur=0.12, b=0)
    f = inject.inject(t, f, pl)
    w = search.mes(t, f, whiten=True)
    nw = search.mes(t, f, whiten=False)
    assert search.matches(w.P, pl.P) == 1 and w.sde > nw.sde


def test_paired_refuses_empty_pairing():
    import pandas as pd
    import pytest as _pt
    from evaluate import paired
    a = pd.Series([True, False], index=[1, 2])
    b = pd.Series([True, False], index=[3, 4])
    with _pt.raises(ValueError):
        paired(a, b)
