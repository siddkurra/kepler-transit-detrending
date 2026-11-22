import numpy as np
import pytest
from scipy import signal

import filters as F

DT = 29.4 / 60 / 24


def test_rc_matches_lfilter():
    x = np.random.default_rng(0).normal(size=3000)
    for fc in (0.1, 1.0, 5.0):
        b, a = F.rc_coeffs(fc, DT)
        ref, _ = signal.lfilter(b, a, x, zi=signal.lfiltic(b, a, [0.0], [x[0]]))
        assert np.max(np.abs(F.rc_highpass(x, fc, DT) - ref)) < 1e-12


def test_rc_kills_dc():
    y = F.rc_highpass(np.full(500, 3.7), 0.3, DT)
    assert np.allclose(y, 0)


def test_rc_pole_inside_unit_circle():
    for fc in (0.01, 1, 20):
        assert 0 < F.rc_alpha(fc, DT) < 1


def test_rc_step_response_matches_analytic():
    fc = 0.3
    rc = 1 / (2 * np.pi * fc)
    t = np.arange(0, 3, DT)
    x = (t >= 1).astype(float)
    y = F.rc_highpass(x, fc, DT)
    ana = np.where(t >= 1, np.exp(-(t - 1) / rc), 0)
    # backward difference: first sample is alpha instead of 1, error ~ T/RC
    assert np.max(np.abs(y - ana)) < 1.5 * DT / rc


def test_rc_gain_at_cutoff():
    fc = 0.2
    b, a = F.rc_coeffs(fc, DT)
    _, h = signal.freqz(b, a, worN=[fc], fs=1 / DT)
    assert abs(abs(h[0]) - 1 / np.sqrt(2)) < 0.01


def test_split_gaps():
    t = np.r_[np.arange(0, 1, DT), np.arange(2, 3, DT)]
    chunks, dt = F.split_gaps(t)
    assert len(chunks) == 2 and np.isclose(dt, DT)


def test_fill_masked_is_linear():
    t = np.arange(10.0)
    x = t.copy()
    x[4:7] = -50
    m = np.zeros(10, bool)
    m[4:7] = True
    assert np.allclose(F.fill_masked(t, x, m), t)


def test_detrend_leaves_masked_transit_untouched():
    t = np.arange(0, 10, DT)
    slow = 0.01 * np.sin(2 * np.pi * t / 7)
    dip = np.where(np.abs(t - 5) < 0.05, -0.01, 0)
    mask = np.abs(t - 5) < 0.08
    out = F.detrend(t, 1 + slow + dip, fc=0.5, two_way=True, mask=mask)
    inside = np.abs(t - 5) < 0.04
    assert np.allclose(out[inside] - 1, -0.01, atol=1e-3)


@pytest.mark.parametrize("two_way", [False, True])
def test_detrend_baseline_is_one(two_way):
    t = np.arange(0, 5, DT)
    out = F.detrend(t, np.full(len(t), 1.0), fc=0.5, two_way=two_way)
    assert np.allclose(out, 1)


@pytest.mark.parametrize("order", [1, 2, 4])
def test_butter_is_3db_at_cutoff(order):
    fc = 0.5
    _, h = signal.sosfreqz(F.butter_sos(fc, DT, order), worN=[fc, 10 * fc, fc / 10], fs=1 / DT)
    assert abs(abs(h[0]) - 1 / np.sqrt(2)) < 1e-6
    assert abs(h[1]) > 0.99
    # rolloff: order N -> about 20N dB per decade below fc
    assert abs(20 * np.log10(abs(h[2])) + 20 * order) < 1


def test_butter_two_way_squares_magnitude():
    fc = 0.5
    t = np.arange(0, 200, DT)
    x = np.sin(2 * np.pi * fc * t)
    out = F.detrend(t, 1 + x, fc=fc, method="butter", order=2, two_way=True) - 1
    mid = slice(len(t) // 4, 3 * len(t) // 4)
    assert abs(np.std(out[mid]) / np.std(x[mid]) - 0.5) < 0.01


def test_butter_kills_dc():
    t = np.arange(0, 5, DT)
    for tw in (False, True):
        assert np.allclose(F.detrend(t, np.full(len(t), 1.0), fc=0.5, method="butter", two_way=tw), 1)


def test_robust_lowpass_equals_rc_when_nothing_to_reject():
    t = np.arange(0, 20, DT)
    x = 1 + 0.01 * np.sin(2 * np.pi * t / 6)
    fc = 0.5
    s = F.robust_lowpass(x, fc, DT, k=1e9)
    y = F.rc_highpass(x, fc, DT)
    # rc_highpass starts from rest, so compare after the start-up transient
    late = t > 5
    assert np.allclose((x - s)[late], y[late], atol=1e-6)


def test_robust_trend_ignores_a_dip():
    rng = np.random.default_rng(0)
    t = np.arange(0, 10, DT)
    x = 1 + 1e-4 * rng.normal(size=len(t))
    dip = np.abs(t - 5) < 0.05
    x[dip] -= 3e-3
    out = F.detrend(t, x, fc=3.0, method="robust_rc", two_way=True)
    plain = F.detrend(t, x, fc=3.0, method="rc", two_way=True)
    kept = 1 - out[dip].mean()
    assert abs(kept - 3e-3) < 2e-4
    assert (1 - plain[dip].mean()) < kept


def test_clip_protects_dip_on_a_steep_trend():
    rng = np.random.default_rng(0)
    t = np.arange(0, 10, DT)
    x = 1 + 0.02 * np.sin(2 * np.pi * t / 5) + 1e-4 * rng.normal(size=len(t))
    dip = np.abs(t - 5.6) < 0.05
    x[dip] -= 2e-3
    plain = F.detrend(t, x, fc=3.0, method="butter", two_way=True)
    clipped = F.detrend(t, x, fc=3.0, method="butter", two_way=True, clip=3.0)
    assert abs((1 - clipped[dip].mean()) - 2e-3) < abs((1 - plain[dip].mean()) - 2e-3)
    assert abs((1 - clipped[dip].mean()) - 2e-3) < 3e-4


def test_odd_pad_is_continuous():
    x = np.linspace(0, 1, 50) ** 2
    xp, n = F.odd_pad(x, 10)
    assert n == 10 and len(xp) == 70
    assert np.allclose(xp[n:-n], x)
    # slope continuous at both ends
    assert np.isclose(xp[n] - xp[n - 1], x[1] - x[0])
    assert np.isclose(xp[-n] - xp[-n - 1], x[-1] - x[-2])


@pytest.mark.parametrize("cfg", [dict(method="rc", two_way=True), dict(method="butter", order=2, two_way=True)])
def test_padding_reduces_edge_error_on_a_ramp(cfg):
    # pure linear ramp: highpass of a ramp settles to a constant, start-up transient is the error
    t = np.arange(0, 6, DT)
    x = 1 + 0.01 * t
    fc = 0.3
    a = F.detrend(t, x, fc=fc, **cfg)
    b = F.detrend(t, x, fc=fc, pad=True, **cfg)
    mid = np.abs(t - 3) < 0.5
    edge = t < 0.5
    err_a = np.std(a[edge] - np.median(a[mid]))
    err_b = np.std(b[edge] - np.median(b[mid]))
    assert err_b < 0.5 * err_a


def test_biweight_ignores_a_dip_and_follows_a_slow_trend():
    rng = np.random.default_rng(0)
    t = np.arange(0, 10, DT)
    x = 1 + 0.01 * np.sin(2 * np.pi * t / 8) + 1e-4 * rng.normal(size=len(t))
    dip = np.abs(t - 5) < 0.05
    x[dip] -= 1e-3
    out = F.detrend(t, x, method="biweight", window=0.5)
    assert np.nanstd(out[~dip]) < 2e-4
    assert (1 - out[dip].mean()) > 0.5e-3
