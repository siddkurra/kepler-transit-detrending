import numpy as np

from filters import split_gaps


def transit_shape(star, f):
    """How much the filter hurt the known planet + how flat the rest got."""
    ph = star.phase()
    D = star.dur
    oot = np.abs(ph) > 1.5 * D
    intr = np.abs(ph) < D / 4
    base = np.nanmedian(f[oot])
    depth = base - np.nanmean(f[intr])
    w = max(0.15, D)
    post = np.nanmean(f[(ph > 0.6 * D) & (ph < 0.6 * D + w)]) - base
    pre = np.nanmean(f[(ph < -0.6 * D) & (ph > -0.6 * D - w)]) - base
    edge = _edges(star.t) & oot
    return dict(rms=np.nanstd(f[oot]) * 1e6,
                cdpp=_cdpp(star.t[oot], f[oot], D) * 1e6,
                kept=depth / star.depth,
                asym=(post - pre) * 1e6,
                edge=np.nanstd(f[edge]) * 1e6)


def _edges(t, width=0.5):
    m = np.zeros(len(t), bool)
    for idx in split_gaps(t)[0]:
        m[idx] = (t[idx] - t[idx[0]] < width) | (t[idx[-1]] - t[idx] < width)
    return m


def _cdpp(t, f, dur):
    # scatter of a running mean over one transit duration (what actually matters
    # for detecting a transit of that length, not the per-point rms)
    dt = np.median(np.diff(t))
    n = max(1, int(round(dur / dt)))
    x = f[np.isfinite(f)] - np.nanmedian(f)
    if len(x) < 3 * n:
        return np.nan
    c = np.convolve(x, np.ones(n) / n, mode="valid")
    return 1.4826 * np.median(np.abs(c - np.median(c)))


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z**2 / n
    mid = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return mid - half, mid + half


def mcnemar(a, b):
    """paired test on the same injections. a, b: bool hit arrays. returns (only_a, only_b, p)
    exact two-sided binomial on the discordant pairs."""
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    if n10 + n01 == 0:
        return n10, n01, 1.0
    return n10, n01, binomtest(n10, n10 + n01, 0.5).pvalue
