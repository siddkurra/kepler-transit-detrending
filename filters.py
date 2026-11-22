import numpy as np
from scipy import signal


def rc_alpha(fc, dt):
    rc = 1 / (2 * np.pi * fc)
    return rc / (rc + dt)


def rc_coeffs(fc, dt):
    a = rc_alpha(fc, dt)
    return np.array([a, -a]), np.array([1.0, -a])


def rc_highpass(x, fc, dt):
    # y[n] = a*(y[n-1] + x[n] - x[n-1]), start from rest
    a = rc_alpha(fc, dt)
    y = np.zeros(len(x))
    xp, yp = x[0], 0.0
    for n in range(len(x)):
        y[n] = a * (yp + x[n] - xp)
        xp, yp = x[n], y[n]
    return y


def split_gaps(t, k=5.0):
    dt = np.median(np.diff(t))
    cuts = np.where(np.diff(t) > k * dt)[0] + 1
    return np.split(np.arange(len(t)), cuts), dt


def fill_masked(t, x, mask):
    # straight line across masked points so the filter never sees the dip
    if not mask.any() or mask.all():
        return x
    x = x.copy()
    x[mask] = np.interp(t[mask], t[~mask], x[~mask])
    return x


def butter_sos(fc, dt, order):
    return signal.butter(order, fc, btype="highpass", fs=1 / dt, output="sos")


def robust_lowpass(x, fc, dt, k=3.0, max_hold=12):
    # rc low-pass s[n] = a*s[n-1] + (1-a)*x[n], except when x drops more than k*sigma
    # below the trend: then hold s (a dip shouldn't drag the trend down). max_hold so a
    # real downward trend can't freeze it forever
    a = rc_alpha(fc, dt)
    sig = 1.4826 * np.median(np.abs(np.diff(x))) / np.sqrt(2)
    s = np.empty(len(x))
    s[0] = x[0]
    held = 0
    for n in range(1, len(x)):
        if x[n] - s[n - 1] < -k * sig and held < max_hold:
            s[n] = s[n - 1]
            held += 1
        else:
            s[n] = a * s[n - 1] + (1 - a) * x[n]
            held = 0
    return s


def odd_pad(x, n):
    # point-reflect around the end samples (what filtfilt does): keeps value and slope
    # continuous at the edge so the filter doesnt see a step
    n = min(n, len(x) - 1)
    if n <= 0:
        return x, 0
    return np.r_[2 * x[0] - x[n:0:-1], x, 2 * x[-1] - x[-2:-n - 2:-1]], n


def pad_len(fc, dt, order=1):
    # ~3 time constants of the slowest pole
    return int(np.ceil(3 * order / (2 * np.pi * fc * dt)))


def _trend(x, fc, dt, method, two_way, order=2, k=3.0, max_hold=12, pad=False):
    if method == "none":
        return np.ones(len(x))
    if method == "butter":
        sos = butter_sos(fc, dt, order)
        if two_way:
            n = pad_len(fc, dt, order) if pad else 3 * (2 * len(sos) + 1)
            return x - signal.sosfiltfilt(sos, x - x[0], padlen=min(len(x) - 1, n))
        xp, n = odd_pad(x, pad_len(fc, dt, order)) if pad else (x, 0)
        # start from rest like the rc filter, otherwise the first sample is a huge step
        y = signal.sosfilt(sos, xp - xp[0])
        return x - y[n:len(y) - n]
    if method == "rc":
        xp, n = odd_pad(x, pad_len(fc, dt)) if pad else (x, 0)
        y = rc_highpass(xp, fc, dt)
        if two_way:
            y = rc_highpass(y[::-1], fc, dt)[::-1]
        return x - y[n:len(y) - n]
    if method == "robust_rc":
        s = robust_lowpass(x, fc, dt, k, max_hold)
        if two_way:
            # average fwd + bwd trends, lags cancel to first order
            s = 0.5 * (s + robust_lowpass(x[::-1], fc, dt, k, max_hold)[::-1])
        return s
    raise ValueError(f"unknown method {method}")


def _detrend_once(t, flux, fc, method, two_way, mask, kw):
    out = np.full(len(flux), np.nan)
    chunks, dt = split_gaps(t)
    for idx in chunks:
        if len(idx) < 3:
            continue
        x = flux[idx]
        xf = x if mask is None else fill_masked(t[idx], x, mask[idx])
        if method == "biweight":
            trend = biweight_trend(t[idx], xf, kw["window"])
        else:
            trend = _trend(xf, fc, dt, method, two_way, **kw)
        out[idx] = x - trend + 1
    return out


def biweight_trend(t, x, window):
    # wotan's time-windowed tukey biweight (hippke et al. 2019), the thing to beat
    from wotan import flatten
    _, trend = flatten(t, x, method="biweight", window_length=window, edge_cutoff=0,
                       break_tolerance=window / 2, return_trend=True)
    bad = ~np.isfinite(trend)
    if bad.any() and (~bad).sum() > 1:
        trend[bad] = np.interp(t[bad], t[~bad], trend[~bad])
    return trend


def detrend(t, flux, fc=None, method="rc", two_way=False, mask=None, clip=None, clip_iter=2,
            **kw):
    out = _detrend_once(t, flux, fc, method, two_way, mask, kw)
    if clip is None:
        return out
    # iterative lower-tail sigma clip on the (zero-lag if two_way) residual, then refit
    # with the clipped points filled in. same idea as wotan/lightkurve's iterative clipping
    base = np.zeros(len(t), bool) if mask is None else mask.copy()
    for _ in range(clip_iter):
        r = out - 1
        sig = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r)))
        low = r < -clip * sig
        low = low | np.r_[low[1:], False] | np.r_[False, low[:-1]]
        out = _detrend_once(t, flux, fc, method, two_way, base | low, kw)
    return out
