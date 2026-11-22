from dataclasses import dataclass

import numpy as np
from astropy.timeseries import BoxLeastSquares

PMIN, PMAX = 0.7, 12.0
DURS = np.array([0.06, 0.09, 0.13, 0.18, 0.25])


def period_grid(t, pmin=PMIN, pmax=PMAX, oversample=3):
    # phase drift over the whole baseline has to stay under dur/oversample:
    # base * df * P < dur/os  ->  d(ln f) = dur / (os * base), so uniform in log f
    step = DURS.min() / (oversample * np.ptp(t))
    lnf = np.arange(np.log(1 / pmax), np.log(1 / pmin), step)
    return np.sort(np.exp(-lnf))


@dataclass
class Hit:
    P: float
    t0: float
    dur: float
    sde: float
    depth: float


def bls(t, f, periods=None):
    ok = np.isfinite(f)
    t, f = t[ok], f[ok]
    periods = period_grid(t) if periods is None else periods
    pg = BoxLeastSquares(t, f).power(periods, DURS, objective="snr")
    pw = pg.power
    i = np.argmax(pw)
    sde = (pw[i] - np.mean(pw)) / np.std(pw)
    return Hit(pg.period[i], pg.transit_time[i], pg.duration[i], sde, pg.depth[i])


def matches(P_found, P_true, tol=0.01):
    # count harmonics too since those still point you at the planet, but log which
    for k in (1, 0.5, 2):
        if abs(P_found / (k * P_true) - 1) < tol:
            return k
    return 0


def _grid(t, f):
    dt = np.median(np.diff(t))
    n = np.round((t - t[0]) / dt).astype(int)
    x = np.zeros(n[-1] + 1)
    w = np.zeros(n[-1] + 1)
    ok = np.isfinite(f)
    x[n[ok]] = f[ok] - np.nanmedian(f)
    w[n[ok]] = 1
    return x, w, dt


def ses(t, f, dur, whiten=True, nperseg=512):
    """single event statistic: matched filter for a box of length dur, optionally after
    whitening by the measured noise psd (kepler tps does a wavelet version of this)"""
    from scipy import signal as sg
    x, w, dt = _grid(t, f)
    N = len(x)
    L = max(1, int(round(dur / dt)))
    nfft = 1 << int(np.ceil(np.log2(N + L)))
    if whiten:
        fr, p = sg.welch(x, nperseg=min(nperseg, N), detrend=False)
        p = p / max(w.mean(), 1e-3)
        # floor at the white level (median of the top half of the band). without this,
        # whitening a high-passed series divides by the tiny psd below fc and undoes the
        # high-pass (amplifies spot/edge junk ~1000x). exp07 v1 had that bug
        white = np.median(p[len(p) // 2:])
        psd = np.maximum(np.interp(np.fft.rfftfreq(nfft), fr, p), white)
    else:
        psd = np.ones(nfft // 2 + 1)
    X = np.fft.rfft(x, nfft)
    tmpl = np.zeros(nfft)
    tmpl[:L] = -1.0 / L
    S = np.fft.rfft(tmpl)
    corr = np.fft.irfft(X * np.conj(S) / psd, nfft)[:N]
    # corr[n] = template starting at n; shift so it lines up with the transit centre
    corr = np.roll(corr, L // 2)
    cover = np.convolve(w, np.ones(L) / L, mode="same")
    good = (w > 0) & (cover > 0.75)
    s = np.where(good, corr, 0.0)
    v = s[good]
    sig = 1.4826 * np.median(np.abs(v - np.median(v)))
    return (s - np.median(v) * good) / sig, good, dt


def mes(t, f, periods=None, durs=(0.08, 0.12, 0.16), whiten=True):
    """fold the ses with bls itself: a one-cadence box on -ses with the snr objective
    gives sum(ses)/sqrt(n) per phase, i.e. the mes, at compiled speed"""
    periods = period_grid(t[np.isfinite(f)]) if periods is None else periods
    best = None
    for d in durs:
        s, good, dt = ses(t, f, d, whiten)
        tt = t[0] + np.nonzero(good)[0] * dt
        pg = BoxLeastSquares(tt, -s[good], dy=np.ones(good.sum())).power(periods, dt, objective="snr")
        if best is None:
            best = pg.power.copy()
            t0s, ds = pg.transit_time.copy(), np.full(len(periods), d)
        else:
            up = pg.power > best
            best[up] = pg.power[up]
            t0s[up] = pg.transit_time[up]
            ds[up] = d
    i = np.argmax(best)
    sde = (best[i] - best.mean()) / best.std()
    return Hit(periods[i], t0s[i], ds[i], sde, best[i])


def mes_loop(t, f, periods=None, durs=(0.08, 0.12, 0.16), whiten=True):
    """slow explicit fold, kept as the reference for the equivalence test:
    mes = sum(ses)/sqrt(n) per 1-cadence phase bin, max bin, max over durations."""
    periods = period_grid(t[np.isfinite(f)]) if periods is None else periods
    best = np.full(len(periods), -np.inf)
    arg = np.zeros((len(periods), 2))
    for d in durs:
        s, good, dt = ses(t, f, d, whiten)
        n = np.nonzero(good)[0].astype(float)
        sv = s[good]
        for i, P in enumerate(periods):
            p = P / dt
            k = (n % p).astype(int)
            m = int(np.ceil(p))
            num = np.bincount(k, sv, m)
            cnt = np.bincount(k, None, m)
            ms = num / np.sqrt(np.maximum(cnt, 1))
            j = np.argmax(ms)
            if ms[j] > best[i]:
                best[i] = ms[j]
                arg[i] = (j * dt + t[0], d)
    i = np.argmax(best)
    sde = (best[i] - best.mean()) / best.std()
    return Hit(periods[i], arg[i, 0], arg[i, 1], sde, best[i])


def tls(t, f):
    import contextlib
    import io
    from transitleastsquares import transitleastsquares
    ok = np.isfinite(f)
    with contextlib.redirect_stdout(io.StringIO()):
        r = transitleastsquares(t[ok], f[ok]).power(
            period_min=PMIN, period_max=PMAX, use_threads=1, show_progress_bar=False,
            oversampling_factor=3, duration_grid_step=1.1)
    return Hit(r.period, r.T0, r.duration, r.SDE, 1 - r.depth)


def find(t, f, how="bls"):
    if how == "bls":
        return bls(t, f)
    if how == "mes":
        return mes(t, f, whiten=True)
    if how == "mes_white":
        return mes(t, f, whiten=False)
    if how == "tls":
        return tls(t, f)
    raise ValueError(how)
